"""
OpenVINOPredictor — CPU predictor backend cho ranking scheduler.

Cài đặt cho mode `device="openvino"` trong PrefillModelConfig (xem
vllm/config_predictor.py). Predictor được scheduler gọi qua BA API:

# Sync API (legacy, giữ cho unit test + backward compat):
    obtain_aux_scores(seq_groups) -> List[float]
    Block caller cho tới khi tính xong tất cả score.

# Async API (legacy — dùng bởi _get_opt_cpu_ordered_requests / opt-cpu-warmup):
    submit_async(seq_groups) -> bool       # fire-and-forget, ~µs
    poll_results() -> int                  # check Future done, set scores, ~µs
    is_busy() -> bool                      # query state, ~µs

# Streaming API (NEW — dùng bởi _get_opt_cpu_async_warmup_ordered_requests):
    submit_streaming(seq_groups) -> bool   # luôn accept, dedup nội bộ
    poll_streaming() -> int                # no-op (worker apply trực tiếp)
    has_pending_stream() -> bool           # diagnostic
Continuous queue model — worker thread tự pull queue, apply scores incremental
cross-thread (GIL atomic). Request mới có thể submit anytime, không bị block
khi worker đang chạy mini-batch khác.

# Tại sao có 2 async API song song?
Streaming API ADD-only — KHÔNG modify legacy submit_async/poll_results/is_busy.
Lý do: opt-cpu-warmup (baseline) phải giữ identical behavior cho A/B test.
Hai API có state TÁCH BIỆT (legacy: _pending_future; streaming: _stream_*).
Schedule_type set 1 lần lúc start server → chỉ 1 API path active tại 1 thời
điểm, 2 bộ state coexist không tương tác.

Cả 2 async API share `_executor` (ThreadPoolExecutor 1 worker). Event loop của
vLLM (asyncio) KHÔNG bị block — `await execute_model_async()` và các task
khác tiếp tục chạy mượt trong khi predictor compute trên thread riêng.

# Tại sao bypass AUXLLMEngine?
AUXLLMEngine là một full vLLM engine wrap predictor như generation model
(max_tokens=1) chạy trên GPU. Để chạy trên CPU, cách trực tiếp là:
  1. Load PyTorch checkpoint
  2. Convert sang OpenVINO IR in-memory (openvino.convert_model)
  3. Compile cho CPU với precision/threads tuỳ chỉnh
  4. Inference bằng compiled_model() — trả numpy array
Đơn giản hơn nhiều so với CPUExecutor pipeline của vLLM.

# Tại sao raw OpenVINO API thay vì optimum.intel.openvino?
optimum-intel mới yêu cầu nncf >= 2.13 → cần torch >= 2.3 (`torch.uint16`).
vllm-ltr cố định torch 2.2.1, nên optimum-intel import bị gãy.
Raw OpenVINO API chỉ cần `openvino` package → không có conflict.
"""
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


class _LogitsOnlyWrapper(torch.nn.Module):
    """Wrap HF SequenceClassification model để forward chỉ trả logits.

    HF model trả `SequenceClassifierOutputWithPast` (dataclass có cả Tensor lẫn
    Tuple). TorchScript tracing trong openvino.convert_model không xử lý được
    mixed dataclass output — phải wrap để chỉ expose `logits` tensor.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


class OpenVINOPredictor:
    """CPU predictor dùng OpenVINO runtime với async dispatch.

    Public interface:
      Sync (legacy):
        obtain_aux_scores(seq_groups) -> List[float]
            Block caller. Dùng cho unit test hoặc fallback.

      Async (recommended cho scheduler):
        submit_async(seq_groups) -> bool
            Submit batch để worker thread score, return ngay (~µs).
            Returns False nếu predictor đang busy (caller skip batch).
        poll_results() -> int
            Non-blocking check. Nếu Future done, set scores vào seq_groups
            và return số seq_group được score. Nếu chưa done, return 0.
        is_busy() -> bool
            True nếu đang có batch chưa xong.
        shutdown()
            Dọn ThreadPoolExecutor khi server stop.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_name: str,
        num_labels: int,
        max_length: int,
        max_batch_size: int,
        num_threads: int = 32,
        inference_precision: str = "f16",
        async_mode: bool = True,
    ):
        """Load PyTorch checkpoint, convert sang OV IR, compile cho CPU.

        Args:
            model_path: đường dẫn checkpoint OPT-125m fine-tuned (HF format).
            tokenizer_name: tokenizer base (vd "facebook/opt-125m").
            num_labels: số label của head — typically 1 cho ranking, >1 cho class.
            max_length: max prompt length truncate khi tokenize.
            max_batch_size: chia prompts thành mini-batch khi inference.
            num_threads: số CPU thread cho OV inference (= INFERENCE_NUM_THREADS).
            inference_precision: f32/f16/bf16 hint cho activation compute
                (f16 nhanh hơn f32 ~1.3-1.7x trên Cascade Lake với drift
                < 5e-2).
            async_mode: True (default) → tạo ThreadPoolExecutor cho submit_async/
                poll_results. False → chỉ obtain_aux_scores (sync) khả dụng.
        """
        # Lazy import để không đụng openvino package nếu user dùng device="auto"
        import openvino as ov
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info(
            f"Loading OpenVINO predictor from {model_path} "
            f"(threads={num_threads}, precision={inference_precision}, "
            f"async_mode={async_mode})"
        )
        t0 = time.time()

        # Bước 1: load PyTorch model (eval để disable dropout)
        pt_model = AutoModelForSequenceClassification.from_pretrained(model_path).eval()
        wrapped = _LogitsOnlyWrapper(pt_model).eval()

        # Bước 2: load tokenizer (cùng vocab với checkpoint, dùng CPU)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # Pythia/GPT-NeoX tokenizer không có pad_token mặc định —
        # `padding=True` ở Bước 3 (sample input) và mọi tokenize call sau
        # này sẽ raise. Reuse eos_token làm pad. OPT/BERT no-op.
        # Lưu ý: pt_model.config.pad_token_id đã được sync trong training
        # (xem prefill_predictor.py) và bake vào IR graph ở Bước 4.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        self.max_batch_size = max_batch_size

        # Bước 3: tạo example input để OV trace shape graph.
        # Shape động (batch, seq_len) — OV sẽ infer shape động từ trace này.
        sample = self.tokenizer(
            ["dummy"], padding=True, truncation=True,
            max_length=16, return_tensors="pt",
        )
        example_input = (sample["input_ids"], sample["attention_mask"])

        # Bước 4: convert PyTorch → OV IR (in-memory, không ghi đĩa)
        ov_model = ov.convert_model(wrapped, example_input=example_input)

        # Bước 5: compile cho CPU. Các config được set theo Plan:
        #   INFERENCE_NUM_THREADS: giới hạn thread để không cướp CPU của
        #     main serving process (tokenizer, asyncio loop, etc.).
        #   INFERENCE_PRECISION_HINT: weight format khi compute (f16 giảm
        #     bandwidth memory, dequant dynamic về f32 khi compute).
        #   PERFORMANCE_HINT: LATENCY tối ưu cho 1 batch nhỏ vs THROUGHPUT
        #     (multi-stream). Predictor được gọi mỗi tick → LATENCY phù hợp.
        core = ov.Core()
        self.compiled_model = core.compile_model(ov_model, "CPU", {
            "INFERENCE_NUM_THREADS": str(num_threads),
            "INFERENCE_PRECISION_HINT": inference_precision,
            "PERFORMANCE_HINT": "LATENCY",
        })

        # Bước 6: warmup — compile graph + first-call có overhead lớn.
        # Chạy 1 dummy forward để dồn cost vào init time, không phải runtime.
        with torch.no_grad():
            dummy = self.tokenizer(
                ["warmup"] * 2, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            )
            self.compiled_model([dummy["input_ids"].numpy(),
                                 dummy["attention_mask"].numpy()])

        # === [opt-cpu] Async infrastructure ===
        # ThreadPoolExecutor với 1 worker — predictor là bottleneck, queueing
        # nhiều batch không tăng throughput (hardware compute là constant).
        # Worker thread dùng thread_name_prefix để dễ debug bằng htop / py-spy.
        self.async_mode = async_mode
        if async_mode:
            self._executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="ov-predictor"
            )
        else:
            self._executor = None
        # State: tối đa 1 future pending tại 1 thời điểm.
        # _pending_seq_groups giữ ref tới các SequenceGroup đã submit, để khi
        # poll_results đọc kết quả thì gán đúng score vào sg tương ứng.
        self._pending_future = None
        self._pending_seq_groups: Optional[List] = None

        # === [STREAMING] State cho streaming API mới (continuous queue model) ===
        # Tách HOÀN TOÀN khỏi legacy state ở trên — đảm bảo zero impact lên
        # opt-cpu-warmup baseline. Schedule_type set 1 lần lúc start server →
        # chỉ 1 API path active tại 1 thời điểm, 2 bộ state coexist nhưng
        # không tương tác.
        #
        # _stream_queue: FIFO queue của (sg, prompt) tuples chờ score
        # _stream_in_flight: set của request_ids đang queue/compute, dedup để
        #                    scheduler có thể call submit_streaming mỗi tick
        #                    không sợ duplicate work
        # _stream_lock: bảo vệ _stream_queue + _stream_in_flight (cả main
        #               thread và worker thread access)
        # _stream_worker_active: cờ worker đang chạy. Worker exit khi queue
        #                        empty, submit_streaming spawn lại nếu cần.
        # _stream_chunk_size: HARD CAP cho mini-batch trong worker loop.
        #                     Khác với max_batch_size (legacy: chunk size trong
        #                     _score_sync với batch nhỏ từ caller, OK với value
        #                     1000 vì caller chỉ submit ít mỗi lần).
        #                     Streaming queue có thể grow lớn (rate cao + post-
        #                     warmup gated) → cần cap thấp để mỗi chunk OV
        #                     inference latency reasonable. Min(max_batch_size,
        #                     32) là sweet spot: amortize tokenize/dispatch
        #                     overhead nhưng không tốn memory/time.
        self._stream_queue: deque = deque()
        self._stream_in_flight: set = set()
        self._stream_lock = threading.Lock()
        self._stream_worker_active: bool = False
        # Hard-cap để chống config max_batch_size lớn (vd 1000) làm worker
        # process huge batch → minute-long latency, predictor không catch up.
        # Cho phép override qua env OV_CHUNK_SIZE để sweep tune. Default
        # giữ nguyên 4 → zero impact existing benchmarks không set env var.
        # Vd: OV_CHUNK_SIZE=32 → tăng throughput predictor ~3-5× ở rate cao
        # (cost: per-call OV latency tăng, first-score-latency tăng).
        _env_chunk = os.environ.get("OV_CHUNK_SIZE")
        if _env_chunk:
            self._stream_chunk_size: int = min(max_batch_size, int(_env_chunk))
            logger.info(
                f"Override _stream_chunk_size = {self._stream_chunk_size} "
                f"từ env OV_CHUNK_SIZE={_env_chunk}"
            )
        else:
            self._stream_chunk_size: int = min(max_batch_size, 4)

        logger.info(f"OpenVINO predictor ready in {time.time() - t0:.1f}s")

        # === Memory profiler hook (gated by PROFILE_MEM=1) ===
        try:
            from vllm._profile_dumper import start as _start_profiler
            _start_profiler()
        except Exception as _e:
            logger.warning(f"profile dumper failed to start: {_e}")

    def _score_sync(self, prompts: List[str]) -> List[float]:
        """Internal pure compute — tokenize + OV inference cho một batch prompts.

        KHÔNG đụng SequenceGroup state. Cả sync và async API đều dùng helper này
        để tránh duplicate logic.

        Có thể được gọi từ:
          - Main thread (qua obtain_aux_scores): block caller.
          - Worker thread (qua submit_async → executor): chạy nền, GIL released
            khi tokenize (HF Rust) và OV inference (C++).

        record_function annotations giúp torch.profiler hiển thị block
        compute này trên Chrome/Perfetto trace — mỗi block 1 row, dễ xác
        nhận overlap với GPU forward (model_executor.execute_model).
        """
        # import torch.profiler as _prof
        scores: List[float] = []
        # with _prof.record_function("ov_predictor.batch_total"):
        for i in range(0, len(prompts), self.max_batch_size):
            batch = prompts[i:i + self.max_batch_size]
            # with _prof.record_function("ov_predictor.tokenize"):
            inps = self.tokenizer(
                batch, max_length=self.max_length,
                padding=True, truncation=True, return_tensors="pt",
            )
            # with _prof.record_function("ov_predictor.inference"):
            ov_out = self.compiled_model([
                inps["input_ids"].numpy(),
                inps["attention_mask"].numpy(),
            ])
            logits = list(ov_out.values())[0]  # shape: [batch, num_labels]
            scores.extend(logits.squeeze(-1).tolist())
        return scores

    @torch.no_grad()
    def obtain_aux_scores(self, seq_groups) -> List[float]:
        """SYNC API — score và set ngay vào seq_groups, block caller.

        Dùng cho:
          - Unit test (assert score values).
          - Fallback nếu cần sync semantic.
        Scheduler ranking flow KHÔNG nên gọi method này (sẽ block event loop).
        """
        if not seq_groups:
            return []
        t0 = time.time()
        prompts = [
            sg.seqs_dict[next(iter(sg.seqs_dict))].prompt
            for sg in seq_groups
        ]
        scores = self._score_sync(prompts)
        for sg, s in zip(seq_groups, scores):
            sg.set_aux_model_score(s)
        if int(os.environ.get("OPT_TIME", 0)):
            logger.info(
                f"OV-PRED-TIME (sync): n={len(prompts)} "
                f"t={time.time() - t0:.3f}s"
            )
        return scores

    # =========================================================================
    # ASYNC API
    # =========================================================================
    def is_busy(self) -> bool:
        """True nếu có Future pending chưa done. Non-blocking ~µs."""
        return self._pending_future is not None and not self._pending_future.done()

    def submit_async(self, seq_groups) -> bool:
        """Submit batch cho worker thread score. Non-blocking ~µs.

        Returns:
            True nếu Future được tạo (caller có thể tiếp tục).
            False nếu predictor đang busy (caller skip batch này, thử tick sau).

        Side effect: lưu seq_groups vào _pending_seq_groups để poll_results
        biết phải set score cho ai khi Future done.
        """
        # import torch.profiler as _prof
        # with _prof.record_function("ov_predictor.submit_async"):
        if self._executor is None:
            raise RuntimeError(
                "Async mode disabled. Init with async_mode=True or use "
                "obtain_aux_scores() instead."
            )
        if self.is_busy() or not seq_groups:
            return False

        # Snapshot prompts NGAY (không lazy) để tránh race nếu seq_group state
        # đổi giữa lúc submit và lúc worker chạy.
        prompts = [
            sg.seqs_dict[next(iter(sg.seqs_dict))].prompt
            for sg in seq_groups
        ]
        # Giữ ref tới seq_groups để poll_results có thể set score cho đúng sg.
        # list() copy để không bị ảnh hưởng khi caller mutate waiting queue.
        self._pending_seq_groups = list(seq_groups)
        self._pending_future = self._executor.submit(self._score_sync, prompts)
        return True

    def poll_results(self) -> int:
        """Check pending Future. Nếu done, set scores vào seq_groups.

        Returns:
            int — số seq_group được set score (0 nếu Future chưa done hoặc
            không có Future pending).

        Non-blocking: ~µs nếu future chưa done, vài µs nếu done (gồm
        deserialize result + set scores). KHÔNG bao giờ block caller.
        """
        import torch.profiler as _prof
        with _prof.record_function("ov_predictor.poll_results"):
            if self._pending_future is None or not self._pending_future.done():
                return 0

            n = 0
            try:
                scores = self._pending_future.result()
                for sg, s in zip(self._pending_seq_groups, scores):
                    # Defensive: seq_group có thể đã bị abort/finished giữa lúc
                    # submit và lúc poll. Set score thất bại không nên crash poll.
                    try:
                        sg.set_aux_model_score(s)
                        n += 1
                    except Exception as e:
                        logger.debug(f"set_aux_model_score skipped: {e}")
            except Exception as e:
                logger.warning(f"OV predictor batch failed, scores discarded: {e}")
            finally:
                # Reset state để cho phép submit mới ở tick sau.
                self._pending_future = None
                self._pending_seq_groups = None

            if int(os.environ.get("OPT_TIME", 0)) and n > 0:
                logger.info(f"OV-PRED-POLL: applied {n} scores")
            return n

    def shutdown(self) -> None:
        """Cleanup ThreadPoolExecutor + cả legacy lẫn streaming state.

        Gọi khi server shutdown.
        """
        if self._executor is not None:
            # wait=False: không chờ pending tasks, server đang shutdown.
            self._executor.shutdown(wait=False)
            self._executor = None

        # === [STREAMING] Cleanup streaming state ===
        # Worker thread sẽ exit tự nhiên khi queue empty (kiểm tra
        # _stream_worker_active = False). Clear queue + in_flight để
        # không leak ref tới seq_groups.
        with self._stream_lock:
            self._stream_queue.clear()
            self._stream_in_flight.clear()
            self._stream_worker_active = False

    # =========================================================================
    # STREAMING API — continuous queue model
    # =========================================================================
    # Thiết kế cho opt-cpu-async-warmup scheduler. ADD-only — KHÔNG modify
    # legacy submit_async/poll_results/is_busy ở trên.
    #
    # Khác biệt cốt lõi với legacy async:
    #   - Legacy: 1 Future per submit, scheduler check is_busy trước submit
    #   - Streaming: continuous queue, scheduler submit anytime, worker pull
    #     liên tục. Score apply incremental (mỗi mini-batch xong → set ngay).
    #
    # Dedup nội bộ qua _stream_in_flight: scheduler có thể call submit_streaming
    # mỗi tick với cùng request_id, predictor tự lọc duplicate.
    #
    # Cross-thread mutation: worker call sg.set_aux_model_score(s) từ worker
    # thread. GIL đảm bảo single-attribute assignment atomic — main thread đọc
    # aux_model_score trong sort key luôn consistent (single read = single value).
    # =========================================================================

    def submit_streaming(self, seq_groups) -> bool:
        """[STREAMING] Append unique seq_groups vào queue, wake worker nếu idle.

        Khác biệt với submit_async (legacy):
          - LUÔN accept (dedup nội bộ qua _stream_in_flight)
          - Không reject khi worker đang chạy — worker tự pull từ queue
          - Có thể call mỗi tick scheduler không sợ duplicate

        State management TÁCH BIỆT khỏi submit_async:
          - submit_streaming dùng _stream_queue, _stream_in_flight, ...
          - submit_async dùng _pending_future, _pending_seq_groups

        Args:
            seq_groups: list of SequenceGroup cần score.

        Returns:
            True nếu có request mới được add vào queue.
            False nếu rỗng hoặc tất cả đã in_flight (dedup'd).
        """
        if self._executor is None:
            raise RuntimeError(
                "Async mode disabled. Init with async_mode=True or use "
                "obtain_aux_scores() instead."
            )
        if not seq_groups:
            return False

        should_start_worker = False
        new_count = 0

        # Critical section: thêm vào queue + check worker state
        with self._stream_lock:
            for sg in seq_groups:
                # Dedup: skip nếu sg đã ở queue hoặc worker đang compute
                if sg.request_id not in self._stream_in_flight:
                    self._stream_in_flight.add(sg.request_id)
                    # Snapshot prompt NGAY (không lazy) để tránh race nếu
                    # seq_group state thay đổi giữa submit và worker pull
                    prompt = sg.seqs_dict[next(iter(sg.seqs_dict))].prompt
                    self._stream_queue.append((sg, prompt))
                    new_count += 1

            # Wake worker nếu có việc và worker idle
            if self._stream_queue and not self._stream_worker_active:
                self._stream_worker_active = True
                should_start_worker = True

        # Spawn worker outside lock (executor.submit có thể block ngắn)
        if should_start_worker:
            try:
                self._executor.submit(self._stream_worker_loop)
            except Exception as e:
                logger.error(f"Failed to spawn stream worker: {e}")
                # Cleanup state để cho phép retry tick sau
                with self._stream_lock:
                    self._stream_worker_active = False
                return False

        return new_count > 0

    def _stream_worker_loop(self):
        """[STREAMING] Worker thread: pull mini-batches từ queue liên tục.

        Chạy đến khi _stream_queue empty thì exit. submit_streaming sẽ
        spawn lại worker mới nếu có request đến sau khi worker exit.

        Mỗi mini-batch:
          1. Pull tối đa max_batch_size items từ queue (with lock)
          2. Tokenize + OV forward (outside lock — slow operation, GIL release)
          3. Apply score INCREMENTAL: mỗi sg, s pair → set_aux_model_score
             (cross-thread, GIL atomic)
          4. Cleanup _stream_in_flight cho các sg đã apply

        Worker không die vì 1 chunk fail — log warning, cleanup, continue.
        """
        while True:
            # Step 1: pull next chunk
            with self._stream_lock:
                if not self._stream_queue:
                    # Queue empty → exit. Sẽ được spawn lại nếu submit mới.
                    self._stream_worker_active = False
                    return

                chunk = []
                # Cap chunk size theo _stream_chunk_size (hard cap, default 32)
                # thay vì max_batch_size — tránh trường hợp config max_batch_size
                # = 1000 (legacy default) làm chunk quá lớn, OV inference quá
                # chậm, predictor không catch up kịp queue growth.
                while (self._stream_queue
                       and len(chunk) < self._stream_chunk_size):
                    chunk.append(self._stream_queue.popleft())

            # Step 2 + 3 + 4: compute + apply (outside lock — slow)
            # [PROFILE] Khi STREAM_TIME=1, đo wall-clock của 1 forward call
            # (tokenize + OV forward + apply scores). Dùng để compute
            # per-chunk latency và per-request amortized cost.
            _t0 = time.time() if int(os.environ.get("STREAM_TIME", 0)) else None
            # [TRACE_EVENTS] forward.start (worker thread)
            from vllm import event_tracer
            _trace_t0 = time.time() if event_tracer.is_enabled() else None
            if _trace_t0 is not None:
                event_tracer.log("predictor.worker.forward.start", {
                    "chunk_size": len(chunk),
                })
            try:
                chunk_sgs = [sg for sg, _ in chunk]
                chunk_prompts = [p for _, p in chunk]

                # Tokenize batch (HF Rust, GIL released)
                inps = self.tokenizer(
                    chunk_prompts, max_length=self.max_length,
                    padding=True, truncation=True, return_tensors="pt",
                )
                # OV forward (C++, GIL released)
                ov_out = self.compiled_model([
                    inps["input_ids"].numpy(),
                    inps["attention_mask"].numpy(),
                ])
                # Logits shape: [batch, num_labels=1] → squeeze → [batch]
                scores = list(ov_out.values())[0].squeeze(-1).tolist()

                # Apply scores incremental + cleanup in_flight
                # Lưu ý: từng sg có lock riêng để discard from in_flight,
                # nhưng set_aux_model_score là cross-thread mutation
                # (GIL atomic single-attr write).
                for sg, s in zip(chunk_sgs, scores):
                    with self._stream_lock:
                        self._stream_in_flight.discard(sg.request_id)
                    try:
                        sg.set_aux_model_score(s)
                    except Exception as e:
                        # Defensive: sg có thể đã abort/finished
                        # giữa submit và worker compute. Set fail không
                        # nên crash worker — log debug, skip.
                        logger.debug(
                            f"set_aux_model_score skipped: {e}"
                        )

                # [PROFILE] STREAM_TIME=1 ưu tiên log timing; OPT_TIME=1
                # fallback log số scores (backward compat).
                if _t0 is not None:
                    logger.info(
                        f"OV-STREAM-TIME: n={len(scores)} "
                        f"t={time.time() - _t0:.4f}s"
                    )
                elif int(os.environ.get("OPT_TIME", 0)):
                    logger.info(
                        f"OV-STREAM: applied {len(scores)} scores"
                    )

                # [TRACE_EVENTS] forward.end (worker thread)
                if _trace_t0 is not None:
                    event_tracer.log("predictor.worker.forward.end", {
                        "chunk_size": len(scores),
                        "lat_ms": (time.time() - _trace_t0) * 1000.0,
                    })

            except Exception as e:
                logger.warning(
                    f"OV stream worker chunk failed, scores discarded: {e}"
                )
                # Cleanup _stream_in_flight để cho phép re-submit
                # (scheduler tick sau sẽ thấy sg.aux_model_score=None
                # và thử submit lại)
                with self._stream_lock:
                    for sg, _ in chunk:
                        self._stream_in_flight.discard(sg.request_id)
                # Continue loop — pull next chunk, không die

    def poll_streaming(self) -> int:
        """[STREAMING] No-op — worker apply scores trực tiếp cross-thread.

        Khác với poll_results (legacy) là apply Future result, poll_streaming
        không có gì để apply (đã apply incremental bởi worker). Giữ method
        này để scheduler có thể call như API parity, hoặc cho future extension.

        Returns:
            Always 0 (cho consistency với poll_results signature).
        """
        return 0

    def has_pending_stream(self) -> bool:
        """[STREAMING] Diagnostic: streaming queue/in_flight còn việc không?

        Dùng cho monitoring/log, không phải critical path scheduler.

        Returns:
            True nếu queue có item HOẶC in_flight set non-empty.
        """
        with self._stream_lock:
            return (bool(self._stream_queue)
                    or len(self._stream_in_flight) > 0)

    # =========================================================================
    # [PROFILE] Tick profiler getter — gọi bởi scheduler khi OPT_PROFILE_TICK=1
    # =========================================================================
    def stream_stats(self) -> tuple:
        """[STREAMING][PROFILE] Snapshot (queue_depth, in_flight_count).

        DIRTY READ — KHÔNG acquire _stream_lock. Đây là choice intentional:
          - Mục đích: aggregated profiling (queue depth có cao không),
            không cần exact count. Sai số ±1 do race với worker
            popleft/discard là noise level.
          - Tránh lock contention với worker thread chạy mini-batch (worker
            acquire lock mỗi lần pop chunk + mỗi lần discard from in_flight).
            Profile reader chạy mỗi scheduler tick → frequency cao → nếu
            lock sẽ làm worker stall, bóp méo chính cái mình đang đo.
          - len(deque) và len(set) trên CPython là single-bytecode atomic
            (GIL bảo vệ) — không crash, chỉ có thể off-by-one.

        Returns:
            (stream_queue_depth, stream_in_flight_count): integers.
        """
        return len(self._stream_queue), len(self._stream_in_flight)
