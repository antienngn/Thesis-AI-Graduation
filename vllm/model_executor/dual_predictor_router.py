"""Router cho scheduler `dual<T>`.

Mỗi tick scheduler scan các request unscored. Router quyết định **whole-batch**:
nếu CPU OV chạy được toàn bộ N request kịp trước khi GPU model_executor xong,
route hết sang CPU; ngược lại route hết sang GPU (sync, opt-xxx style).

Decision logic:
  n_req = len(unscored)
  longest = max(prompt_len for sg in unscored)
  T_cpu  = LUT_CPU[(n_req, longest)]              # 2D lookup với cận trên
  T_main = LUT_MAIN[(n_running, n_decode,         # 4D lookup state
                     n_prefill, n_tokens_next)]
  T_main >= T_cpu → CPU (giấu được)
  T_main <  T_cpu → GPU (CPU sẽ lộ phần overlap không hết)

LUT lookup khi miss:
  - Exact match → return p50
  - In-range miss → ceiling (cận trên cả 2 chiều), tăng tiếp nếu vẫn empty
  - Out-of-LUT → clamp về boundary cell (option A: không extrapolate)

Trong warmup phase (t < T), mọi request route GPU bất kể (cold-start).
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DualPredictorRouter:
    """Hold 2 LUTs + quyết định CPU/GPU cho 1 batch unscored requests/tick.

    Args:
        lut_cpu_path: JSON LUT CPU (n_requests × longest_OPT_tokens → latency)
        lut_main_path: JSON LUT main (n_running × n_decode × n_prefill × n_tokens → latency)
        opt_tokenizer: HF tokenizer của OPT-125m. Bắt buộc nếu LUT CPU dùng
            OPT tokens (schema v4_simple). Router sẽ tokenize prompt bằng OPT
            để tính `longest_tokens` đúng đơn vị LUT.
        max_length: max truncation length của OV predictor (default 2048).
    """

    def __init__(
        self,
        lut_cpu_path: str,
        lut_main_path: str,
        opt_tokenizer=None,
        max_length: int = 2048,
    ):
        self.lut_cpu = self._load_cpu_lut(lut_cpu_path)
        self.lut_main = self._load_main_lut(lut_main_path)
        self.opt_tokenizer = opt_tokenizer
        self.max_length = max_length
        if opt_tokenizer is None:
            logger.warning(
                "DualPredictorRouter: opt_tokenizer=None. Fallback to "
                "Llama prompt_len (mismatch với CPU LUT v4_simple). Pass "
                "opt_tokenizer=cpu_predictor.tokenizer khi init."
            )
        logger.info(
            f"DualPredictorRouter ready: "
            f"{len(self.lut_cpu['cells'])} CPU cells, "
            f"{len(self.lut_main['cells'])} main cells, "
            f"opt_tokenizer={'set' if opt_tokenizer is not None else 'None'}, "
            f"max_length={max_length}"
        )

    # =========================================================================
    # LUT loading
    # =========================================================================

    def _load_cpu_lut(self, path: str) -> Dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"CPU LUT không tìm thấy: {path}. "
                f"Chạy benchmarks/LUT_CREATE/profile_cpu_predictor_lut.py trước."
            )
        data = json.loads(p.read_text())
        if data.get("schema_version") != "v4_simple":
            raise ValueError(
                f"CPU LUT schema không đúng: expect v4_simple, "
                f"got {data.get('schema_version')!r}. Rebuild bằng "
                f"profile_cpu_predictor_lut.py."
            )
        # Index cells theo (n_requests, longest_tokens_HI) — dùng upper edge
        # làm key để khớp với convention "ceiling = cận trên".
        # Vd cell (longest_lo=256, longest_hi=512) → key (n, 512).
        cells_dict = {}
        for c in data["cells"]:
            k = (c["n_requests"], c["longest_tokens_hi"])
            cells_dict[k] = c
        data["_cells_dict"] = cells_dict
        data["_n_req_edges"] = sorted(data["bucket_edges"]["n_requests"])
        data["_longest_edges"] = sorted(data["bucket_edges"]["longest_tokens"])
        # _longest_hi_list = list các upper edge có cell. Edges = [0,64,128,256,
        # 512,1024,...] → HI list = [64, 128, 256, 512, 1024, ...]
        data["_longest_hi_list"] = data["_longest_edges"][1:]
        return data

    def _load_main_lut(self, path: str) -> Dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Main LUT không tìm thấy: {path}. "
                f"Chạy benchmarks/LUT_CREATE/run_bench_sweep_main_lut.sh + "
                f"build_main_model_lut.py trước."
            )
        data = json.loads(p.read_text())
        cells_dict = {}
        for c in data["cells"]:
            k = (
                tuple(c["key"]["n_running"]),
                tuple(c["key"]["n_decode"]),
                tuple(c["key"]["n_prefill"]),
                tuple(c["key"]["n_tokens"]),
            )
            cells_dict[k] = c
        data["_cells_dict"] = cells_dict
        return data

    # =========================================================================
    # CPU LUT lookup — 2D (n_requests, longest_tokens)
    # =========================================================================

    def _ceil_n_req(self, n: int) -> Optional[int]:
        """Smallest n_req_bucket >= n. None nếu n > max (out of LUT)."""
        edges = self.lut_cpu["_n_req_edges"]
        for e in edges:
            if e >= n:
                return e
        return None  # out of LUT

    def _ceil_longest_hi(self, longest: int) -> Optional[int]:
        """Smallest upper edge >= longest (cận trên). None nếu out of LUT.

        Vd edges=[0,64,128,256,512,1024], HI list=[64,128,256,512,1024]:
          longest=400  → 512  (smallest hi >= 400)
          longest=512  → 512  (exact match)
          longest=1500 → None (out of LUT)
        """
        for hi in self.lut_cpu["_longest_hi_list"]:
            if hi >= longest:
                return hi
        return None  # out of LUT

    def estimate_cpu_lat(
        self, n_requests: int, longest_tokens: int,
    ) -> float:
        """Lookup CPU batch latency với ceiling lookup đơn giản.

        Cell key = (n_requests, longest_tokens_HI). Vd query (n=5, longest=400),
        edges L=[0,64,128,256,512,1024,...]:
          → n_ceil=6, L_ceil_hi=512
          → lookup cell (6, 512)

        1. Ceiling cả 2 chiều
        2. Cell exists → return latency_ms
        3. Cell empty hoặc out-of-LUT → return inf (force GPU)

        LUT build từ ShareGPT, bench cũng ShareGPT → distribution khớp,
        không cần fallback spiral/nearest-neighbor.
        """
        if n_requests <= 0:
            return 0.0

        n_ceil = self._ceil_n_req(n_requests)
        L_ceil_hi = self._ceil_longest_hi(longest_tokens)

        # Out-of-LUT → force GPU
        if n_ceil is None or L_ceil_hi is None:
            return float("inf")

        cell = self.lut_cpu["_cells_dict"].get((n_ceil, L_ceil_hi))
        if cell is not None:
            return float(cell["latency_ms"])

        # Cell empty (sparse) → force GPU
        return float("inf")

    # =========================================================================
    # Main LUT lookup — 4D state
    # =========================================================================

    def _floor_bucket(self, value: int, edges: List[int]) -> Tuple[int, int]:
        """Cận dưới: largest bucket [lo, hi) sao cho lo <= value.

        Examples (edges=[0,4,8,16,32]):
          value=5  → (4, 8)
          value=8  → (8, 16)
          value=100 → (16, 32)   # clamp: bucket cuối
          value=-1 → (0, 4)      # clamp: bucket đầu
        """
        chosen = (edges[0], edges[1])
        for i in range(len(edges) - 1):
            if edges[i] <= value:
                chosen = (edges[i], edges[i + 1])
            else:
                break
        return chosen

    def estimate_main_lat(
        self,
        n_running: int,
        n_decode: int,
        n_prefill: int,
        n_tokens: int,
    ) -> float:
        """Lookup main model latency với floor 4 chiều, exact lookup đơn giản.

        1. Floor mỗi feature → bucket cận dưới
        2. Cell exists → return latency_ms
        3. Cell empty → return 0 (force GPU)

        LUT build từ bench ShareGPT, sử dụng cùng workload → cells đủ phủ
        state space thực tế, không cần spiral fallback.
        """
        edges = self.lut_main["bucket_edges"]
        edge_lists = [edges["n_running"], edges["n_decode"],
                       edges["n_prefill"], edges["n_tokens"]]
        cells_dict = self.lut_main["_cells_dict"]

        key = tuple(self._floor_bucket(v, el)
                    for v, el in zip(
                        [n_running, n_decode, n_prefill, n_tokens],
                        edge_lists))
        cell = cells_dict.get(key)
        if cell is not None:
            return float(cell["latency_ms"])

        # Cell empty (sparse) → force GPU
        return 0.0

    # =========================================================================
    # Routing decision — whole batch
    # =========================================================================

    def route_batch(
        self,
        seq_groups: List,
        state: Dict[str, int],
    ) -> Tuple[List, List]:
        """Decide cho cả batch unscored requests.

        Args:
            seq_groups: list SequenceGroup cần score
            state: {n_running, n_decode, n_prefill, n_tokens_next}

        Returns:
            (cpu_list, gpu_list) — all-or-nothing decision.
            Nếu T_main >= T_cpu → all CPU. Else all GPU.
        """
        if not seq_groups:
            return [], []

        n_req = len(seq_groups)
        # longest tính theo OPT tokens (đơn vị LUT CPU)
        longest = max(self._get_opt_prompt_len(sg) for sg in seq_groups)

        t_cpu = self.estimate_cpu_lat(n_req, longest)
        t_main = self.estimate_main_lat(
            state.get("n_running", 0),
            state.get("n_decode", 0),
            state.get("n_prefill", 0),
            state.get("n_tokens_next", 0),
        )

        if t_main >= t_cpu:
            return list(seq_groups), []
        return [], list(seq_groups)

    def _get_opt_prompt_len(self, seq_group) -> int:
        """Trả về OPT n_tokens của prompt (sau truncation max_length).

        Cache trên seq_group để tránh tokenize lại mỗi tick. Nếu router
        không có opt_tokenizer (fallback), trả về Llama tokens.
        """
        # Cache hit?
        cached = getattr(seq_group, "_router_opt_n_tokens", None)
        if cached is not None:
            return cached

        # Fallback nếu thiếu tokenizer (giữ contract, mismatch unit)
        if self.opt_tokenizer is None:
            try:
                return seq_group.get_seqs()[0].get_prompt_len()
            except Exception:
                return 0

        # Tokenize bằng OPT, cap tại max_length
        try:
            seq = seq_group.get_seqs()[0]
            prompt_text = seq.prompt
            if prompt_text is None:
                # Edge case: prompt text not stored; fallback Llama len
                return seq.get_prompt_len()
            ids = self.opt_tokenizer(
                prompt_text, add_special_tokens=True
            )["input_ids"]
            n = min(len(ids), self.max_length)
            seq_group._router_opt_n_tokens = n  # cache
            return n
        except Exception as e:
            logger.debug(f"_get_opt_prompt_len failed: {e}; fallback Llama len")
            try:
                return seq_group.get_seqs()[0].get_prompt_len()
            except Exception:
                return 0
