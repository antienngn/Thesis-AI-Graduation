"""DualAuxModel — wrap cả CPU OV + GPU AUXLLM predictor + Router.

Cấu trúc:
  DualAuxModel
    ├── .cpu     : OpenVINOPredictor (async streaming, CPU AVX-512)
    ├── .gpu     : AUXLLM (sync GPU AUX-LLM, opt-xxx style)
    └── .router  : DualPredictorRouter (LUT-driven decision)

Scheduler `dual<T>` orchestrate:
  - Warmup (t < T): mọi unscored → self.gpu.obtain_aux_scores(...) sync
  - Post-warmup:
      cpu_list, gpu_list = self.router.route_batch(unscored, state)
      self.cpu.submit_streaming(cpu_list)        # async, non-blocking
      if gpu_list: self.gpu.obtain_aux_scores(gpu_list)  # sync, blocks tick
"""
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class DualAuxModel:
    """Composite predictor: CPU OV + GPU AUXLLM, route bởi LUT-based router."""

    def __init__(self, cpu_predictor, gpu_predictor, router):
        self.cpu = cpu_predictor
        self.gpu = gpu_predictor
        self.router = router
        logger.info(
            f"DualAuxModel ready: cpu={type(cpu_predictor).__name__}, "
            f"gpu={type(gpu_predictor).__name__}"
        )

    # -------------------------------------------------------------------------
    # API contract với scheduler (giống OpenVINOPredictor + AUXLLM)
    # -------------------------------------------------------------------------

    def poll_streaming(self) -> int:
        """Forward to CPU predictor (no-op nhưng giữ contract)."""
        return self.cpu.poll_streaming()

    def submit_streaming(self, seq_groups) -> bool:
        """Submit batch sang CPU OV (async)."""
        return self.cpu.submit_streaming(seq_groups)

    def obtain_aux_scores(self, seq_groups) -> List[float]:
        """Sync score qua GPU AUXLLM. Giống flow opt-xxx."""
        return self.gpu.obtain_aux_scores(seq_groups)

    def is_busy(self) -> bool:
        """Có CPU worker đang chạy không (GPU không busy ở mức scheduler)."""
        # Delegate to CPU since CPU is async; GPU sync luôn 'done' khi return
        return self.cpu.is_busy() if hasattr(self.cpu, "is_busy") else False

    # -------------------------------------------------------------------------
    # Router helper
    # -------------------------------------------------------------------------

    def route_batch(self, seq_groups, state: Dict[str, int]):
        """Chia 1 batch thành (cpu_subgroup, gpu_subgroup) qua router."""
        return self.router.route_batch(seq_groups, state)

    def is_pending_cpu_score(self, seq_group) -> bool:
        """True nếu request đang ở trong CPU OV queue/worker (chưa landing).

        Dùng để scheduler tránh re-submit request đã trong CPU pipeline (race
        khác sẽ làm AUXLLM assert fail nếu cùng request được dispatch GPU
        trong khi CPU worker đang chạy → CPU landing giữa lúc GPU sync forward).
        """
        return seq_group.request_id in self.cpu._stream_in_flight

    def shutdown(self):
        """Cleanup cả 2 predictor."""
        if hasattr(self.cpu, "shutdown"):
            self.cpu.shutdown()
        # AUXLLM không có shutdown explicit, GC handle
