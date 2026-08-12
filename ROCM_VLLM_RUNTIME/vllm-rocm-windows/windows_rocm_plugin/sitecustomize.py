# vllm-win: apply the single-process torch.distributed shim at EVERY interpreter startup, so vLLM's
# model-inspection subprocess (`python -m vllm.model_executor.models.registry`, which does NOT load the
# plugin) also gets a working torch.distributed on this USE_DISTRIBUTED=0 Windows torch. Without this,
# inspecting an un-cached model arch (e.g. ERNIE-4.5) fails with
# ModuleNotFoundError: torch._C._distributed_c10d. Runs after .pth processing (editable finder installed),
# guarded so a non-vllm python never breaks.
try:
    import vllm_windows_rocm.bootstrap  # noqa: F401  (import triggers torchdist_shim.apply())
except Exception:
    pass
