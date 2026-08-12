"""WindowsRocmPlatform: vLLM RocmPlatform adapted for native Windows.

RocmPlatform itself imports cleanly on Windows (its `amdsmi` import is wrapped in
try/except) and most methods already work: get_device_capability() derives (major, minor)
from the module-load gcnArchName (resolved via torch.cuda fallback on Windows), and
get_device_total_memory()/set_device() use torch.cuda. We only override the handful of
methods that hard-depend on amdsmi, routing them through torch.cuda instead.
"""
import os

import torch

from vllm.platforms.rocm import RocmPlatform


class WindowsRocmPlatform(RocmPlatform):
    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        try:
            uuid = getattr(torch.cuda.get_device_properties(device_id), "uuid", None)
            return str(uuid) if uuid is not None else f"gpu-{device_id}"
        except Exception:
            return f"gpu-{device_id}"

    @classmethod
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        # Single-GPU only on Windows (no amdsmi topology, no RCCL). Never "fully connected".
        return False

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads=None):
        # Per-group backend selection for KVarN: a group whose KV cache is kvarn-quantized MUST
        # use the KVARN backend (the only one accepting kvarn_* dtypes); return it directly.
        # fp16/bf16 (sliding-window) groups fall through to the normal ROCm selection. This is
        # necessary because vllm_config.attention_config.backend resolves to TRITON_ATTN and
        # would otherwise be forced on all groups (and the module-level priority patch was fragile).
        kvd = getattr(attn_selector_config, "kv_cache_dtype", None)
        if kvd is not None and str(kvd).startswith("kvarn_"):
            from vllm.v1.attention.backends.registry import AttentionBackendEnum
            return AttentionBackendEnum.KVARN.get_path()
        return super().get_attn_backend_cls(
            selected_backend, attn_selector_config, num_heads
        )

    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        super().check_and_update_config(vllm_config)
        # KVarN KV-cache quant: by default keep sliding-window layers in fp16 (only global
        # attention layers compress). Otherwise the SlidingWindowSpec would be built with the
        # uint8 kvarn dtype but sized by the fp16 page formula -> allocator mismatch. Route SWA
        # layers to "auto" via kv_cache_dtype_skip_layers (attention.py skip logic). Set
        # KVARN_QUANT_SLIDING=1 to also compress the 256-dim sliding layers (less tested).
        try:
            cc = vllm_config.cache_config
            kvd = str(getattr(cc, "cache_dtype", "") or "")
            if kvd.startswith("kvarn_"):
                if os.environ.get("KVARN_QUANT_SLIDING", "0") != "1":
                    skip = list(cc.kv_cache_dtype_skip_layers or [])
                    if "sliding_window" not in skip:
                        skip.append("sliding_window")
                        cc.kv_cache_dtype_skip_layers = skip
                    print("vllm-win kvarn: skip_layers ->", cc.kv_cache_dtype_skip_layers,
                          "| cc id", id(cc))
                # Per-layer backend selection: prepend KVARN to the dense ROCm priorities so
                # kvarn-dtype attention groups pick it (gated by its supports_kv_cache_dtype);
                # the fp16/bf16 (sliding-window) groups fall through to the normal backend.
                # Do NOT pass --attention-backend KVARN (that forces it on ALL layers, breaking
                # the fp16 groups). _get_backend_priorities is a module global -> monkeypatchable.
                import vllm.platforms.rocm as _rocm
                from vllm.v1.attention.backends.registry import AttentionBackendEnum
                if not getattr(_rocm._get_backend_priorities, "_kvarn_wrapped", False):
                    _orig_prio = _rocm._get_backend_priorities

                    def _prio(use_mla, use_sparse, _o=_orig_prio):
                        b = list(_o(use_mla, use_sparse))
                        if (not use_mla and not use_sparse
                                and AttentionBackendEnum.KVARN not in b):
                            b = [AttentionBackendEnum.KVARN] + b
                        return b

                    _prio._kvarn_wrapped = True
                    _rocm._get_backend_priorities = _prio
        except Exception as e:  # noqa: BLE001
            print("vllm-win kvarn config warning:", repr(e))
        # Register the fast M=1 AWQ-uint4 GEMV ahead of conch. Done here (engine config setup)
        # rather than at import/bootstrap: vllm + the platform plugin are fully loaded by now,
        # so importing vllm.model_executor.kernels.linear won't circular-import this package.
        try:
            from . import awq_gemv
            awq_gemv.register()
        except Exception as e:  # noqa: BLE001
            print("vllm-win awq_gemv register warning:", repr(e))
