
import sys, types

# --- Disable torch.compile early to avoid dynamo importing broken DTensor ---
import torch
_original_compile = torch.compile
def _no_compile(*args, **kwargs):
    def decorator(fn):
        return fn
    if args and callable(args[0]) and len(args)==1 and not kwargs:
        # used as @torch.compile without args
        return args[0]
    return decorator
torch.compile = _no_compile
print("torch.compile disabled")

# --- Shim torch.distributed.tensor as package ---
try:
    import torch.distributed.tensor.parallel
except ModuleNotFoundError:
    try:
        import torch.distributed.tensor as tdt
    except ModuleNotFoundError:
        tdt = types.ModuleType('torch.distributed.tensor')
        sys.modules['torch.distributed.tensor'] = tdt
    if not hasattr(tdt, '__path__'):
        tdt.__path__ = []
    parallel_mod = types.ModuleType('torch.distributed.tensor.parallel')
    class Dummy: pass
    for n in ["ColwiseParallel","RowwiseParallel","PrepareModuleInput","PrepareModuleOutput","SequenceParallel","ParallelStyle","Shard","DTensor"]:
        setattr(parallel_mod, n, Dummy)
    sys.modules['torch.distributed.tensor.parallel'] = parallel_mod
    print("shimmed parallel")

# Shim _ops and related for inductor
for mod_name in [
    'torch.distributed.tensor._ops',
    'torch.distributed.tensor._ops._pointwise_ops',
    'torch.distributed.tensor._ops._embedding_ops',
    'torch.distributed.tensor._ops._matrix_ops',
]:
    if mod_name not in sys.modules:
        m = types.ModuleType(mod_name)
        if 'pointwise' in mod_name:
            m.register_inductor_prims = lambda: None
        sys.modules[mod_name] = m
        print(f"shimmed {mod_name}")

# Also shim torch._inductor.inductor_prims if needed
try:
    import torch._inductor.inductor_prims
except Exception as e:
    print(f"inductor_prims shim not needed or failed: {e}")

print(f"torch {torch.__version__} | cuda_avail {torch.cuda.is_available()} | dev {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda'}")

import first_token
