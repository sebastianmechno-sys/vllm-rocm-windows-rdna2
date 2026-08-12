import sys, types
try:
    import torch.distributed.tensor.parallel
except ModuleNotFoundError:
    import torch.distributed.tensor as tdt
    if not hasattr(tdt, '__path__'):
        tdt.__path__ = []
    parallel_mod = types.ModuleType('torch.distributed.tensor.parallel')
    class Dummy: pass
    for name in ["ColwiseParallel","RowwiseParallel","PrepareModuleInput","PrepareModuleOutput","SequenceParallel","ParallelStyle","ColwiseParallel","RowwiseParallel"]:
        setattr(parallel_mod, name, Dummy)
    sys.modules['torch.distributed.tensor.parallel'] = parallel_mod
    print("shimmed torch.distributed.tensor.parallel")

from first_token import *