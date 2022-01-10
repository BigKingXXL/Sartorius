import torch
import cupy as cp

from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

def tensorToCupy(tensor: torch.Tensor) -> cp.ndarray:
    return cp.from_dlpack(to_dlpack(tensor))

def cupyToTensor(cupy: cp.ndarray) -> torch.Tensor:
    return from_dlpack(cupy.toDlpack())