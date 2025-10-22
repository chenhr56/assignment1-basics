import torch.nn as nn
import numpy as np
import torch
from .Linear import Linear


def SiLU(x: torch.Tensor):
    return x * torch.sigmoid(x)


class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        self.w1 = Linear(d_model, d_ff, **factory_kwargs)
        self.w2 = Linear(d_ff, d_model, **factory_kwargs)
        self.w3 = Linear(d_model, d_ff, **factory_kwargs)

    def GLU(self, x: torch.Tensor):
        return SiLU(self.w1(x)) * self.w3(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(SiLU(self.w1(x)) * self.w3(x))
