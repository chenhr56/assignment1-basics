# import torch.nn.Module
import torch.nn as nn
import numpy as np
import torch


class RMSNorm (nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        w_init = self.weight_initialization(factory_kwargs)
        self.weight = nn.Parameter(w_init.to(device))

    def weight_initialization(self, factory_kwargs: dict):
        W = torch.ones(self.d_model, **factory_kwargs)
        return W

    def RMS(self, x: torch.Tensor):
        ms = x.pow(2).mean(-1, keepdim=True)
        return torch.sqrt(ms+self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(self.device).to(torch.float32)
        result = (x / self.RMS(x)) * self.weight.to(x.device)
        return result.to(in_dtype)
