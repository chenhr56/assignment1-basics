# import torch.nn.Module
import torch.nn as nn
import numpy as np
import torch


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):

        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        w_init = self.weight_initialization(
            in_features, out_features, factory_kwargs)
        self.weight = nn.Parameter(w_init)

    def weight_initialization(self, in_dim: int, out_dim: int, factory_kwargs: dict):
        W = torch.empty(out_dim, in_dim, **factory_kwargs)
        mean, std = 0, np.sqrt(2.0/(in_dim+out_dim))
        nn.init.trunc_normal_(W, mean=mean, std=std, a=-3*std, b=3*std)
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
