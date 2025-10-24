# import torch.nn.Module
import torch.nn as nn
import numpy as np
import torch
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, device=None, dtype=None):
        super().__init__()
        # factory_kwargs = {"device": device, "dtype": dtype}

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.dtype = dtype

        std = np.sqrt(2.0/(in_dim+out_dim))

        self.weight = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(out_dim, in_dim), std=std, a=-3*std, b=3*std),
            requires_grad=True
        )

    #     w_init = self.weight_initialization(
    #         in_features, out_features, factory_kwargs)
    #     self.weight = nn.Parameter(w_init)
    #
    # def weight_initialization(self, in_dim: int, out_dim: int, factory_kwargs: dict):
    #     W = torch.empty(out_dim, in_dim, **factory_kwargs)
    #     mean, std = 0, np.sqrt(2.0/(in_dim+out_dim))
    #     nn.init.trunc_normal_(W, mean=mean, std=std, a=-3*std, b=3*std)
    #     return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in d_out d_in -> ... d_out")
        # return x @ self.weight.T.to(x.device)
