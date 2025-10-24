# import torch.nn.Module
import torch.nn as nn
import numpy as np
import torch
from torch import Tensor
from einops import einsum
from jaxtyping import Float

class Linear(nn.Module):
    def __init__(self, d_in, d_out, device=None, dtype=None):
        super().__init__()
        # factory_kwargs = {"device": device, "dtype": dtype}

        self.d_in = d_in
        self.d_out = d_out
        self.device = device
        self.dtype = dtype

        std = np.sqrt(2.0/(d_in+d_out))

        self.weight: Float[Tensor, " d_out d_in"] = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(d_out, d_in), std=std, a=-3*std, b=3*std),
            requires_grad=True
        )

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
