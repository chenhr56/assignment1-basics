import torch.nn as nn
import numpy as np
import math
import torch
import einops


def softmax(x: torch.tensor, dim: int = -1):
    x -= x.max(dim=dim, keepdim=True).values
    x = torch.exp(x)
    return x / x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:

    d_k = K.shape[-1]
    score = einops.einsum(
        Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)

    if mask is not None:
        score = score.masked_fill(~mask, float('-inf'))

    return softmax(score) @ V


