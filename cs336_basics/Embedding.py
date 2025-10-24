# import torch.nn.Module
import torch.nn as nn
import numpy as np
import torch


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.device = device
        self.dtype = dtype

        std = 1

        self.weight: Float[Tensor, " d_out d_in"] = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(vocab_size, d_model), std=std, a=-3*std, b=3*std),
            requires_grad=True
        )

    def forward(self, token_ids: Int[Tensor, " ..."]) -> torch.Tensor:
        batch_size, seq_length = token_ids.shape
        output = torch.empty(batch_size, seq_length, self.d_model)
        for i, seq in enumerate(token_ids):
            for j, token_id in enumerate(seq):
                output[i][j] = self.weight[token_id]
        return output
