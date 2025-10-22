# import torch.nn.Module
import torch.nn as nn
import numpy as np
import torch


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        w_init = self.weight_initialization(
            num_embeddings, embedding_dim, factory_kwargs)
        self.weight = nn.Parameter(w_init)

    def weight_initialization(self, vocab_size: int, d_model: int, factory_kwargs: dict):
        W = torch.empty(vocab_size, d_model, **factory_kwargs)
        mean, std = 0, 1
        nn.init.trunc_normal_(W, mean=mean, std=std, a=-3*std, b=3*std)
        return W

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = token_ids.shape
        output = torch.empty(batch_size, seq_length, self.embedding_dim)
        for i, seq in enumerate(token_ids):
            for j, token_id in enumerate(seq):
                output[i][j] = self.weight[token_id]
        return output
