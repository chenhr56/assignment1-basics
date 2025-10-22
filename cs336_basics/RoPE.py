# import torch.nn.Module
import torch.nn as nn
import numpy as np
import torch


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        self.rotate_matrix_table = self.build_rotate_matrix()

        self.register_buffer(
            "rotate_matrix", self.rotate_matrix_table, persistent=False)

    def build_rotate_block(self, block_index: int, seq_pos: int):
        theta_ik = torch.tensor(seq_pos / (self.theta ** (2 * block_index / self.d_k)))
        sin, cos = torch.sin(theta_ik), torch.cos(theta_ik)
        return torch.Tensor([[cos, -sin], [sin, cos]])

    def build_rotate_matrix(self):
        matrix = torch.zeros(self.max_seq_len, self.d_k, self.d_k)
        for i in range(self.max_seq_len):
            tmp = [self.build_rotate_block(k, i) for k in range(self.d_k//2)]
            matrix[i::] = torch.block_diag(*tmp)
        return matrix

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        *_, seq_len, d_k = x.shape
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        rotate_matrix = self.rotate_matrix_table[token_positions]
        x_rotated = rotate_matrix @ x.unsqueeze(-1)
        return x_rotated.squeeze(-1)
