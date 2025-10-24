import torch.nn as nn
import torch
from .RoPE import RotaryPositionalEmbedding
from .Linear import Linear
import einops

from .softmax import scaled_dot_product_attention


class MultiheadSelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            use_rope: bool = False,
            theta: float | None = None,
            max_seq_len: int | None = None,
            token_pos: torch.Tensor | None = None, ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.rope = RotaryPositionalEmbedding(theta=theta,
                                              d_k=d_model // num_heads,
                                              max_seq_len=max_seq_len
                                              ) if use_rope else None
        self.Q_ = Linear(d_model, d_model)
        self.K_ = Linear(d_model, d_model)
        self.V_ = Linear(d_model, d_model)
        self.O_ = Linear(d_model, d_model)
        self.token_pos = token_pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        qkv_proj = torch.cat([self.Q_.weight, self.K_.weight, self.V_.weight])
        qkv = x @ qkv_proj.T.to(x.device)
        q, k, v = qkv.chunk(3, -1)
        q = einops.rearrange(q, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads)
        k = einops.rearrange(k, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads)
        v = einops.rearrange(v, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads)

        if self.use_rope:
            q = self.rope(q, self.token_pos)
            k = self.rope(k, self.token_pos)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()[None, None, :, :]
        output = scaled_dot_product_attention(q, k, v, mask=~causal_mask)
        output = einops.rearrange(output, "... h seq_len d_head -> ... seq_len (h d_head)")

        return self.O_(output)
