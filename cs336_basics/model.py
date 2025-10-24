import einx
import torch.nn as nn
import numpy as np
import torch
from torch import Tensor
from einops import einsum, rearrange, einops
from jaxtyping import Float, Int

from cs336_basics.softmax import scaled_dot_product_attention


class Linear(nn.Module):
    def __init__(self, d_in, d_out, device=None, dtype=None):
        super().__init__()
        # factory_kwargs = {"device": device, "dtype": dtype}

        self.d_in = d_in
        self.d_out = d_out
        self.device = device
        self.dtype = dtype

        std = np.sqrt(2.0 / (d_in + d_out))

        self.weight: Float[Tensor, " d_out d_in"] = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(d_out, d_in), std=std, a=-3 * std, b=3 * std),
            requires_grad=True
        )

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, device=None, dtype=None):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.device = device
        self.dtype = dtype

        std = 1

        self.weight = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(vocab_size, d_model), std=std, a=-3 * std, b=3 * std),
            requires_grad=True
        )

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return self.weight[token_ids, :]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(
            nn.init.trunc_normal_(torch.ones(self.d_model, device=device))
        )

    def RMS(self, x: torch.Tensor):
        ms = x.pow(2).mean(-1, keepdim=True)
        return torch.sqrt(ms + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(self.device).to(torch.float32)
        result = (x / self.RMS(x)) * self.weight
        return result.to(in_dtype)


def SiLU(x: torch.Tensor):
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def GLU(self, x: torch.Tensor):
        return SiLU(self.w1(x)) * self.w3(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(SiLU(self.w1(x)) * self.w3(x))



class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # self.theta = theta
        # self.d_k = d_k
        # self.max_seq_len = max_seq_len
        # self.device = device

        # self.rotate_matrix_table = self.build_rotate_matrix()

        self.register_buffer(
            "rotate_matrix", self._init_cache(theta, d_k, max_seq_len), persistent=False)

    @staticmethod
    def _init_cache(theta: float, d_k: int, max_seq_len: int) -> Float[Tensor, " 2 context_length half_dim"]:
        assert d_k%2==0
        d = torch.arange(0, d_k, 2) / d_k
        freqs = theta ** -d
        t = torch.arange(max_seq_len)
        freqs = einsum(t, freqs, "t, f -> t f")

        cos, sin = torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos, sin))

    # def build_rotate_block(self, block_index: int, seq_pos: int):
    #     theta_ik = torch.tensor(seq_pos / (self.theta ** (2 * block_index / self.d_k)))
    #     sin, cos = torch.sin(theta_ik), torch.cos(theta_ik)
    #     return torch.Tensor([[cos, -sin], [sin, cos]])
    #
    # def build_rotate_matrix(self):
    #     matrix = torch.zeros(self.max_seq_len, self.d_k, self.d_k)
    #     for i in range(self.max_seq_len):
    #         tmp = [self.build_rotate_block(k, i) for k in range(self.d_k//2)]
    #         matrix[i::] = torch.block_diag(*tmp)
    #     return matrix

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x1, x2 = rearrange(x, '... (half_d xy) -> xy ... half_d', xy=2)
        cos, sin =einx.get_at('cos_sin [pos] half_dim, ... -> cos_sin ... half_dim', self.rotate_matrix, token_positions)
        # cos, sin = einx.rearrange('... x_half, ... x_half -> ... (x_half (1 + 1))', )

        x1_rot, x2_rot = cos*x1-sin*x2, sin*x1+cos*x2
        return einx.rearrange('... x_half, ... x_half -> ... (x_half (1 + 1))', x1_rot, x2_rot).contiguous()


        # *_, seq_len, d_k = x.shape
        # if token_positions is None:
        #     token_positions = torch.arange(seq_len, device=x.device)
        # rotate_matrix = self.rotate_matrix_table[token_positions]
        # x_rotated = rotate_matrix @ x.unsqueeze(-1)
        # return x_rotated.squeeze(-1)


class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 max_seq_len: int,
                 theta: float, ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.pwff = SwiGLU(d_model=d_model, d_ff=d_ff)

        self.msa = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, use_rope=True, theta=theta,
                                          max_seq_len=max_seq_len)

        self.rms1 = RMSNorm(d_model)
        self.rms2 = RMSNorm(d_model)

    def forward(self, in_features: torch.Tensor):
        mid1 = self.rms1(in_features)
        x = in_features + self.msa(mid1)
        return x + self.pwff(self.rms2(x))


class TransformerLM(nn.Module):

    def __init__(self, vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float, ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.token_embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.layers = nn.ModuleList(
            TransformerBlock(d_model=d_model,
                             num_heads=num_heads,
                             d_ff=d_ff,
                             max_seq_len=context_length,
                             theta=rope_theta, ) for _ in range(num_layers))
        self.norm = RMSNorm(d_model=d_model)
        self.linear = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, in_features: torch.Tensor):
        print(f'TransformerLM forward, in_features.device: {in_features.device}')
        x = self.token_embedding(in_features)

        for layer in self.layers:
            x = layer(x)
        print(f'TransformerLM forward, after layers, x.device: {x.device}')
        x = self.norm(x)
        print(f'TransformerLM forward, after norm, x.device: {x.device}')
        x = self.linear(x)
        print(f'TransformerLM forward, after linear, x.device: {x.device}')
        x.to(in_features.device)
        return x  # no softmax


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
