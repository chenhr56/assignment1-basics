import einx
import torch.nn as nn
import numpy as np
import torch, logging
from torch import Tensor
from einops import einsum, rearrange, einops
from jaxtyping import Float, Int
import math

logger = logging.getLogger(__name__)

def softmax(x: torch.tensor, dim: int = -1):
    x -= x.max(dim=dim, keepdim=True).values
    x = torch.exp(x)
    return x / x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:

    d_k = K.shape[-1]
    score = einops.einsum(
        Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)

    if mask is not None:
        score = score.masked_fill(~mask, float('-inf'))

    return softmax(score) @ V

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
            "_rotate_matrix", RotaryPositionalEmbedding._init_cache(theta, d_k, max_seq_len), persistent=False)

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

    def forward(self, x: Float[Tensor, " ... seq d"], token_positions: Int[Tensor, " ... seq"]) -> Float[Tensor, " ... seq d"]:
        x1, x2 = rearrange(x, '... (half_d xy) -> xy ... half_d', xy=2)
        cos, sin =einx.get_at('cos_sin [pos] half_dim, ... -> cos_sin ... half_dim', self._rotate_matrix, token_positions)
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
                 pos_encoder: RotaryPositionalEmbedding):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.pwff = SwiGLU(d_model=d_model, d_ff=d_ff)

        self.attn = MultiheadSelfAttention(
            d_model=d_model, 
            num_heads=num_heads, 
            pos_encoder=pos_encoder
        )

        self.rms1 = RMSNorm(d_model)
        self.rms2 = RMSNorm(d_model)

    def forward(self, in_features: torch.Tensor):
        mid1 = self.rms1(in_features)
        x = in_features + self.attn(mid1)
        return x + self.pwff(self.rms2(x))


class TransformerLM(nn.Module):

    def __init__(
        self, 
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float, ):
        self.config = {
            k: v for k, v in locals().items() if k != "self" and not (k.startswith("__") and k.endswith("__"))
        }
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.token_embedding = Embedding(vocab_size=vocab_size, d_model=d_model)
        self.pos_encoder = RotaryPositionalEmbedding(
            theta=rope_theta, 
            d_k=d_model // num_heads, 
            max_seq_len=context_length
        )
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                pos_encoder=self.pos_encoder 
            ) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model=d_model)
        self.linear = Linear(d_in=d_model, d_out=vocab_size)

        logger.info(f"number of non-embedding parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.linear.weight.numel()

        return n_params

    def forward(self, in_features: Int[Tensor, " ... sequence_length"]) -> Float[Tensor, " ... sequence_length vocab_size"]:
        _, sequence_length = in_features.size()
        # print(f'TransformerLM forward, in_features.device: {in_features.device}')
        x = self.token_embedding(in_features)

        for layer in self.layers:
            x = layer(x)
        # print(f'TransformerLM forward, after layers, x.device: {x.device}')
        x = self.norm(x)
        # print(f'TransformerLM forward, after norm, x.device: {x.device}')
        return self.linear(x)  # no softmax


class MultiheadSelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            pos_encoder: RotaryPositionalEmbedding | None = None,
        ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.Q_ = Linear(d_model, d_model)
        self.K_ = Linear(d_model, d_model)
        self.V_ = Linear(d_model, d_model)
        self.O_ = Linear(d_model, d_model)
        self.pos_encoder = pos_encoder

    def forward(
        self, 
        x: Float[Tensor, " ... seq d_k"], 
        token_pos: Int[Tensor, " ... seq"] | None = None
    ) -> Float[Tensor, " ... seq d_v"]:

        *b, seq_len, d_model = x.shape
        assert d_model == self.d_model

        q, k, v = self.Q_(x), self.K_(x), self.V_(x)
        q, k, v = (
            rearrange(_, "... seq (heads d) -> ... heads seq d", heads=self.num_heads) for _ in (q, k, v)
        )
        

        if token_pos is None:
            token_pos = einx.rearrange("seq -> b... seq", torch.arange(seq_len, device=x.device), b=[1] * len(b))

        token_pos = rearrange(token_pos, "... seq -> ... 1 seq")

        if self.pos_encoder is not None:
            q, k = self.pos_encoder(q, token_pos), self.pos_encoder(k, token_pos)

        seq = torch.arange(seq_len, device=x.device)
        qi = einx.rearrange('query -> b... 1 query 1', seq, b=[1] * len(b))
        kj = einx.rearrange('key -> b... 1 1 key', seq, b=[1] * len(b))


        causal_mask = qi>=kj
        output = scaled_dot_product_attention(q, k, v, mask=causal_mask)
        output = rearrange(output, "batch heads seq d_v -> batch seq (heads d_v)").contiguous()

        return self.O_(output)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     seq_len = x.shape[-2]
    #     qkv_proj = torch.cat([self.Q_.weight, self.K_.weight, self.V_.weight])
    #     qkv = x @ qkv_proj.T.to(x.device)
    #     q, k, v = qkv.chunk(3, -1)
    #     q = einops.rearrange(q, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads)
    #     k = einops.rearrange(k, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads)
    #     v = einops.rearrange(v, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads)

    #     if self.use_rope:
    #         q = self.rope(q, self.token_pos)
    #         k = self.rope(k, self.token_pos)

    #     causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()[None, None, :, :]
    #     output = scaled_dot_product_attention(q, k, v, mask=~causal_mask)
    #     output = einops.rearrange(output, "... h seq_len d_head -> ... seq_len (h d_head)")

    #     return self.O_(output)
