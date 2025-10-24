import torch.nn as nn
import torch

from .Embedding import Embedding
from .Linear import Linear
from .MultiheadSelfAttention import MultiheadSelfAttention
from .PositionwiseFeedforward import PositionwiseFeedforward
from .RMSNorm import RMSNorm
from .softmax import softmax


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
        self.pwff = PositionwiseFeedforward(d_model=d_model, d_ff=d_ff)

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
        print(f'TransformerLM forward, after layers, x.device: {x.device}')
        x = self.norm(x)
        print(f'TransformerLM forward, after norm, x.device: {x.device}')
        print(f'TransformerLM forward, after norm, x.device: {x.device}')
        x = self.linear(x)
        print(f'TransformerLM forward, after linear, x.device: {x.device}')
        x.to(in_features.device)
        return x # no softmax
