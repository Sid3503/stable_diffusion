import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias, out_proj_bias = True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask = False):
        # x: (Batch, Seq_Len, Dim)  where Seq_Len = H*W and Dim = Features
        input_shape = x.shape

        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (Batch, Seq_Len, Dim) -> (Batch, Seq_Len, Dim * 3) -> into 3 tensors of shape (Batch, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (Batch, Seq_Len, Dim) -> (Batch, Seq_Len, num_heads, head_dim) -> (Batch, num_heads, Seq_Len, head_dim)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch, num_heads, Seq_Len, head_dim) x (Batch, num_heads, head_dim, Seq_Len) --> (Batch, num_heads, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # (Batch, num_heads, Seq_Len, Seq_Len) x (Batch, num_heads, Seq_Len, head_dim) --> (Batch, num_heads, Seq_Len, head_dim)
        output = weight @ v

        # (Batch, num_heads, Seq_Len, head_dim) -> (Batch, Seq_Len, num_heads, head_dim)
        output = output.transpose(1, 2)

        # (Batch, Seq_Len, Dim)
        output = output.reshape(input_shape)

        return output