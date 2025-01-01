import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int):
        super().__init__()

        self.token_embeddding = nn.Embedding(n_vocab, n_embed)

        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))

    def forward(self, tokens):

        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        x = self.token_embeddding(tokens)

        # (B, Seq_Len, Dim)
        x += self.position_embedding

        # (B, Seq_Len, Dim)
        return x
    
class CLIPLayer(nn.Module):
    def __init__(self, n_heads: int, n_embed: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_heads, n_embed)
        
        self.layernorm_2 = nn.LayerNorm(n_embed)

        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # (B, Seq_Len, Dim)
        residue = x


        # --FIRST HALF == SELF ATTENTION--

        # (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        x = self.layernorm_1(x)

        # (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        x = self.attention(x, causal_mask=True)

        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        x += residue

        # --SECOND HALF == FEED FORWARD--

        residue = x

        # (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        x = self.layernorm_2(x)

        # (B, Seq_Len, Dim) -> (B, Seq_Len, 4 * Dim)
        x = self.linear_1(x)

        # (B, Seq_Len, 4 * Dim) -> (B, Seq_Len, 4 * Dim)
        x = x * torch.sigmoid(1.702 * x)  # Quick GeLU activation function

        # (B, Seq_Len, 4 * Dim) -> (B, Seq_Len, Dim)
        x = self.linear_2(x)

        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        x += residue

        return x



class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (B, Seq_Len) -> (B, Seq_len, Dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # (B, Seq_Len, Dim)
        output = self.layernorm(state)

        return output