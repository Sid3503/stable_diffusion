import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention


class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):

        # TAKING INPUT FROM `VAE_Encoder` --> latent: (Batch, 4, H/8, W/8)
        # TAKING INPUT FROM `CLIP` --> context: (Batch, SeqLen, Dim)
        # time: (1, 320)

        # (1, 320) --> (1, 1280)
        time = self.time_embedding(time)

        # (Batch, 4, H/8, W/8) --> (Batch, 320, H/8, W/8)
        output = self.unet(latent, context, time)

        # (Batch, 320, H/8, W/8) --> (Batch, 4, H/8, W/8)
        output = self.final(output)

        # (Batch, 4, H/8, W/8)
        return output