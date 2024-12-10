import torch
from torch import nn
from torch.nn import functional as F

from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (Batch, Channels, H, W) -> (Batch, 128, H, W)
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),

            # (Batch, 128, H, W) -> (Batch, 128, H, W)
            VAE_ResidualBlock(128, 128),

            # (Batch, 128, H, W) -> (Batch, 128, H, W)
            VAE_ResidualBlock(128, 128),

            # (Batch, 128, H, W) -> (Batch, 128, H/2, W/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (Batch, 128, H/2, W/2) -> (Batch, 256, H/2, W/2)
            VAE_ResidualBlock(128, 256),

            # (Batch, 256, H/2, W/2) -> (Batch, 256, H/2, W/2)
            VAE_ResidualBlock(256, 256),

            # (Batch, 256, H/2, W/2) -> (Batch, 256, H/4, W/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (Batch, 256, H/4, W/4) -> (Batch, 512, H/4, W/4)
            VAE_ResidualBlock(256, 512),

            # (Batch, 512, H/4, W/4) -> (Batch, 512, H/4, W/4)
            VAE_ResidualBlock(512, 512),

            # (Batch, 512, H/2, W/2) -> (Batch, 512, H/8, W/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # (Batch, 512, H/8, W/8) -> (Batch, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            # (Batch, 512, H/8, W/8) -> (Batch, 512, H/8, W/8)
            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            # (Batch, 512, H/8, W/8) -> (Batch, 512, H/8, W/8)
            nn.GroupNorm(32, 512),

            # (Batch, 512, H/8, W/8) -> (Batch, 512, H/8, W/8)
            nn.SiLU(),

            # (Batch, 512, H/8, W/8) -> (Batch, 8, H/8, W/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (Batch, 8, H/8, W/8) -> (Batch, 8, H/8, W/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Channels, Height, Width)
        # noise: (Batch, out_channels, H/8, W/8)

        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # o/p from last conv layer of autoencoder: (Batch, 8, H/8, W/8) -> 2 Tensors of shape: (Batch, 4, H/8, W/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # (Batch, 4, H/8, W/8)
        log_variance = torch.clamp(log_variance, -30, 20)

        # (Batch, 4, H/8, W/8)
        variance = log_variance.exp()

        # (Batch, 4, H/8, W/8)
        stdev = variance.sqrt()

        # Z = N(0, 1) -> N(mean, variance) = X?
        x = mean + stdev * noise

        # scaling
        x *= 0.18215

        return x