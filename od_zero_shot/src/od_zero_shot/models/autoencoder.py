"""OD 矩阵卷积自编码器。"""

from __future__ import annotations

import torch
from torch import nn


class ODAutoencoder(nn.Module):
    """输入 `1x100x100`，latent 为 `16x25x25`。"""

    def __init__(self, latent_channels: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, latent_channels, kernel_size=4, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def _ensure_4d(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return x.unsqueeze(1)
        if x.ndim == 4:
            return x
        raise ValueError(f"ODAutoencoder 仅支持 [B,N,N] 或 [B,1,N,N]，收到 {tuple(x.shape)}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self._ensure_4d(x))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return {"reconstruction": reconstruction, "latent": latent}
