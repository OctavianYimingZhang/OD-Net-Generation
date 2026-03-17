"""最小 latent diffusion 组件。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        factor = math.log(10000) / max(half_dim - 1, 1)
        exponents = torch.exp(torch.arange(half_dim, device=timesteps.device) * -factor)
        args = timesteps.float().unsqueeze(1) * exponents.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.gelu(self.norm1(self.conv1(x)))
        x = F.gelu(self.norm2(self.conv2(x)) + residual)
        return x


class TinyLatentUNet(nn.Module):
    def __init__(self, latent_channels: int, cond_channels: int, base_channels: int = 64) -> None:
        super().__init__()
        time_dim = base_channels * 2
        self.cond_channels = cond_channels
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_block = ConvBlock(latent_channels + cond_channels, base_channels)
        self.down_block = ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 2)
        self.up_block = ConvBlock(base_channels * 2 + base_channels * 2 + cond_channels, base_channels * 2)
        self.out_block = ConvBlock(base_channels * 2 + base_channels + cond_channels, base_channels)
        self.final = nn.Conv2d(base_channels, latent_channels, kernel_size=1)
        self.time_to_down = nn.Linear(time_dim, base_channels * 2)
        self.time_to_bottleneck = nn.Linear(time_dim, base_channels * 2)
        self.time_to_out = nn.Linear(time_dim, base_channels)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cond_l1 = cond
        cond_l2 = F.avg_pool2d(cond, kernel_size=2, stride=2, ceil_mode=True)
        time_embed = self.time_embed(timesteps)
        h1 = self.input_block(torch.cat([x, cond_l1], dim=1))
        h2 = self.down_block(F.avg_pool2d(h1, kernel_size=2, stride=2, ceil_mode=True))
        h2 = h2 + self.time_to_down(time_embed).unsqueeze(-1).unsqueeze(-1)
        h_mid = self.bottleneck(h2)
        h_mid = h_mid + self.time_to_bottleneck(time_embed).unsqueeze(-1).unsqueeze(-1)
        up1 = F.interpolate(h_mid, size=h2.shape[-2:], mode="bilinear", align_corners=False)
        up1 = self.up_block(torch.cat([up1, h2, cond_l2], dim=1))
        up2 = F.interpolate(up1, size=h1.shape[-2:], mode="bilinear", align_corners=False)
        up2 = self.out_block(torch.cat([up2, h1, cond_l1], dim=1))
        up2 = up2 + self.time_to_out(time_embed).unsqueeze(-1).unsqueeze(-1)
        return self.final(up2)


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps: int) -> None:
        super().__init__()
        betas = torch.linspace(1e-4, 2e-2, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        prev_alpha_bars = torch.cat([torch.tensor([1.0], dtype=torch.float32), alpha_bars[:-1]])
        self.register_buffer("posterior_variance", betas * (1.0 - prev_alpha_bars) / (1.0 - alpha_bars))

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=device)

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar = self.sqrt_alpha_bars[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[timesteps].view(-1, 1, 1, 1)
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise, noise

    def training_loss(self, denoiser: nn.Module, x_start: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        timesteps = self.sample_timesteps(x_start.shape[0], x_start.device)
        x_t, noise = self.q_sample(x_start, timesteps)
        pred_noise = denoiser(x_t, timesteps, cond)
        return F.mse_loss(pred_noise, noise)

    def sample(self, denoiser: nn.Module, shape: tuple[int, ...], cond: torch.Tensor, device: torch.device) -> torch.Tensor:
        current = torch.randn(shape, device=device)
        for timestep in reversed(range(self.timesteps)):
            t_batch = torch.full((shape[0],), timestep, device=device, dtype=torch.long)
            predicted_noise = denoiser(current, t_batch, cond)
            beta_t = self.betas[timestep]
            sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[timestep]
            sqrt_recip_alpha = self.sqrt_recip_alphas[timestep]
            current = sqrt_recip_alpha * (current - beta_t / sqrt_one_minus_alpha_bar * predicted_noise)
            if timestep > 0:
                current = current + torch.sqrt(torch.clamp(self.posterior_variance[timestep], min=1e-8)) * torch.randn_like(current)
        return current


class ConditionalLatentDiffusion(nn.Module):
    """支持 unconditional / conditional 两种模式的条件扩散封装。"""

    def __init__(self, latent_channels: int = 16, pair_dim: int = 64, diffusion_steps: int = 100, conditional: bool = True) -> None:
        super().__init__()
        self.conditional = conditional
        self.denoiser = TinyLatentUNet(latent_channels=latent_channels, cond_channels=pair_dim)
        self.diffusion = GaussianDiffusion(diffusion_steps)

    def training_loss(self, clean_latent: torch.Tensor, pair_condition: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if pair_condition is None:
            pair_condition = torch.zeros(
                clean_latent.shape[0],
                self.denoiser.cond_channels,
                clean_latent.shape[-2],
                clean_latent.shape[-1],
                device=clean_latent.device,
            )
        if pair_condition.shape[-2:] != clean_latent.shape[-2:]:
            pair_condition = F.interpolate(pair_condition, size=clean_latent.shape[-2:], mode="bilinear", align_corners=False)
        loss = self.diffusion.training_loss(self.denoiser, clean_latent, pair_condition)
        timesteps = self.diffusion.sample_timesteps(clean_latent.shape[0], clean_latent.device)
        noisy, _ = self.diffusion.q_sample(clean_latent, timesteps)
        pred_noise = self.denoiser(noisy, timesteps, pair_condition)
        return {"loss": loss, "pred_noise": pred_noise}

    def sample(self, num_samples: int, device: str | torch.device, pair_condition: torch.Tensor | None = None) -> torch.Tensor:
        device_obj = torch.device(device)
        if pair_condition is None:
            if self.conditional:
                raise ValueError("conditional diffusion 采样时 pair_condition 不能为空。")
            cond = torch.zeros(num_samples, self.denoiser.cond_channels, 25, 25, device=device_obj)
        else:
            cond = F.interpolate(pair_condition, size=(25, 25), mode="bilinear", align_corners=False).to(device_obj)
        return self.diffusion.sample(self.denoiser, (num_samples, self.denoiser.final.out_channels, 25, 25), cond.to(device_obj), device_obj)
