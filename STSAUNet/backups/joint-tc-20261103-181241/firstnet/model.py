"""FIRST-Net: RPCANet-style unfolding with five-frame deformable alignment."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
        nn.ReLU(inplace=True),
    )


class DeepUnfoldingStage(nn.Module):
    """Learned proximal/gradient updates for background and sparse target maps."""

    def __init__(self, channels: int = 32):
        super().__init__()
        self.background_prox = _conv_block(1, channels)
        self.background_out = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.sparse_gradient = _conv_block(1, channels)
        self.sparse_out = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.merge = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
        )
        self.gamma_logit = nn.Parameter(torch.tensor(0.0))
        self.epsilon_logit = nn.Parameter(torch.tensor(-4.5951))

    def forward(self, image: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        background_input = image - target
        background = background_input + self.background_out(self.background_prox(background_input))
        residual = target + image - background
        gradient = self.sparse_out(self.sparse_gradient(residual))
        epsilon = torch.sigmoid(self.epsilon_logit)
        gamma = torch.sigmoid(self.gamma_logit)
        next_target = gamma * target + (1.0 - gamma) * residual - epsilon * gradient
        merged = background + next_target
        reconstruction = merged + self.merge(merged)
        return background, next_target, reconstruction


class FeatureSeparation(nn.Module):
    def __init__(self, stages: int = 6, channels: int = 32):
        super().__init__()
        self.stages = nn.ModuleList([DeepUnfoldingStage(channels) for _ in range(stages)])

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target = torch.zeros_like(images)
        background = torch.zeros_like(images)
        reconstruction = images
        for stage in self.stages:
            background, target, reconstruction = stage(reconstruction, target)
        return background, target, reconstruction


class DeformableConv2dGridSample(nn.Module):
    """Differentiable 3x3 deformable convolution implemented with grid_sample.

    It follows the bilinear offset sampling used by deformable convolution and
    avoids requiring a compiled MMCV/torchvision CUDA extension.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.weight = nn.Parameter(torch.empty(channels, channels, 3, 3))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, features: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = features.shape
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=features.device, dtype=features.dtype),
            torch.linspace(-1.0, 1.0, width, device=features.device, dtype=features.dtype),
            indexing="ij",
        )
        base_grid = torch.stack((x, y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        scale_x = 2.0 / max(width - 1, 1)
        scale_y = 2.0 / max(height - 1, 1)
        output = features.new_zeros((batch, self.channels, height, width))
        for ky in range(3):
            for kx in range(3):
                kernel_index = ky * 3 + kx
                dx = offsets[:, 2 * kernel_index] + (kx - 1)
                dy = offsets[:, 2 * kernel_index + 1] + (ky - 1)
                grid = base_grid.clone()
                grid[..., 0] = grid[..., 0] + dx * scale_x
                grid[..., 1] = grid[..., 1] + dy * scale_y
                sampled = F.grid_sample(
                    features, grid, mode="bilinear", padding_mode="zeros", align_corners=True
                )
                kernel = self.weight[:, :, ky, kx].unsqueeze(-1).unsqueeze(-1)
                output = output + F.conv2d(sampled, kernel)
        return output + self.bias.view(1, -1, 1, 1)


class DeformableAlign(nn.Module):
    def __init__(self, channels: int = 32, max_offset: float = 3.0):
        super().__init__()
        hidden = max(channels // 2, 8)
        self.max_offset = max_offset
        self.offset_net = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 18, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)
        self.deform = DeformableConv2dGridSample(channels)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, reference: torch.Tensor, support: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        offsets = torch.tanh(self.offset_net(torch.cat((reference, support), dim=1))) * self.max_offset
        aligned = self.deform(support, offsets)
        return aligned + self.refine(aligned), offsets


class FIRSTNet(nn.Module):
    def __init__(
        self,
        window_size: int = 5,
        stages: int = 6,
        unfold_channels: int = 32,
        feature_channels: int = 32,
        max_offset: float = 3.0,
    ):
        super().__init__()
        if window_size < 1 or window_size % 2 != 1:
            raise ValueError("window_size must be a positive odd integer")
        self.window_size = window_size
        self.separation = FeatureSeparation(stages, unfold_channels)
        self.target_features = nn.Sequential(
            nn.Conv2d(1, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.aligner = DeformableAlign(feature_channels, max_offset)
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_channels * window_size, feature_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels * 2, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.segmentation_head = nn.Conv2d(feature_channels, 1, kernel_size=1)
        self.reconstruction_head = nn.Conv2d(feature_channels, 1, kernel_size=3, padding=1)

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        if frames.ndim != 5:
            raise ValueError(f"Expected B,T,C,H,W input, got shape {tuple(frames.shape)}")
        batch, time, channels, height, width = frames.shape
        if time != self.window_size:
            raise ValueError(f"Model expects {self.window_size} frames, received {time}")
        if channels != 1:
            raise ValueError(f"FIRST-Net expects single-channel grayscale input, received {channels} channels")

        flat = frames.reshape(batch * time, channels, height, width)
        background, target, reconstruction = self.separation(flat)
        background = background.reshape(batch, time, 1, height, width)
        target = target.reshape(batch, time, 1, height, width)
        reconstruction = reconstruction.reshape(batch, time, 1, height, width)

        features = self.target_features(target.reshape(batch * time, 1, height, width))
        features = features.reshape(batch, time, -1, height, width)
        center_index = time // 2
        reference = features[:, center_index]
        aligned_features = []
        offsets = []
        for frame_index in range(time):
            if frame_index == center_index:
                aligned_features.append(reference)
                offsets.append(reference.new_zeros((batch, 18, height, width)))
            else:
                aligned, offset = self.aligner(reference, features[:, frame_index])
                aligned_features.append(aligned)
                offsets.append(offset)

        fused = self.fusion(torch.cat(aligned_features, dim=1))
        logits = self.segmentation_head(fused)
        reconstructed_center = background[:, center_index] + self.reconstruction_head(fused)
        return {
            "logits": logits,
            "reconstruction": reconstructed_center,
            "background": background[:, center_index],
            "target": target[:, center_index],
            "aligned_features": torch.stack(aligned_features, dim=1),
            "offsets": torch.stack(offsets, dim=1),
            "center_features": reference,
        }
