"""Configurable segmentation, reconstruction, and temporal alignment losses."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FIRSTNetLoss(nn.Module):
    def __init__(self, reconstruction_weight: float = 0.01, alignment_weight: float = 0.01):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.alignment_weight = alignment_weight

    @staticmethod
    def soft_iou_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probability * target).sum(dim=dims)
        union = probability.sum(dim=dims) + target.sum(dim=dims) - intersection
        return (1.0 - (intersection + 1.0) / (union + 1.0)).mean()

    def forward(self, output: dict[str, torch.Tensor], image: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        loss_bce = F.binary_cross_entropy_with_logits(output["logits"], mask)
        loss_iou = self.soft_iou_loss(output["logits"], mask)
        loss_segmentation = loss_bce + loss_iou
        loss_reconstruction = F.mse_loss(output["reconstruction"], image)

        aligned = output["aligned_features"]
        center = output["center_features"].unsqueeze(1)
        center_index = aligned.shape[1] // 2
        support = torch.cat((aligned[:, :center_index], aligned[:, center_index + 1 :]), dim=1)
        loss_alignment = F.l1_loss(support, center.expand_as(support)) if support.numel() else image.new_zeros(())

        total = (
            loss_segmentation
            + self.reconstruction_weight * loss_reconstruction
            + self.alignment_weight * loss_alignment
        )
        return {
            "loss": total,
            "loss_segmentation": loss_segmentation.detach(),
            "loss_bce": loss_bce.detach(),
            "loss_soft_iou": loss_iou.detach(),
            "loss_reconstruction": loss_reconstruction.detach(),
            "loss_alignment": loss_alignment.detach(),
        }
