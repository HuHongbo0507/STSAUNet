"""Configurable segmentation, reconstruction, and temporal alignment losses."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class STSAUNetLoss(nn.Module):
    def __init__(self, reconstruction_weight: float = 0.01, alignment_weight: float = 0.01,
                 segmentation_mode: str = "baseline", hard_negative_ratio: float = 3.0,
                 minimum_negatives: int = 256):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.alignment_weight = alignment_weight
        if segmentation_mode not in ("baseline", "hard_bce"):
            raise ValueError("segmentation_mode must be baseline or hard_bce")
        if hard_negative_ratio <= 0 or minimum_negatives < 1:
            raise ValueError("Invalid hard-negative budget")
        self.segmentation_mode = segmentation_mode
        self.hard_negative_ratio = hard_negative_ratio
        self.minimum_negatives = minimum_negatives

    def segmentation_bce(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.segmentation_mode == "baseline":
            return F.binary_cross_entropy_with_logits(logits, mask)
        pixel_loss = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
        samples = []
        for losses, truth in zip(pixel_loss.flatten(1), mask.flatten(1)):
            positive = losses[truth >= 0.5]
            negative = losses[truth < 0.5]
            count = min(negative.numel(), max(self.minimum_negatives,
                        int(self.hard_negative_ratio * positive.numel())))
            selected = negative.topk(count, sorted=False).values if count else negative
            samples.append((positive.sum() + selected.sum()) / max(positive.numel() + count, 1))
        return torch.stack(samples).mean()

    @staticmethod
    def soft_iou_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probability * target).sum(dim=dims)
        union = probability.sum(dim=dims) + target.sum(dim=dims) - intersection
        return (1.0 - (intersection + 1.0) / (union + 1.0)).mean()

    def forward(self, output: dict[str, torch.Tensor], image: torch.Tensor, mask: torch.Tensor,
                frames: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        loss_bce = self.segmentation_bce(output["logits"], mask)
        loss_iou = self.soft_iou_loss(output["logits"], mask)
        loss_segmentation = loss_bce + loss_iou
        loss_reconstruction = F.mse_loss(output["reconstruction"], image)
        if "unfolding_reconstruction" in output:
            unfolded = output["unfolding_reconstruction"]
            loss_unfolding = (F.mse_loss(unfolded, frames) if frames is not None
                              else F.mse_loss(unfolded[:, unfolded.shape[1] // 2], image))
            loss_reconstruction = 0.5 * (loss_reconstruction + loss_unfolding)

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
