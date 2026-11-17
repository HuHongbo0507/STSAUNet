"""Pixel segmentation metrics and optional connected-component PD/FA."""
from __future__ import annotations

import numpy as np


def _components(mask: np.ndarray) -> list[tuple[float, float]]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    centers: list[tuple[float, float]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            points = []
            while stack:
                cy, cx = stack.pop()
                points.append((cy, cx))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            centers.append((sum(p[1] for p in points) / len(points), sum(p[0] for p in points) / len(points)))
    return centers


class MetricAccumulator:
    def __init__(self, threshold: float = 0.5, pd_radius: float = 3.0):
        self.threshold = threshold
        self.pd_radius = pd_radius
        self.reset()

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.background_pixels = 0
        self.detected_targets = 0
        self.total_targets = 0

    def update(self, probabilities, target) -> None:
        pred = np.asarray(probabilities)
        gt = np.asarray(target)
        if pred.ndim == 4:
            pred = pred[:, 0]
        if gt.ndim == 4:
            gt = gt[:, 0]
        if pred.ndim == 2:
            pred = pred[None, ...]
        if gt.ndim == 2:
            gt = gt[None, ...]
        pred_binary = pred >= self.threshold
        gt_binary = gt >= 0.5
        if pred_binary.shape != gt_binary.shape:
            raise ValueError(f"Prediction/label shape mismatch: {pred_binary.shape} vs {gt_binary.shape}")
        self.tp += int(np.logical_and(pred_binary, gt_binary).sum())
        self.fp += int(np.logical_and(pred_binary, ~gt_binary).sum())
        self.fn += int(np.logical_and(~pred_binary, gt_binary).sum())
        self.background_pixels += int((~gt_binary).sum())
        for pred_map, gt_map in zip(pred_binary, gt_binary):
            predicted_centers = _components(pred_map)
            target_centers = _components(gt_map)
            self.total_targets += len(target_centers)
            pairs = []
            for ti, (tx, ty) in enumerate(target_centers):
                for pi, (px, py) in enumerate(predicted_centers):
                    distance = ((tx - px) ** 2 + (ty - py) ** 2) ** 0.5
                    if distance <= self.pd_radius:
                        pairs.append((distance, ti, pi))
            used_targets: set[int] = set()
            used_predictions: set[int] = set()
            for _, ti, pi in sorted(pairs):
                if ti not in used_targets and pi not in used_predictions:
                    used_targets.add(ti)
                    used_predictions.add(pi)
            self.detected_targets += len(used_targets)

    def compute(self) -> dict[str, float]:
        eps = 1e-12
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        pd = self.detected_targets / (self.total_targets + eps)
        fa = self.fp / (self.background_pixels + eps)
        return {
            "IoU": float(iou),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1": float(f1),
            "PD": float(pd),
            "FA": float(fa),
            "TP": float(self.tp),
            "FP": float(self.fp),
            "FN": float(self.fn),
            "targets": float(self.total_targets),
        }
