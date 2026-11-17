"""Fixed-draw sampling policies, with no increase in optimizer steps."""
from collections import Counter

import torch
from torch.utils.data import WeightedRandomSampler


def make_sampler(dataset, policy, seed):
    if policy == "uniform":
        return None
    if not hasattr(dataset, "_prepared"):
        raise ValueError("Balanced policies require prepared manifest records")
    records = dataset.records
    if policy == "real_synthetic":
        groups = ["synthetic" if row.get("generated", False) else "real" for row in records]
    elif policy == "source_balanced":
        groups = [row["source_dataset"] for row in records]
    else:
        raise ValueError("Unknown sampler policy")
    counts = Counter(groups)
    weights = torch.tensor([1.0/counts[group] for group in groups], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True,
                                 generator=torch.Generator().manual_seed(seed))
