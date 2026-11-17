from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from firstnet.dataset import InfraredSequenceDataset
from firstnet.metrics import MetricAccumulator
from firstnet.model import FIRSTNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a FIRST-Net checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=r"F:\FIRST-NET")
    parser.add_argument("--dataset", default=None, help="Dataset folder or prepared FIRST source selector, e.g. FIRST_prepared@SIATD")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--image-size", type=int, default=None, help="Defaults to the checkpoint setting; use 0 for native size.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    return parser.parse_args()


def load_checkpoint(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    checkpoint = load_checkpoint(args.checkpoint, device)
    run_config = checkpoint.get("run_config", {})
    model_config = checkpoint.get("model_config", run_config.get("model", {}))
    dataset_name = args.dataset or run_config.get("dataset", "FIRST_dataset")
    size = args.image_size if args.image_size is not None else run_config.get("image_size", 256)
    image_size = size if size and size > 0 else None
    window_size = model_config.get("window_size", 5)
    model = FIRSTNet(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = InfraredSequenceDataset(
        args.data_root,
        dataset_name,
        args.split,
        window_size=window_size,
        image_size=image_size,
        boundary=run_config.get("boundary", "replicate"),
        val_fraction=args.val_fraction,
        seed=run_config.get("seed", 1),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    metrics = MetricAccumulator()
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            mask = batch["mask"]
            probabilities = torch.sigmoid(model(frames)["logits"]).cpu().numpy()
            metrics.update(probabilities, mask.numpy())
    print(f"Dataset: {dataset_name} | split: {args.split} | samples: {len(dataset)}")
    print(f"Split source: {dataset.split_info['source']}")
    for name, value in metrics.compute().items():
        print(f"{name}: {value:.8g}")


if __name__ == "__main__":
    main()
