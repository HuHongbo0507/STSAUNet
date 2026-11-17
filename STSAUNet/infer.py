from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from stsaunet.dataset import InfraredSequenceDataset
from stsaunet.model import STSAUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run STSAUNet on one sequence or a dataset.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True, help="External dataset directory or its parent.")
    parser.add_argument("--dataset", default=None, help="Dataset folder or prepared STSAU source selector, e.g. STSAU_prepared@SIATD")
    parser.add_argument("--sequence", default=None, help="Optional sequence name, e.g. Sequence1 or 00003.")
    parser.add_argument("--output-dir", default="predictions")
    parser.add_argument("--image-size", type=int, default=None, help="Defaults to checkpoint setting; use 0 for native size.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    run_config = checkpoint.get("run_config", {})
    model_config = dict(checkpoint.get("model_config", run_config.get("model", {})))
    model_config.setdefault("unfolding_mode", "framewise")
    dataset_name = args.dataset or run_config.get("dataset", "STSAU_dataset")
    size = args.image_size if args.image_size is not None else run_config.get("image_size", 256)
    image_size = size if size and size > 0 else None
    model = STSAUNet(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = InfraredSequenceDataset(
        args.data_root,
        dataset_name,
        "all",
        window_size=model_config.get("window_size", 5),
        image_size=image_size,
        boundary=run_config.get("boundary", "replicate"),
        sequence=args.sequence,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    output_root = Path(args.output_dir) / dataset_name
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            probability_batch = torch.sigmoid(model(frames)["logits"]).cpu().numpy()
            for i, probability in enumerate(probability_batch):
                sequence = batch["sequence"][i]
                name = batch["name"][i]
                height = int(batch["original_size"][0][i])
                width = int(batch["original_size"][1][i])
                probability = probability[0]
                probability_image = Image.fromarray(np.uint8(np.clip(probability, 0, 1) * 255))
                binary_image = Image.fromarray(np.uint8(probability >= 0.5) * 255)
                probability_image = probability_image.resize((width, height), Image.Resampling.BILINEAR)
                binary_image = binary_image.resize((width, height), Image.Resampling.NEAREST)
                destination = output_root / sequence
                (destination / "probability").mkdir(parents=True, exist_ok=True)
                (destination / "mask").mkdir(parents=True, exist_ok=True)
                probability_image.save(destination / "probability" / name)
                binary_image.save(destination / "mask" / name)
    print(f"Saved probability maps and binary masks under: {output_root.resolve()}")


if __name__ == "__main__":
    main()
