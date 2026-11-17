from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from stsaunet.dataset import InfraredSequenceDataset
from stsaunet.model import STSAUNet
from stsaunet.evaluation import evaluate_loader
from stsaunet.experiment import write_json
from stsaunet.losses import STSAUNetLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a STSAUNet checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True, help="External dataset directory or its parent.")
    parser.add_argument("--dataset", default=None, help="Dataset folder or prepared STSAU source selector, e.g. STSAU_prepared@SIATD")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--image-size", type=int, default=None, help="Defaults to the checkpoint setting; use 0 for native size.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--examples-dir", default=None)
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
    model_config = dict(checkpoint.get("model_config", run_config.get("model", {})))
    model_config.setdefault("unfolding_mode", "framewise")
    dataset_name = args.dataset or run_config.get("dataset", "STSAU_dataset")
    size = args.image_size if args.image_size is not None else run_config.get("image_size", 256)
    image_size = size if size and size > 0 else None
    window_size = model_config.get("window_size", 5)
    model = STSAUNet(**model_config).to(device)
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
    criterion = STSAUNetLoss(run_config.get("reconstruction_weight",0.01),run_config.get("alignment_weight",0.01),
                             **run_config.get("loss_config", {}))
    results = evaluate_loader(model,loader,device,criterion,compute_objects=True,examples_dir=args.examples_dir)
    if args.output_json:
        write_json(args.output_json,results)
    print(f"Dataset: {dataset_name} | split: {args.split} | samples: {len(dataset)}")
    print(f"Split source: {dataset.split_info['source']}")
    import json
    print(json.dumps(results["groups"],indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
