from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from firstnet.dataset import InfraredSequenceDataset
from firstnet.losses import FIRSTNetLoss
from firstnet.model import FIRSTNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FIRST-Net.")
    parser.add_argument("--data-root", default=r"F:\FIRST-NET")
    parser.add_argument("--dataset", default="NUDT-MIRSDT", help="Dataset folder; prepared FIRST also accepts FIRST_prepared@SIATD etc.")
    parser.add_argument("--output-dir", default=None, help="Defaults to runs/<dataset>-<timestamp>.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256, help="Use 0 to preserve the native image size.")
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--boundary", choices=("replicate", "reflect"), default="replicate")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--stages", type=int, default=6)
    parser.add_argument("--unfold-channels", type=int, default=32)
    parser.add_argument("--feature-channels", type=int, default=32)
    parser.add_argument("--max-offset", type=float, default=3.0)
    parser.add_argument("--reconstruction-weight", type=float, default=0.01)
    parser.add_argument("--alignment-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this Python environment.")
    return device


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("epochs and batch-size must be positive; workers cannot be negative")
    if args.window_size % 2 != 1:
        raise ValueError("window-size must be odd")
    set_seed(args.seed)
    device = choose_device(args.device)
    image_size = args.image_size if args.image_size > 0 else None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("runs") / f"{args.dataset}-{stamp}"

    common_data = dict(
        data_root=args.data_root,
        dataset=args.dataset,
        window_size=args.window_size,
        image_size=image_size,
        boundary=args.boundary,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    train_data = InfraredSequenceDataset(split="train", **common_data)
    val_data = InfraredSequenceDataset(split="val", **common_data)
    output_dir.mkdir(parents=True, exist_ok=False)

    def report(message: str) -> None:
        print(message)
        with (output_dir / "train.log").open("a", encoding="utf-8") as log_stream:
            log_stream.write(message + "\n")

    report(f"Device: {device}")
    report(f"Train samples: {len(train_data)} | sequences: {len(train_data.sequence_records)}")
    report(f"Validation samples: {len(val_data)} | sequences: {len(val_data.sequence_records)}")
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    model_config = {
        "window_size": args.window_size,
        "stages": args.stages,
        "unfold_channels": args.unfold_channels,
        "feature_channels": args.feature_channels,
        "max_offset": args.max_offset,
    }
    run_config = vars(args).copy()
    run_config.update({
        "image_size": image_size,
        "model": model_config,
        "device": str(device),
        "train_summary": train_data.summary(),
        "val_summary": val_data.summary(),
    })
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")

    model = FIRSTNet(**model_config).to(device)
    criterion = FIRSTNetLoss(args.reconstruction_weight, args.alignment_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    best_iou = -1.0
    history_path = output_dir / "history.csv"

    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_iou", "val_precision", "val_recall", "val_f1", "lr"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_total = 0.0
            train_count = 0
            for batch in train_loader:
                frames = batch["frames"].to(device, non_blocking=True)
                image = batch["image"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                output = model(frames)
                losses = criterion(output, image, mask)
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                optimizer.step()
                train_total += float(losses["loss"].detach())
                train_count += 1

            model.eval()
            val_total = 0.0
            val_count = 0
            tp = fp = fn = 0
            with torch.no_grad():
                for batch in val_loader:
                    frames = batch["frames"].to(device, non_blocking=True)
                    image = batch["image"].to(device, non_blocking=True)
                    mask = batch["mask"].to(device, non_blocking=True)
                    output = model(frames)
                    losses = criterion(output, image, mask)
                    val_total += float(losses["loss"])
                    val_count += 1
                    pred = torch.sigmoid(output["logits"]) >= 0.5
                    truth = mask >= 0.5
                    tp += int((pred & truth).sum().item())
                    fp += int((pred & ~truth).sum().item())
                    fn += int((~pred & truth).sum().item())

            eps = 1e-12
            iou = tp / (tp + fp + fn + eps)
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            train_loss = train_total / max(train_count, 1)
            val_loss = val_total / max(val_count, 1)
            lr = optimizer.param_groups[0]["lr"]
            report(
                f"epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} IoU={iou:.6f} F1={f1:.6f} "
                f"Precision={precision:.6f} Recall={recall:.6f} lr={lr:.2e}"
            )
            writer.writerow([epoch, train_loss, val_loss, iou, precision, recall, f1, lr])
            stream.flush()

            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model_config": model_config,
                "run_config": run_config,
                "val_metrics": {"IoU": iou, "Precision": precision, "Recall": recall, "F1": f1},
            }
            torch.save(checkpoint, output_dir / "last.pt")
            if iou > best_iou:
                best_iou = iou
                torch.save(checkpoint, output_dir / "best.pt")

    report(f"Training complete. Best validation IoU: {best_iou:.6f}")
    report(f"Run directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
