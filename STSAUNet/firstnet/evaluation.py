"""Single-pass grouped evaluation, with fixed-threshold center-frame examples."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from .metrics import MetricAccumulator


def save_example(destination, batch, index, probability, threshold=0.5):
    destination.mkdir(parents=True, exist_ok=False)
    image = np.uint8(np.clip(batch["image"][index,0].cpu().numpy(),0,1)*255)
    target = np.uint8(batch["mask"][index,0].cpu().numpy() >= 0.5)*255
    probability = np.asarray(probability, dtype=np.float32)
    binary = np.uint8(probability >= threshold)*255
    arrays = [image, target, np.uint8(np.clip(probability,0,1)*255), binary]
    for name, array in zip(("image", "ground_truth", "probability", "prediction"), arrays):
        Image.fromarray(array).save(destination / f"{name}.png")
    np.save(destination / "probability.npy", probability)
    canvas = Image.new("RGB", (256*4, 280), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (label, array) in enumerate(zip(("Center frame", "Ground truth", "Probability", f"Prediction >= {threshold:.4g}"), arrays)):
        draw.text((256*column+5, 5), label, fill="black")
        canvas.paste(Image.fromarray(array).convert("RGB"), (256*column,24))
    canvas.save(destination / "comparison.png")


@torch.no_grad()
def evaluate_loader(model, loader, device, criterion=None, compute_objects=False,
                    examples_dir=None, report=print, progress_every=200):
    model.eval()
    accumulators = {}
    counts = defaultdict(int)
    loss_sums = defaultdict(float)
    example_counts = defaultdict(int)
    example_records = []
    start = time.perf_counter()
    seen = 0
    for step, batch in enumerate(loader, 1):
        frames = batch["frames"].to(device, non_blocking=True)
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        output = model(frames)
        if not torch.isfinite(output["logits"]).all():
            raise FloatingPointError("Non-finite evaluation logits")
        count = frames.shape[0]
        if criterion is not None:
            losses = criterion(output, image, mask, frames)
            for key,value in losses.items():
                number = float(value)
                if not np.isfinite(number):
                    raise FloatingPointError(f"Non-finite evaluation {key}")
                loss_sums[key] += number*count
        probabilities = torch.sigmoid(output["logits"]).cpu().numpy()
        truth = batch["mask"].numpy()
        for index in range(count):
            source = batch.get("source_dataset", ["unspecified"]*count)[index]
            annotation = batch.get("annotation_type", ["pixel_mask"]*count)[index]
            groups = ("overall", f"source/{source}", f"annotation/{annotation}")
            for group in groups:
                if group not in accumulators:
                    accumulators[group] = MetricAccumulator(compute_objects=compute_objects)
                accumulators[group].update(probabilities[index:index+1], truth[index:index+1])
                counts[group] += 1
            if examples_dir is not None and example_counts[source] < 2:
                number = example_counts[source]+1
                relative = f"{source}/sample-{number:02d}"
                save_example(Path(examples_dir)/relative, batch, index, probabilities[index,0])
                example_records.append(dict(path=relative,sequence=batch["sequence"][index],
                                            name=batch["name"][index],annotation_type=annotation))
                example_counts[source] += 1
        seen += count
        if progress_every and (step % progress_every == 0 or step == len(loader)):
            report(f"evaluation step {step}/{len(loader)} | samples={seen} elapsed={time.perf_counter()-start:.1f}s")
    if seen != len(loader.dataset):
        raise RuntimeError(f"Incomplete evaluation: {seen}/{len(loader.dataset)} samples")
    return dict(samples=seen, elapsed_seconds=time.perf_counter()-start,
                losses={key:value/seen for key,value in loss_sums.items()},
                groups={key:dict(samples=counts[key], **value.compute()) for key,value in accumulators.items()},
                threshold=0.5, object_metrics=compute_objects, examples=example_records,
                metric_definitions=dict(IoU="pooled TP/(TP+FP+FN)",Precision="TP/(TP+FP)",Recall="TP/(TP+FN)",
                                        F1="2*Precision*Recall/(Precision+Recall)",
                                        PD="8-connected component centroids; greedy one-to-one matches within 3 pixels",
                                        FA="false-positive pixels / ground-truth background pixels"),
                annotation_note="bbox_weak scores use filled boxes, not true pixel segmentation; WideIRSTD is training only.")
