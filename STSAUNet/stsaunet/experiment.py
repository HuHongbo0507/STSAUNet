"""Audits and real-sample checks for the prepared STSAU experiment."""
from __future__ import annotations

from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import PIL
import torch
from torch.utils.data import DataLoader

from .losses import STSAUNetLoss
from .model import STSAUNet


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_prepared(root):
    """Audit active JSONL indexes only; never walk sources or quarantine."""
    root = Path(root).resolve()
    parents = {}
    manifest = {}
    summaries = {}
    errors = []
    for label in ("manifest", "train", "val", "test"):
        path = root / "manifest.jsonl" if label == "manifest" else root / "splits" / f"{label}.jsonl"
        sources, annotations, groups = Counter(), Counter(), set()
        keys = set()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                key = (row["source_dataset"], row["source_sequence"], row["clip_id"])
                if key in keys:
                    errors.append(f"duplicate {label}: {key}")
                keys.add(key)
                parent = (row["source_dataset"], row["parent_sequence"])
                groups.add(parent)
                split = row["split"]
                if split not in ("train", "val", "test"):
                    errors.append(f"invalid split: {key}")
                if label != "manifest" and split != label:
                    errors.append(f"split mismatch: {label} {key}")
                if parent in parents and parents[parent] != split:
                    errors.append(f"parent split leakage: {parent}")
                parents[parent] = split
                if row["source_dataset"] == "WideIRSTD" and split != "train":
                    errors.append("WideIRSTD must be training only")
                ids = row["source_frame_ids"]
                if (len(ids) != 7 or any(b != a + 1 for a, b in zip(ids, ids[1:]))
                        or row["center_frame"] != 4):
                    errors.append(f"invalid frame order/center: {key}")
                prefix = Path("sources").joinpath(*key)
                for kind in ("images", "masks"):
                    paths = row[kind]
                    if len(paths) != 7:
                        errors.append(f"not seven {kind}: {key}")
                    for index, relative in enumerate(paths, 1):
                        # Exact expected relative paths exclude quarantine, traversal and cross-clip windows.
                        if Path(relative) != prefix / kind / f"{index:05d}.png":
                            errors.append(f"unexpected {kind} path: {relative}")
                digest = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
                if label == "manifest":
                    manifest[key] = (split, digest)
                elif manifest.get(key) != (label, digest):
                    errors.append(f"split entry differs from manifest: {label} {key}")
                sources[row["source_dataset"]] += 1
                annotations[row["annotation_type"]] += 1
        if label != "manifest" and keys != {key for key, value in manifest.items() if value[0] == label}:
            errors.append(f"split coverage mismatch: {label}")
        summaries[label] = dict(samples=len(keys), parents=len(groups), by_source=dict(sources),
                                by_annotation=dict(annotations), sha256=sha256(path))
    result = dict(passed=not errors, root=str(root), splits=summaries, parent_split_overlap=0 if not errors else None,
                  errors=errors[:50], scope="Active manifests, exact split equality, frame order, clip paths and parent groups; no quarantine scan.")
    validation = root / "reports" / "validation.json"
    if validation.exists():
        result["prior_full_decode_validation"] = json.loads(validation.read_text(encoding="utf-8"))
        result["prior_validation_sha256"] = sha256(validation)
    if errors:
        raise ValueError(json.dumps(result, ensure_ascii=False))
    return result


def environment_info():
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                torch=torch.__version__, cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                cuda_available=torch.cuda.is_available(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                numpy=np.__version__, pillow=PIL.__version__, pid=os.getpid(), command=sys.argv,
                precision="float32 (no AMP)", tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32)


def check_unfolding(dataset, model_config, batch_size, device, lr, reconstruction_weight, alignment_weight,
                    loss_config=None):
    """Full configured model, native-size real TRAIN samples and a throwaway Adam step."""
    start = time.perf_counter()
    batch = next(iter(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)))
    frames = batch["frames"].to(device)
    image, mask = batch["image"].to(device), batch["mask"].to(device)
    b, t, c, h, w = frames.shape
    assert (t, c, h, w) == (7, 1, 256, 256)
    assert torch.equal(frames[:, 3], image)
    assert torch.all((mask == 0) | (mask == 1))
    assert frames.min() >= 0 and frames.max() <= 1
    packed = frames.reshape(b, t * c, h, w)
    assert torch.equal(packed.reshape(b, t, c, h, w), frames)
    for bi in range(b):
        for ti in range(t):
            assert torch.equal(packed[bi, ti * c:(ti + 1) * c], frames[bi, ti])
    traces = {}
    handles = []
    def hook(name):
        def capture(module, inputs, output):
            traces[name] = dict(inputs=[list(x.shape) for x in inputs],
                                outputs=[list(x.shape) for x in (output if isinstance(output, tuple) else [output])])
        return capture
    # Legacy computation is exercised explicitly to document the old stage input.
    legacy = STSAUNet(**dict(model_config, unfolding_mode="framewise")).to(device).eval()
    handles = [legacy.separation.register_forward_hook(hook("before.FeatureSeparation")),
               legacy.separation.stages[0].register_forward_hook(hook("before.DeepUnfoldingStage.0"))]
    altered = frames.clone()
    altered[0, 2] = (altered[0, 2] + 0.1).clamp(0, 1)
    assert torch.equal(altered[:, 3], frames[:, 3])
    with torch.no_grad():
        legacy_base = legacy.separation(frames.reshape(b*t, c, h, w))[1].reshape(b,t,c,h,w)[:,3].clone()
        legacy_changed = legacy.separation(altered.reshape(b*t,c,h,w))[1].reshape(b,t,c,h,w)[:,3]
        legacy_delta = float((legacy_base - legacy_changed).abs().max())
    for handle in handles:
        handle.remove()
    del legacy, legacy_base, legacy_changed
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = STSAUNet(**model_config).to(device)
    handles = [model.register_forward_hook(lambda m,i,o: traces.update({"after.STSAUNet": dict(input=list(i[0].shape), logits=list(o["logits"].shape))})),
               model.separation.register_forward_hook(hook("after.FeatureSeparation"))]
    handles.extend(stage.register_forward_hook(hook(f"after.DeepUnfoldingStage.{index}"))
                   for index, stage in enumerate(model.separation.stages))
    model.eval()
    # cuDNN may select different TF32 kernels for batch=1 and batch=2. Disable
    # TF32 only for this numerical consistency check, not for formal training.
    with torch.no_grad(), torch.backends.cudnn.flags(allow_tf32=False):
        base = model.separation(packed)[1]
        changed = model.separation(altered.reshape(b,t*c,h,w))[1]
        delta = float((base[0,3] - changed[0,3]).abs().max())
        assert legacy_delta == 0.0, legacy_delta
        assert delta > 1e-8, delta
        independent_delta = float((base[1:] - changed[1:]).abs().max()) if b > 1 else None
        if independent_delta is not None:
            assert independent_delta == 0.0, independent_delta
        singles = torch.cat([model.separation(packed[i:i+1])[1] for i in range(b)])
        batch_error = float((singles - base).abs().max())
        assert torch.allclose(singles, base, atol=1e-5, rtol=1e-4), batch_error
    del base, changed, singles, altered
    model.train()
    criterion = STSAUNetLoss(reconstruction_weight, alignment_weight, **(loss_config or {}))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    output = model(frames)
    assert output["logits"].shape == mask.shape == output["reconstruction"].shape
    assert all(torch.isfinite(value).all() for value in output.values())
    losses = criterion(output, image, mask, frames)
    assert all(torch.isfinite(value).all() for value in losses.values())
    optimizer.zero_grad(set_to_none=True)
    losses["loss"].backward()
    missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    before = model.separation.stages[0].background_prox[0].weight.detach().clone()
    optimizer.step()
    update = float((before - model.separation.stages[0].background_prox[0].weight).abs().max())
    assert update > 0
    result = dict(passed=True, dataset_sample_shape=list(dataset[0]["frames"].shape),
                  loader_shape=list(frames.shape), packed_shape=list(packed.shape), logits_shape=list(mask.shape),
                  model_config=model_config, traces=traces, packing="packed[b,t*C+c,h,w]=frames[b,t,c,h,w]",
                  exact_pack_round_trip=True, frame_order_verified=True, center_unchanged_during_perturbation=True,
                  legacy_center_target_delta=legacy_delta, joint_center_target_delta=delta,
                  other_batch_sample_delta=independent_delta, batched_vs_single_max_error=batch_error,
                  batch_consistency_precision="float32; cuDNN TF32 disabled within this check only",
                  losses={key:float(value.detach()) for key,value in losses.items()},
                  parameters_without_gradient=missing, leading_unfold_weight_update_max=update,
                  cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=="cuda" else None,
                  elapsed_seconds=time.perf_counter()-start, sample_sequences=list(batch["sequence"]),
                  note="Throwaway model/optimizer; formal training is re-seeded and starts from scratch.")
    for handle in handles:
        handle.remove()
    del output, losses, optimizer, model, frames, packed, image, mask
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
