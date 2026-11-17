from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from PIL import Image

from stsaunet.index import discover_records, split_records, window_records


def check_dataset(data_root: str, dataset_name: str, val_fraction: float) -> None:
    from stsaunet.prepared import prepared_root
    root = prepared_root(data_root, dataset_name)
    if root is not None:
        import json
        from stsaunet.experiment import audit_prepared
        from stsaunet.dataset import InfraredSequenceDataset
        audit = audit_prepared(root)
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        for split in ("train", "val", "test"):
            # A source with no val/test data is explicitly allowed to be absent.
            source = dataset_name.split("@",1)[1] if "@" in dataset_name else None
            split_summary = audit["splits"][split]
            if not split_summary["samples"] or (source and not split_summary["by_source"].get(source)):
                print(f"{dataset_name} split={split}: no samples")
                continue
            dataset = InfraredSequenceDataset(data_root,dataset_name,split,window_size=7)
            for index in {0,len(dataset)//2,len(dataset)-1}:
                sample = dataset[index]
                assert tuple(sample["frames"].shape)==(7,1,256,256)
                assert tuple(sample["mask"].shape)==(1,256,256)
            print(f"{dataset_name} split={split}: {len(dataset)} clips; sampled native-size 7-frame decode passed")
        return
    root, records = discover_records(data_root, dataset_name)
    sequences = defaultdict(list)
    dimensions = Counter()
    problems = []
    for record in records:
        sequences[record.sequence].append(record)
        try:
            with Image.open(record.image_path) as image:
                image_size = image.size
                dimensions[image_size] += 1
            with Image.open(record.mask_path) as mask:
                if mask.size != image_size:
                    problems.append(f"image/mask size mismatch: {record.image_path} vs {record.mask_path}")
        except Exception as exc:
            problems.append(f"cannot read {record.image_path} or {record.mask_path}: {exc}")
    for seq_records in sequences.values():
        seq_records.sort(key=lambda r: (r.frame_index, r.name.lower()))

    print(f"\n{dataset_name}: {len(records)} matched image/mask pairs in {root}")
    print(f"Sequences: {len(sequences)}; distinct image sizes: {len(dimensions)}")
    print(f"Most common sizes (width,height): {dimensions.most_common(8)}")
    for name in sorted(sequences, key=lambda s: (int("".join(ch for ch in s if ch.isdigit()) or -1), s)):
        values = sequences[name]
        frame_ids = [r.frame_index for r in values]
        gaps = [f"{a}->{b}" for a, b in zip(frame_ids, frame_ids[1:]) if b != a + 1]
        print(
            f"  {name}: {len(values)} frames, IDs {frame_ids[0]}..{frame_ids[-1]}"
            + (f", gaps={gaps[:5]}" if gaps else "")
        )
    for split in ("train", "val", "test"):
        _, selected, info = split_records(data_root, dataset_name, split, val_fraction=val_fraction)
        by_sequence = defaultdict(list)
        for record in selected:
            by_sequence[record.sequence].append(record)
        for seq_records in by_sequence.values():
            seq_records.sort(key=lambda r: (r.frame_index, r.name.lower()))
            for index in {0, len(seq_records) // 2, len(seq_records) - 1}:
                window = window_records(seq_records, index, 5, "replicate")
                if len(window) != 5 or any(item.sequence != seq_records[index].sequence for item in window):
                    problems.append(f"invalid 5-frame window in {split}, {seq_records[index].sequence}")
        print(f"  {split}: {len(selected)} frames across {len(by_sequence)} sequences ({info['source']})")
    if problems:
        for problem in problems[:20]:
            print(f"ERROR: {problem}")
        raise ValueError(f"Found {len(problems)} data errors.")
    print("Pairing, readable image/mask files, dimensions, and same-sequence 5-frame windows: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate STSAUNet dataset pairing and temporal windows.")
    parser.add_argument("--data-root", required=True, help="External dataset directory or its parent.")
    parser.add_argument("--dataset", required=True, help="Dataset folder or prepared STSAU source selector")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()
    check_dataset(args.data_root, args.dataset, args.val_fraction)


if __name__ == "__main__":
    main()
