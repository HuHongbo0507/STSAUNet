"""Filesystem indexing and deterministic sequence splits for FIRST-Net."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
import re

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_SEQUENCE_RE = re.compile(r"(Sequence\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class FrameRecord:
    dataset: str
    sequence: str
    frame_index: int
    image_path: Path
    mask_path: Path

    @property
    def name(self) -> str:
        return self.image_path.name


def dataset_root(data_root: str | Path, dataset: str) -> Path:
    root = Path(data_root)
    if (root / 'manifest.jsonl').is_file():
        return root
    if '@' in dataset:
        dataset = dataset.split('@', 1)[0]
    return root if root.name.lower() == dataset.lower() else root / dataset


def _prepared_records(root, dataset, split='all'):
    path = root / 'manifest.jsonl' if split == 'all' else root / 'splits' / f'{split}.jsonl'
    source = dataset.split('@', 1)[1] if '@' in dataset else None
    records = []
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if source and row['source_dataset'] != source:
                continue
            if len(row['images']) != 7 or len(row['masks']) != 7 or row['center_frame'] != 4:
                raise ValueError('Invalid FIRST prepared clip')
            sequence = f"{row['source_dataset']}/{row['source_sequence']}/{row['clip_id']}"
            for index, (image, mask) in enumerate(zip(row['images'], row['masks']), 1):
                records.append(FrameRecord(dataset, sequence, index, root / image, root / mask))
    return records


def _sequence_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    return (int(match.group(1)) if match else -1, name)


def _frame_sort_key(record: FrameRecord) -> tuple[int, str]:
    return (record.frame_index, record.image_path.name.lower())


def _first_sequence_and_frame(stem: str) -> tuple[str, int]:
    match = re.match(r"^(.*)-(\d+)$", stem)
    if match:
        return match.group(1), int(match.group(2))
    numbers = re.findall(r"\d+", stem)
    return (stem, int(numbers[-1]) if numbers else 0)


def _list_sequence_and_stem(line: str) -> tuple[str, str] | None:
    clean = line.strip().replace("\\", "/")
    if not clean:
        return None
    match = _SEQUENCE_RE.search(clean)
    if not match:
        return None
    parts = [part for part in clean.split("/") if part]
    filename = parts[-1] if parts else ""
    return match.group(1), Path(filename).stem


def _first_mapping_path() -> Path:
    return Path(__file__).resolve().parent.parent / "first_dataset_pairs.csv"


def _read_first_mapping(path: Path) -> dict[str, tuple[str, int]]:
    if not path.is_file():
        return {}
    mapping: dict[str, tuple[str, int]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            image_sequence = (row.get("image_sequence") or "").strip()
            mask_sequence = (row.get("mask_sequence") or "").strip()
            offset_text = (row.get("frame_offset") or "1").strip() or "1"
            if not image_sequence:
                continue
            if not mask_sequence:
                raise ValueError(
                    f"FIRST_dataset mapping is incomplete for image sequence {image_sequence}. "
                    f"Fill mask_sequence in {path} after confirming the annotation correspondence."
                )
            try:
                frame_offset = int(offset_text)
            except ValueError as exc:
                raise ValueError(f"frame_offset must be an integer in {path}: {row}") from exc
            key = image_sequence.lower()
            if key in mapping:
                raise ValueError(f"Duplicate image_sequence {image_sequence} in {path}")
            mapping[key] = (mask_sequence, frame_offset)
    return mapping


def discover_records(data_root: str | Path, dataset: str) -> tuple[Path, list[FrameRecord]]:
    root = dataset_root(data_root, dataset)
    if (root / 'manifest.jsonl').is_file():
        return root, _prepared_records(root, dataset)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    records: list[FrameRecord] = []
    errors: list[str] = []
    if dataset == "FIRST_dataset":
        image_dir = root / "FIRST"
        mask_dir = root / "FIRSTmasks"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(
                f"Expected FIRST and FIRSTmasks directories under {root}"
            )
        mapping_path = _first_mapping_path()
        mapping = _read_first_mapping(mapping_path)
        mask_lookup: dict[tuple[str, int], Path] = {}
        for mask in mask_dir.iterdir():
            if not mask.is_file() or mask.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            sequence, frame = _first_sequence_and_frame(mask.stem)
            key = (sequence.lower(), frame)
            if key in mask_lookup:
                errors.append(f"duplicate mask sequence/frame: {mask}")
            mask_lookup[key] = mask

        images = [
            image for image in image_dir.iterdir()
            if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
        ]
        images.sort(key=lambda p: p.name.lower())
        used_masks: set[Path] = set()
        for image in images:
            sequence, frame = _first_sequence_and_frame(image.stem)
            mask_sequence, frame_offset = mapping.get(sequence.lower(), (sequence, 1))
            mask = mask_lookup.get((mask_sequence.lower(), frame + frame_offset))
            if mask is None:
                errors.append(
                    f"no mask mapping for image {image.name}; expected "
                    f"{mask_sequence}-{frame + frame_offset}<extension> using {mapping_path.name}"
                )
                continue
            if mask in used_masks:
                errors.append(f"mask assigned to more than one image: {mask}")
                continue
            used_masks.add(mask)
            records.append(FrameRecord(dataset, sequence, frame, image, mask))
        unused_masks = sorted(set(mask_lookup.values()) - used_masks)
        if unused_masks:
            errors.append(
                "unmatched mask files: " + ", ".join(p.name for p in unused_masks[:12])
                + (f" and {len(unused_masks) - 12} more" if len(unused_masks) > 12 else "")
            )
        if errors:
            preview = "\n".join(f"  - {item}" for item in errors[:20])
            suffix = f"\n  ... and {len(errors) - 20} more" if len(errors) > 20 else ""
            raise ValueError(
                f"FIRST_dataset pairing/layout errors ({len(errors)}):\n{preview}{suffix}\n"
                f"Image groups and mask groups do not fully match. Confirm the labels, then edit {mapping_path} "
                "with image_sequence,mask_sequence,frame_offset rows. Do not infer a mapping from sort order."
            )
    else:
        sequence_dirs = sorted(
            (p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"Sequence\d+", p.name, re.I)),
            key=lambda p: _sequence_sort_key(p.name),
        )
        if not sequence_dirs:
            raise FileNotFoundError(f"No Sequence* directories found under {root}")
        for sequence_dir in sequence_dirs:
            image_dir = sequence_dir / "images"
            mask_dir = sequence_dir / "masks"
            if not image_dir.is_dir() or not mask_dir.is_dir():
                errors.append(f"{sequence_dir} must contain images/ and masks/")
                continue
            for image in sorted(image_dir.iterdir(), key=lambda p: p.name.lower()):
                if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                mask = mask_dir / image.name
                if not mask.is_file():
                    errors.append(f"missing mask for {image}")
                    continue
                numbers = re.findall(r"\d+", image.stem)
                frame = int(numbers[-1]) if numbers else 0
                records.append(FrameRecord(dataset, sequence_dir.name, frame, image, mask))

    if errors:
        preview = "\n".join(f"  - {item}" for item in errors[:20])
        suffix = f"\n  ... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ValueError(f"Dataset pairing/layout errors ({len(errors)}):\n{preview}{suffix}")
    if not records:
        raise ValueError(f"No supported image files found under {root}")
    records.sort(key=lambda r: (_sequence_sort_key(r.sequence), *_frame_sort_key(r)))
    return root, records


def _official_list(root: Path, role: str, lookup: dict[tuple[str, str], FrameRecord]) -> list[FrameRecord] | None:
    candidates = [root / f"{role}1.txt", root / f"{role}.txt"]
    list_path = next((p for p in candidates if p.is_file()), None)
    if list_path is None:
        return None

    rows: list[FrameRecord] = []
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for line_number, raw in enumerate(list_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        parsed = _list_sequence_and_stem(raw)
        if parsed is None:
            continue
        sequence, stem = parsed
        key = (sequence.lower(), stem.lower())
        record = lookup.get(key)
        if record is None:
            errors.append(f"{list_path.name}:{line_number}: no image/mask pair for {raw.strip()}")
            continue
        if key in seen:
            errors.append(f"{list_path.name}:{line_number}: duplicate frame {raw.strip()}")
            continue
        seen.add(key)
        rows.append(record)
    if errors:
        preview = "\n".join(f"  - {item}" for item in errors[:10])
        suffix = f"\n  ... and {len(errors) - 10} more" if len(errors) > 10 else ""
        raise ValueError(f"Invalid {role} list {list_path}:\n{preview}{suffix}")
    if not rows:
        return None
    return rows


def _partition_sequences(sequences: list[str], val_fraction: float, test_fraction: float) -> dict[str, set[str]]:
    ordered = sorted(set(sequences), key=_sequence_sort_key)
    count = len(ordered)
    if count < 3:
        raise ValueError(
            f"At least three sequences are required for train/val/test splitting; found {count}: {ordered}"
        )
    test_count = max(1, round(count * test_fraction))
    val_count = max(1, round(count * val_fraction))
    while count - test_count - val_count < 1:
        if test_count >= val_count and test_count > 1:
            test_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            raise ValueError(f"Cannot create non-empty train/val/test splits from {count} sequences")
    train_end = count - val_count - test_count
    val_end = count - test_count
    return {
        "train": set(ordered[:train_end]),
        "val": set(ordered[train_end:val_end]),
        "test": set(ordered[val_end:]),
    }


def split_records(
    data_root: str | Path,
    dataset: str,
    split: str,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[Path, list[FrameRecord], dict[str, object]]:
    """Return one split. NUDT official sequence lists are preferred when available."""
    if split not in {"train", "val", "test", "all"}:
        raise ValueError("split must be train, val, test, or all")
    root, all_records = discover_records(data_root, dataset)
    if (root / 'manifest.jsonl').is_file():
        selected = all_records if split == 'all' else _prepared_records(root, dataset, split)
        return root, selected, {'source': 'explicit FIRST prepared manifest; original-parent grouped splits'}
    if split == "all":
        return root, all_records, {
            "source": "all indexed pairs",
            "sequences": sorted({r.sequence for r in all_records}, key=_sequence_sort_key),
        }

    lookup = {(r.sequence.lower(), r.image_path.stem.lower()): r for r in all_records}
    official_train = None if dataset == "FIRST_dataset" else _official_list(root, "train", lookup)
    official_test = None if dataset == "FIRST_dataset" else _official_list(root, "test", lookup)

    if official_train and official_test:
        train_sequences = sorted({r.sequence for r in official_train}, key=_sequence_sort_key)
        test_sequences = sorted({r.sequence for r in official_test}, key=_sequence_sort_key)
        train_test_overlap = set(s.lower() for s in train_sequences) & set(s.lower() for s in test_sequences)
        if train_test_overlap:
            raise ValueError(f"Official train/test sequence lists overlap: {sorted(train_test_overlap)}")
        all_sequences = {r.sequence for r in all_records}
        listed_sequences = set(train_sequences) | set(test_sequences)
        unused_sequences = all_sequences - listed_sequences
        if unused_sequences:
            val_sequences = unused_sequences
            fit_sequences = set(train_sequences)
            selected_by_split = {
                "train": [r for r in official_train],
                "val": [r for r in all_records if r.sequence in val_sequences],
                "test": [r for r in official_test],
            }
            source = "official train1/test1 or train/test lists; unlisted complete sequences reserved for validation"
        else:
            if len(train_sequences) < 2:
                raise ValueError("Official training split needs at least two sequences to reserve validation.")
            val_count = max(1, round(len(train_sequences) * val_fraction))
            val_count = min(val_count, len(train_sequences) - 1)
            val_sequences = set(train_sequences[-val_count:])
            fit_sequences = set(train_sequences) - val_sequences
            selected_by_split = {
                "train": [r for r in official_train if r.sequence in fit_sequences],
                "val": [r for r in official_train if r.sequence in val_sequences],
                "test": [r for r in official_test],
            }
            source = "official train1/test1 or train/test lists; validation held out by whole training sequence"
        selected = selected_by_split[split]
        info = {
            "source": source,
            "train_sequences": sorted(fit_sequences, key=_sequence_sort_key),
            "val_sequences": sorted(val_sequences, key=_sequence_sort_key),
            "test_sequences": test_sequences,
            "seed": seed,
        }
    else:
        partition = _partition_sequences([r.sequence for r in all_records], val_fraction, 0.2)
        chosen = partition[split]
        selected = [r for r in all_records if r.sequence in chosen]
        info = {
            "source": "deterministic sequence-level split; no usable official lists found",
            "train_sequences": sorted(partition["train"], key=_sequence_sort_key),
            "val_sequences": sorted(partition["val"], key=_sequence_sort_key),
            "test_sequences": sorted(partition["test"], key=_sequence_sort_key),
            "seed": seed,
        }

    selected.sort(key=lambda r: (_sequence_sort_key(r.sequence), *_frame_sort_key(r)))
    if not selected:
        raise ValueError(f"Split '{split}' is empty for {dataset}. Split summary: {info}")
    return root, selected, info


def window_records(
    records: list[FrameRecord],
    center_index: int,
    window_size: int = 5,
    boundary: str = "replicate",
) -> list[FrameRecord]:
    """Build an odd-sized same-sequence window with replicate or reflect boundaries."""
    if window_size < 1 or window_size % 2 != 1:
        raise ValueError("window_size must be a positive odd integer")
    if boundary not in {"replicate", "reflect"}:
        raise ValueError("boundary must be replicate or reflect")
    center = records[center_index]
    same_sequence = [r for r in records if r.sequence == center.sequence]
    positions = {id(r): i for i, r in enumerate(same_sequence)}
    center_pos = positions.get(id(center))
    if center_pos is None:
        raise ValueError("center record is not present in its sequence records")
    radius = window_size // 2
    n = len(same_sequence)
    result = []
    for offset in range(-radius, radius + 1):
        pos = center_pos + offset
        if boundary == "replicate" or n == 1:
            pos = max(0, min(n - 1, pos))
        else:
            period = 2 * n - 2
            pos %= period
            if pos >= n:
                pos = period - pos
        result.append(same_sequence[pos])
    return result
