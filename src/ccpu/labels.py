"""Validate, deduplicate, and split labelled text."""
import csv
import hashlib
import random
from collections import Counter
from pathlib import Path
from .common import utc_now, write_csv, write_json


def read_labels(path):
    csv.field_size_limit(100_000_000)
    records, seen = [], {}
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"text", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("Labelled CSV requires text and label columns")
        for i, row in enumerate(reader, 2):
            text, label = row["text"], row["label"]
            if not text.strip() or label not in ("0", "1"):
                raise ValueError(f"Empty text or non-binary label at line {i}")
            key = hashlib.sha256(text.encode()).hexdigest()
            if key in seen:
                if seen[key] != label:
                    raise ValueError(f"Conflicting labels for identical text at line {i}")
                continue
            seen[key] = label
            records.append({**row, "label": int(label), "text_sha256": key})
    if not records:
        raise ValueError("No labelled examples")
    return records


def split_labels(path, output, seed=666, train_fraction=0.7, val_fraction=0.15, roles=("main",)):
    """Stratified (by label) train/validation/test split; all input columns are kept.

    Only rows whose `role` is in ``roles`` are used: the classifier is applied to
    keyword candidates only, so the non-candidate filter-check rows stay out.
    """
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1 or train_fraction + val_fraction > 1:
        raise ValueError("Invalid train/validation proportions")
    records = [r for r in read_labels(path) if "role" not in r or r["role"] in roles]
    buckets = {k: [r for r in records if r["label"] == k] for k in (0, 1)}
    rng = random.Random(seed)
    splits = {k: [] for k in ("train", "validation", "test")}
    for label, values in buckets.items():
        rng.shuffle(values)
        a, b = int(len(values) * train_fraction), int(len(values) * (train_fraction + val_fraction))
        if a < 1 or b - a < 1:
            raise ValueError(f"Too few examples of class {label} for requested split")
        splits["train"].extend(values[:a])
        splits["validation"].extend(values[a:b])
        splits["test"].extend(values[b:])
    output = Path(output)
    fields = list(records[0])
    for name, values in splits.items():
        rng.shuffle(values)
        if values:
            write_csv(output / f"{name}.csv", fields, values)
    summary = {name: {"rows": len(values), "class_counts": dict(Counter(r["label"] for r in values))}
               for name, values in splits.items()}
    write_json(output / "split.json", {"generated_at_utc": utc_now(), "seed": seed,
        "train_fraction": train_fraction, "val_fraction": val_fraction,
        "identical_text_policy": "deduplicated before splitting; conflicting labels fail",
        "splits": summary})
    return summary


def validate_disjoint(*datasets):
    seen = set()
    for records in datasets:
        current = {r["text_sha256"] for r in records}
        if seen & current:
            raise ValueError("Identical texts occur across train/validation/test sets")
        seen.update(current)
