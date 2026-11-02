"""Paths, stable serialization and file provenance."""
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_config(path=None):
    path = Path(path or ROOT / "config.json").resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["source_dir"] = str((path.parent / cfg["source_dir"]).resolve())
    local_model = path.parent / cfg.get("base_model", "")
    if local_model.is_dir():
        cfg["base_model"] = str(local_model.resolve())
    elif cfg.get("base_model", "").startswith("models/"):
        cfg["base_model"] = "hfl/chinese-macbert-base"  # fall back to the Hub copy
    return cfg


def rel_path(path):
    """Path relative to the repository root (forward slashes), for logs and metadata.

    Strings that are not local paths, such as a Hugging Face model id, are returned unchanged.
    """
    path = Path(path)
    if not path.is_absolute():
        if not path.exists():
            return path.as_posix()
        path = path.resolve()
    try:
        return Path(os.path.relpath(path, ROOT)).as_posix()
    except ValueError:  # another drive on Windows
        return path.as_posix()


def logger(path):
    """print() that is also appended, with a timestamp, to a log file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def log(message):
        line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    return log


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
