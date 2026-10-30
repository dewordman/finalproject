# src/acquire.py
from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.config import require


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hash of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_metadata(cfg):
    if cfg is None:
        return {}
    metadata = {}
    for source in ("enron_spam", "spamassassin", "nazario"):
        url = require(cfg, f"sources.{source}.url")
        entry = require(cfg, f"sources.{source}")
        note = require(cfg, f"sources.{source}.note") if source != "nazario" else entry.get("note", "")
        licence = require(cfg, f"sources.{source}.licence") if source == "nazario" else entry.get("licence", "")
        if any(value is None for value in (url, note, licence)):
            raise ValueError(f"sources.{source} metadata must not contain null values")
        metadata[source] = {"source_url": str(url), "source_licence": str(licence),
                            "source_note": str(note)}
    return metadata


def build_manifest(raw_dir: Union[str, Path], out_csv: Union[str, Path],
                   *, cfg: Optional[Dict[str, Any]] = None) -> None:
    """Inventory every raw file, preserving the original two-argument API."""
    raw_dir, out_csv = Path(raw_dir), Path(out_csv)
    if out_csv.exists():
        raise FileExistsError(f"Manifest already exists and will not be overwritten: {out_csv}")
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw_dir not found: {raw_dir}")
    if out_csv.resolve().is_relative_to(raw_dir.resolve()):
        raise ValueError("The manifest output must be outside raw_dir")
    metadata = _source_metadata(cfg)
    roots = {source: (raw_dir / source).resolve()
             for source in ("enron_spam", "spamassassin", "nazario")}
    if cfg is not None:
        for key in ("main_dir", "external_dir"):
            path = Path(require(cfg, f"nazario.{key}"))
            project_root = Path(cfg.get("_project_root") or Path(__file__).resolve().parents[1])
            roots[key] = (path if path.is_absolute() else project_root / path).resolve()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    files = sorted(path for path in raw_dir.rglob("*") if path.is_file())
    file_info = []
    for path in files:
        relative = path.relative_to(raw_dir)
        dataset = relative.parts[0] if len(relative.parts) > 1 else "."
        source = ""
        for key in ("external_dir", "main_dir", "enron_spam", "spamassassin", "nazario"):
            if key in roots and path.resolve().is_relative_to(roots[key]):
                source = "nazario" if key in ("external_dir", "main_dir") else key
                break
        item = Path(*relative.parts[1:]).as_posix() if dataset != "." else relative.as_posix()
        kind = "archive" if path.suffix.lower() in {".zip", ".tar", ".gz", ".bz2"} else "file"
        record = {"dataset": dataset, "item": item, "item_type": kind,
            "bytes": path.stat().st_size, "sha256": sha256_file(path), "retrieved_utc": now,
            "source_url": "", "source_licence": "", "source_note": "",
            "path": relative.as_posix()}
        record.update(metadata.get(source, {}))
        rows.append(record)
        file_info.append((dataset, record))
    # Retain the original per-dataset directory summaries for existing callers.
    for directory in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        members = [record for dataset, record in file_info if dataset == directory.name]
        digest_input = "".join(f"{row['item']}|{row['sha256']}\n" for row in members)
        source = "nazario" if directory.name == "nazario_external" else directory.name
        summary = {"dataset": directory.name, "item": ".", "item_type": "directory",
            "bytes": sum(row["bytes"] for row in members),
            "sha256": hashlib.sha256(digest_input.encode("utf-8")).hexdigest(),
            "retrieved_utc": now, "source_url": "", "source_licence": "",
            "source_note": "", "path": directory.name}
        summary.update(metadata.get(source, {}))
        rows.append(summary)
    rows.sort(key=lambda row: (row["dataset"], row["item"]))
    fields = ["dataset", "item", "item_type", "bytes", "sha256", "retrieved_utc",
              "source_url", "source_licence", "source_note", "path"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
