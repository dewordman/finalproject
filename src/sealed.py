# src/sealed.py
"""Persistent final-test reservations and append-only revision records."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@contextmanager
def final_test_session(locks_dir: Path):
    """Hold an OS mutex for the entire test run; the guard file is never deleted."""
    Path(locks_dir).mkdir(parents=True, exist_ok=True)
    guard = Path(locks_dir) / "final_test_guard.lock"
    with guard.open("a+b") as handle:
        if guard.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another final-test process holds the global guard") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_lock(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid lock document: {path}")
    return payload


def _replace_lock(path: Path, payload: Dict[str, Any]) -> None:
    """Replace atomically; never open an existing lock with a truncating mode."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_revision(path: Path, entry: Dict[str, Any]) -> None:
    """The .json revision journal is JSON Lines: one object per appended line.

    Keeping it append-only preserves every earlier byte. Existing malformed
    content is rejected rather than repaired, replaced or reset.
    """
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict) or "run_id" not in value:
                        raise ValueError(f"Invalid revision entry {line_number}: {path}")
    needs_separator = path.exists() and path.stat().st_size > 0
    if needs_separator:
        with path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            needs_separator = handle.read(1) != b"\n"
    with path.open("a", encoding="utf-8") as handle:
        if needs_separator:
            handle.write("\n")
        handle.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _adopt_legacy_lock(lock_path: Path, results_dir: Path, cfg_path: Path) -> None:
    """Preserve discovered old result-folder locks without moving/deleting them."""
    if lock_path.exists():
        return
    candidates = {results_dir / "test_lock.json"}
    candidates.update(cfg_path.parent.glob("results*/test_lock.json"))
    legacy = [
        {"path": str(path.resolve()), "payload": _read_lock(path)}
        for path in sorted(candidates) if path.is_file()
    ]
    if legacy:
        with lock_path.open("x", encoding="utf-8") as handle:
            json.dump({"status": "legacy", "timestamp": _now(),
                       "legacy_locks": legacy, "history": []}, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())


def enforce_test_lock(
    results_dir: Path, cfg_path: Path, force: bool, reason: Optional[str],
    *, locks_dir: Path, run_id: Optional[str] = None,
) -> str:
    results_dir, cfg_path, locks_dir = map(Path, (results_dir, cfg_path, locks_dir))
    if locks_dir.resolve().is_relative_to(results_dir.resolve()):
        raise ValueError("paths.locks_dir must be outside the results directory")
    if force and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError("--reason must contain non-blank text when using --force")
    locks_dir.mkdir(parents=True, exist_ok=True)
    lock_path = locks_dir / "final_test_lock.json"
    _adopt_legacy_lock(lock_path, results_dir, cfg_path)
    digest = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    old = _read_lock(lock_path) if lock_path.exists() else None
    if old is not None and not force:
        raise RuntimeError(
            f"Test lock already present at {lock_path}. "
            "Use --force --reason '<explanation>' only for an explicit revision."
        )
    run_id = run_id or uuid.uuid4().hex
    entry = {"timestamp": _now(), "run_id": run_id, "reason": reason,
             "config_digest": digest, "results_dir": str(results_dir.resolve())}
    if force:
        _append_revision(locks_dir / "revision_log.json", entry)
    history = [] if old is None else list(old.get("history", []))
    if old is not None:
        history.append({key: value for key, value in old.items() if key != "history"})
    payload = {**entry, "status": "in_progress", "history": history}
    if old is None:
        try:
            with lock_path.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise RuntimeError(f"A final-test lock was created concurrently: {lock_path}") from exc
    else:
        _replace_lock(lock_path, payload)
    return digest


def write_test_lock(
    locks_dir: Path, digest: str, summary: Dict[str, Any],
    *, run_id: Optional[str] = None, status: str = "completed",
) -> None:
    lock_path = Path(locks_dir) / "final_test_lock.json"
    payload = _read_lock(lock_path)
    if run_id is not None and payload.get("run_id") != run_id:
        raise RuntimeError("Refusing to update another run's final-test reservation")
    if payload.get("config_digest") != digest:
        raise RuntimeError("Final-test reservation digest changed")
    payload.update({"status": status, "completed_at": _now(),
                    "summary_keys": list(summary)})
    _replace_lock(lock_path, payload)


def reserve_experiment_lock(
    name: str, results_dir: Path, cfg_path: Path, force: bool, reason: Optional[str],
    *, locks_dir: Path, run_id: Optional[str] = None,
) -> str:
    """Reserve a once-only experiment (for example the external temporal test).

    The lock file is ``<locks_dir>/<name>_lock.json``. It is created exclusively;
    a second run needs ``force`` with a non-blank reason, which appends an entry
    (tagged with the experiment name) to the shared append-only revision log.
    Nothing here ever deletes or truncates a lock.
    """
    results_dir, cfg_path, locks_dir = map(Path, (results_dir, cfg_path, locks_dir))
    if not name.replace("_", "").isalnum():
        raise ValueError("Experiment lock name must be alphanumeric/underscore")
    if locks_dir.resolve().is_relative_to(results_dir.resolve()):
        raise ValueError("paths.locks_dir must be outside the results directory")
    if force and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError("--reason must contain non-blank text when using --force")
    locks_dir.mkdir(parents=True, exist_ok=True)
    lock_path = locks_dir / f"{name}_lock.json"
    digest = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    run_id = run_id or uuid.uuid4().hex
    entry = {"timestamp": _now(), "run_id": run_id, "reason": reason, "experiment": name,
             "config_digest": digest, "results_dir": str(results_dir.resolve())}
    old = _read_lock(lock_path) if lock_path.exists() else None
    if old is not None and not force:
        raise RuntimeError(
            f"The {name} experiment was already run (lock: {lock_path}). "
            "Use --force --reason '<explanation>' only for an explicit revision.")
    if force:
        _append_revision(locks_dir / "revision_log.json", entry)
    history = [] if old is None else list(old.get("history", [])) + [
        {key: value for key, value in old.items() if key != "history"}]
    payload = {**entry, "status": "in_progress", "history": history}
    if old is None:
        try:
            with lock_path.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise RuntimeError(f"A {name} lock was created concurrently: {lock_path}") from exc
    else:
        _replace_lock(lock_path, payload)
    return digest


def complete_experiment_lock(name: str, locks_dir: Path, digest: str, run_id: str,
                             *, status: str = "completed") -> None:
    """Mark a reserved experiment completed or failed (the lock stays in place)."""
    lock_path = Path(locks_dir) / f"{name}_lock.json"
    payload = _read_lock(lock_path)
    if payload.get("run_id") != run_id or payload.get("config_digest") != digest:
        raise RuntimeError("Refusing to update another run's experiment reservation")
    payload.update({"status": status, "completed_at": _now()})
    _replace_lock(lock_path, payload)
