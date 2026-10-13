"""Shared overwrite protection for deterministic generated paper assets."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_generation(
    root: Path,
    outputs: Iterable[Path],
    *,
    refresh: bool,
    label: str,
) -> Path | None:
    """Refuse replacement by default; archive any existing files on refresh.

    The archive preserves each artifact at its repository-relative path and
    records its digest. Existing files are moved out of their active path so
    generators with their own collision checks can proceed. Partial
    generations are archived file by file so an interrupted previous run is
    also recoverable.
    """
    root = root.resolve()
    paths = list(dict.fromkeys(Path(path).resolve() for path in outputs))
    existing = [path for path in paths if path.exists()]
    if existing and not refresh:
        formatted = ", ".join(str(path.relative_to(root)) for path in existing)
        raise FileExistsError(
            f"generated output already exists; pass --refresh-generated to archive it first: {formatted}"
        )
    if not existing:
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = root / "results" / "jbi" / "_archive" / f"{label}_{stamp}"
    archive.mkdir(parents=True, exist_ok=False)
    records = []
    for path in existing:
        relative = path.relative_to(root)
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.move(str(path), str(target))
            files = sorted(p for p in target.rglob("*") if p.is_file())
            records.append({
                "path": relative.as_posix(),
                "archive_path": target.relative_to(archive).as_posix(),
                "type": "directory",
                "files": [{
                    "path": file.relative_to(target).as_posix(),
                    "size_bytes": file.stat().st_size,
                    "sha256": sha256(file),
                } for file in files],
            })
        else:
            shutil.move(str(path), str(target))
            records.append({
                "path": relative.as_posix(),
                "archive_path": target.relative_to(archive).as_posix(),
                "type": "file",
                "size_bytes": target.stat().st_size,
                "sha256": sha256(target),
            })
    (archive / "archive_manifest.json").write_text(
        json.dumps({
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "reason": "--refresh-generated requested",
            "label": label,
            "files": records,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return archive


def refresh_from_argv() -> bool:
    """Read the common flag for the small standalone figure renderers."""
    import sys

    args = sys.argv[1:]
    if any(arg in {"-h", "--help"} for arg in args):
        print("Usage: renderer [--refresh-generated]")
        raise SystemExit(0)
    unknown = [arg for arg in args if arg != "--refresh-generated"]
    if unknown:
        raise SystemExit(f"unsupported argument(s): {' '.join(unknown)}; use --refresh-generated to replace outputs")
    return "--refresh-generated" in args
