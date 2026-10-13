"""Download and validate the official PhysioNet Challenge 2012 data.

The official Set-A input is downloaded by default.  Validation is intentionally redundant: archive integrity, exactly
4,000 numeric record files, matching RecordIDs, binary labels, and SHA-256
digests are all recorded in ``manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from .p12_parse import load_outcomes


BASE_URL = "https://physionet.org/files/challenge-2012/1.0.0"
EXPECTED_RECORDS_PER_SET = 4000


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> Dict[str, object]:
    """Download one URL atomically, resuming a partial file when supported."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "ICU-KAN/0.1 (research data downloader)"}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    last_error: Exception | None = None
    for attempt in range(3):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", None)
                accepts_range = status == 206 and existing > 0
                if not accepts_range and existing:
                    partial.unlink()
                    existing = 0
                mode = "ab" if accepts_range else "wb"
                with partial.open(mode) as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            break
        except Exception as exc:
            # PhysioNet's front end occasionally closes a long HTTPS stream
            # early.  Keep the partial file and retry with the same range.
            last_error = exc
            if attempt == 2:
                raise
            time.sleep(1.0)
    if last_error is not None and not partial.exists():
        raise last_error

    os.replace(partial, destination)
    digest = sha256_file(destination)
    return {
        "url": url,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": digest,
    }


def _safe_extract(tar_path: Path, output_dir: Path) -> None:
    """Extract a tarball without permitting path traversal."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            target = (output_dir / member.name).resolve()
            if output_dir.resolve() not in target.parents and target != output_dir.resolve():
                raise ValueError(f"unsafe path in archive: {member.name}")
        for member in archive.getmembers():
            archive.extract(member, path=output_dir)


def _record_ids_from_directory(directory: Path) -> List[int]:
    return sorted(int(path.stem) for path in directory.glob("*.txt") if path.stem.isdigit())


def validate_set(
    raw_dir: str | Path,
    set_name: str,
    outcomes_path: str | Path,
    expected_records: int = EXPECTED_RECORDS_PER_SET,
) -> Dict[str, object]:
    """Validate one extracted set against its official outcomes file."""

    raw_dir = Path(raw_dir)
    set_dir = raw_dir / f"set-{set_name}"
    record_ids = _record_ids_from_directory(set_dir)
    outcomes = load_outcomes(outcomes_path)
    if len(record_ids) != expected_records:
        raise ValueError(
            f"expected {expected_records} Set-{set_name.upper()} records, found {len(record_ids)}"
        )
    if set(record_ids) != set(outcomes):
        missing = sorted(set(record_ids) - set(outcomes))[:10]
        extra = sorted(set(outcomes) - set(record_ids))[:10]
        raise ValueError(
            f"Set-{set_name.upper()}/outcome RecordID mismatch; "
            f"missing={missing}, extra={extra}"
        )
    labels = list(outcomes.values())
    return {
        "record_count": len(record_ids),
        "record_id_min": min(record_ids),
        "record_id_max": max(record_ids),
        "outcome_count": len(outcomes),
        "death_count": int(sum(labels)),
        "survival_count": int(len(labels) - sum(labels)),
        "unknown_parameters_checked": False,
    }


def validate_set_a(raw_dir: str | Path, outcomes_path: str | Path) -> Dict[str, object]:
    """Backward-compatible Set-A validation entry point."""

    return validate_set(raw_dir, "a", outcomes_path)


def download_p12(
    raw_dir: str | Path,
    include_set_b: bool = False,
    include_set_c: bool = False,
    sets: Iterable[str] | None = None,
) -> Dict[str, object]:
    """Download, extract, validate, and manifest selected P12 sets.

    The historical default remains Set-A only for Phase 0 reproducibility.
    Phase 1 calls ``sets=("a", "b", "c")`` and validates every set before
    it can enter preprocessing.
    """

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    downloads: List[Dict[str, object]] = []

    if sets is None:
        selected = ["a"]
        if include_set_b:
            selected.append("b")
        if include_set_c:
            selected.append("c")
    else:
        selected = [str(item).lower().replace("set-", "") for item in sets]
    if not selected or any(item not in {"a", "b", "c"} for item in selected):
        raise ValueError(f"sets must be a non-empty subset of a/b/c, received {selected}")
    selected = list(dict.fromkeys(selected))

    validations: Dict[str, Dict[str, object]] = {}
    for set_name in selected:
        archive_name = f"set-{set_name}.tar.gz"
        archive_path = raw_dir / archive_name
        downloads.append(_download(f"{BASE_URL}/{archive_name}", archive_path))
        _safe_extract(archive_path, raw_dir)

        outcomes_name = f"Outcomes-{set_name}.txt"
        outcomes_path = raw_dir / outcomes_name
        downloads.append(_download(f"{BASE_URL}/{outcomes_name}", outcomes_path))
        validations[f"set_{set_name}"] = validate_set(raw_dir, set_name, outcomes_path)

    # Retain the Phase 0 flat fields while adding per-set validation for the
    # 12k Phase 1 manifest.  Phase 1 normally selects all three sets, but a
    # subset remains useful for a local download smoke test.
    base_validation = validations.get("set_a", next(iter(validations.values())))
    validation: Dict[str, object] = dict(base_validation)
    validation["sets"] = validations
    validation["total_record_count"] = int(sum(item["record_count"] for item in validations.values()))
    manifest: Dict[str, object] = {
        "dataset": "PhysioNet/Computing in Cardiology Challenge 2012",
        "version": "1.0.0",
        "source_base_url": BASE_URL,
        "license_note": "Open Data Commons Attribution License v1.0 (per official source page)",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "downloads": downloads,
        "validation": validation,
    }
    manifest_path = raw_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw", help="raw data directory")
    parser.add_argument(
        "--include-set-b",
        action="store_true",
        help="also download Set B and Outcomes-b",
    )
    parser.add_argument("--include-set-c", action="store_true", help="also download Set C and Outcomes-c")
    parser.add_argument(
        "--sets",
        help="comma-separated official sets to download, e.g. a,b,c (overrides include flags)",
    )
    args = parser.parse_args()
    selected = None if args.sets is None else [item.strip() for item in args.sets.split(",") if item.strip()]
    manifest = download_p12(
        args.out,
        include_set_b=args.include_set_b,
        include_set_c=args.include_set_c,
        sets=selected,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
