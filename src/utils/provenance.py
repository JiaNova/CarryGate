"""Small, explicit provenance records for experiment result rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


PROVENANCE_FIELDS = (
    "config_sha256",
    "code_sha256",
    "src_sha256",
    "torch",
    "cuda",
    "device_name",
    "device_index",
    "device_count",
    "batch_size",
    "effective_batch",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _digest_tree(project_root: Path, directories: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    source_files = sorted(
        path
        for name in directories
        if (project_root / name).is_dir()
        for path in (project_root / name).rglob("*.py")
    )
    for path in source_files:
        digest.update(path.relative_to(project_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _code_digest(project_root: Path) -> str:
    return _digest_tree(project_root, ("src", "scripts"))


def _src_digest(project_root: Path) -> str:
    """Digest of the library only, excluding orchestration and reporting.

    ``code_sha256`` covers ``scripts/`` too, so adding a plotting or reporting
    script changes it even though no model, training loop, or metric moved.
    That granularity cannot answer the question a bitwise regression actually
    asks -- "could the numbers have changed?" -- and it made the MIMIC-IV L-M1
    gate ambiguous.  ``src_sha256`` is the field to compare when asserting that
    two runs should agree bit for bit.
    """

    return _digest_tree(project_root, ("src",))


def provenance_fingerprint(
    cfg: Mapping[str, Any],
    *,
    project_root: str | Path,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Return the minimum config/code/environment fingerprint required by §8.1."""

    root = Path(project_root).resolve()
    config_payload = json.dumps(
        _jsonable(dict(cfg)), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    resolved = torch.device(device) if device is not None else torch.device("cpu")
    if resolved.type == "cuda" and torch.cuda.is_available():
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(index)
        device_index = int(index)
    else:
        device_name = "cpu"
        device_index = -1
    device_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    batch_size = int(cfg.get("batch_size", -1))
    effective_batch = batch_size * (device_count if cfg.get("multi_gpu") else 1)
    return {
        "config_sha256": hashlib.sha256(config_payload).hexdigest()[:16],
        "code_sha256": _code_digest(root),
        "src_sha256": _src_digest(root),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "device_name": str(device_name),
        # Two identical A100s still record the same device_name, so a
        # bitwise-regression failure could not be attributed to the card
        # without the index.  Sharded matrices split across both.
        "device_index": device_index,
        "device_count": device_count,
        "batch_size": batch_size,
        "effective_batch": int(effective_batch),
    }


def provenance_columns(result: Mapping[str, Any]) -> dict[str, Any]:
    """Select flat provenance fields for a CSV result row."""

    return {field: result[field] for field in PROVENANCE_FIELDS if field in result}


def write_config_snapshot(output_dir: str | Path, cfg: Mapping[str, Any]) -> Path:
    """Persist the small resolved base configuration beside a result matrix."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config_snapshot.json"
    path.write_text(
        json.dumps(_jsonable(dict(cfg)), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path
