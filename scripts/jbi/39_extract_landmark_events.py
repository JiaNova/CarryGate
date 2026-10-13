#!/usr/bin/env python
"""Extract and audit the frozen P1 192-hour event streams.

The script reuses only the stay IDs, patient-level splits, and standardizers in
the committed JBI caches.  It writes the new full event stream only after the
all-stay identity, split, and 0--48-hour fidelity audits pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import eicu_extract as eicu  # noqa: E402
from src.data import mimiciv_extract as mimiciv  # noqa: E402
from src.data.p12_events import EPS, EventStandardizer, TIME_SERIES_VARIABLES  # noqa: E402


PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "jbi"
PLAN = ROOT / "paper" / "预设用例分析计划_P1.md"
PLAN_VERSION = "p1_landmark_v1"
HORIZON_HOURS = 192.0
LEGACY_HORIZON_HOURS = 48.0
CHUNK_SIZE = 250

FROZEN = {
    "eicu": {
        "cache": "eicu_events_jbi_v1.pkl",
        "split": "eicu_split_indices_jbi_v1.json",
        "cache_sha256": "77bf221e2d374fbbd40e69b3dcb4ace4d0cbcd1cff8086f360e571991585d07a",
        "split_sha256": "ccb72fe14d8c245e16b0dac3ffaad3d350c717bfbcf8d1ff864b3b7bc79686e4",
        "standardizer_sha256": "30e828587979073a2d4ca313311ae7f648564798ff1b18653f6ec0af82ccc8b1",
        "map": "configs/eicu_variable_map.yaml",
        "map_sha256": "7ce2a095250826cb8d392edca983bcaf6aaba1b5bd69bce8171adfb205ff4ec6",
        "group_column": "uniquepid",
        "split_sizes": {"train": 8397, "val": 1794, "test": 1809},
        "trackb_inputs": {
            "results/jbi/trackb_controls/run_manifest.json": "4fda82898d6235922c42490e62d0bd3b1c12766e1fba158d5b051c1768b53a3d",
            "results/jbi/trackb_controls/search_results.csv": "ab5d7bfa5d0a8aa740a76b96ef1e3df6a98f49a2108620f556d0aab6768bda3c",
        },
    },
    "mimiciv": {
        "cache": "mimiciv_events_jbi_v1.pkl",
        "split": "mimiciv_split_indices_jbi_v1.json",
        "cache_sha256": "f24072c22274b6473bcc8bdb817b413f7dcbcd7a90ae74f6791ea9bafcbef725",
        "split_sha256": "990a308705f80d784da9ee39a9d1bb267f4130e2bfbf498495c9dc6133a909ec",
        "standardizer_sha256": "0b16ea6738c98642aaf4d1f53664ac4b94fbe57235f1cdc6e0c609fe8094148d",
        "map": "configs/mimiciv_variable_map.yaml",
        "map_sha256": "d00805ae7cfd90cb8a33d09bb72d84a8162b9bf9107c3fe251fedbcffd8fec1b",
        "group_column": "subject_id",
        "split_sizes": {"train": 8403, "val": 1810, "test": 1787},
        "trackb_inputs": {
            "results/jbi/trackb_controls_mimiciv/run_manifest.json": "368d7a33bd78119fc3c94c93b682b9cd67e7c908ffe230f54ddf8b763c5eaf55",
            "results/jbi/trackb_controls_mimiciv/search_results.csv": "7030cc51c6a63e6833349fe7d52a09d913bbd5a169df2652c6dca90a624a0a13",
        },
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (Path,)):
        return str(value)
    raise TypeError(type(value).__name__)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_status() -> str:
    return subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout


def _frozen_inputs(cohort: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    frozen = FROZEN[cohort]
    cache_path = PROCESSED / frozen["cache"]
    split_path = PROCESSED / frozen["split"]
    map_path = ROOT / frozen["map"]
    expected_hashes = {
        str(cache_path.relative_to(ROOT)): frozen["cache_sha256"],
        str(split_path.relative_to(ROOT)): frozen["split_sha256"],
        str(map_path.relative_to(ROOT)): frozen["map_sha256"],
    }
    for relative, digest in frozen["trackb_inputs"].items():
        expected_hashes[relative] = digest
    actual_hashes: dict[str, str] = {}
    for relative, expected in expected_hashes.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"frozen P1 input is missing: {path}")
        actual = _sha256_file(path)
        actual_hashes[relative] = actual
        if actual != expected:
            raise RuntimeError(
                f"frozen P1 input changed: {relative}; expected {expected}, found {actual}"
            )

    with cache_path.open("rb") as handle:
        cache = pickle.load(handle)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    standardizer_payload = cache.get("standardizer")
    if not isinstance(standardizer_payload, Mapping):
        raise TypeError(f"{cache_path} standardizer is not a mapping")
    standardizer_hash = _sha256_json(standardizer_payload)
    if standardizer_hash != frozen["standardizer_sha256"]:
        raise RuntimeError(
            f"{cohort} standardizer changed: expected {frozen['standardizer_sha256']}, "
            f"found {standardizer_hash}"
        )
    if list(cache.get("variable_names", ())) != list(TIME_SERIES_VARIABLES):
        raise RuntimeError(f"{cohort} variable order does not match P12's frozen 37 variables")
    if float(cache.get("t_obs", -1)) != 24.0:
        raise RuntimeError(f"{cohort} source cache is not the frozen 24-hour task")
    examples = list(cache.get("examples", ()))
    if len(examples) != 12_000:
        raise RuntimeError(f"{cohort} expected 12,000 source stays, found {len(examples)}")
    ids = [int(example["record_id"]) for example in examples]
    sample_ids = [str(example["sample_id"]) for example in examples]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{cohort} source cache contains duplicate stay IDs")
    if split.get("sample_ids") != sample_ids:
        raise RuntimeError(f"{cohort} split sample order differs from the frozen cache")
    if split.get("group_column") != frozen["group_column"]:
        raise RuntimeError(f"{cohort} split group column changed")
    if split.get("split_unit") != "patient_grouped":
        raise RuntimeError(f"{cohort} P1 requires the frozen patient-grouped split")
    split_sizes = {key: len(split[key]) for key in ("train", "val", "test")}
    if split_sizes != frozen["split_sizes"]:
        raise RuntimeError(f"{cohort} split sizes changed: {split_sizes}")
    all_indices = [int(index) for name in ("train", "val", "test") for index in split[name]]
    if sorted(all_indices) != list(range(len(examples))):
        raise RuntimeError(f"{cohort} split indices are not an exact partition")
    group_column = frozen["group_column"]
    groups = {
        name: {str(examples[int(index)][group_column]) for index in split[name]}
        for name in ("train", "val", "test")
    }
    overlaps = {
        f"{left}&{right}": len(groups[left] & groups[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    if any(overlaps.values()):
        raise RuntimeError(f"{cohort} frozen patient split leaks groups: {overlaps}")
    return cache, split, standardizer_payload, actual_hashes


def _validate_identity(cohort: str, source: Mapping[str, Any], stay: Any) -> list[str]:
    errors: list[str] = []
    checks = {
        "record_id": int(source["record_id"]) == int(stay.stay_id),
        "label": int(source["label"]) == int(stay.label),
        "subject_id": int(source["subject_id"]) == int(stay.subject_id),
        "hadm_id": int(source["hadm_id"]) == int(stay.hadm_id),
    }
    if cohort == "eicu":
        checks.update(
            {
                "uniquepid": str(source["uniquepid"]) == str(stay.uniquepid),
                "hospital_id": int(source["hospital_id"]) == int(stay.hospital_id),
                "gender": str(source["gender"]) == str(stay.gender),
                "anchor_age": int(source["anchor_age"]) == int(stay.anchor_age),
            }
        )
    else:
        checks["icu_type"] = int(source["icu_type"]) == int(stay.icu_type)
    for field, passed in checks.items():
        if not passed:
            errors.append(
                f"{cohort} stay {source['record_id']} identity mismatch in {field}: "
                f"source={source.get(field)!r}, database={getattr(stay, field, None)!r}"
            )
    return errors


def _outtime_hours(cohort: str, stay: Any) -> float:
    if cohort == "eicu":
        return float(stay.outtime)
    return float((stay.icu_outtime - stay.icu_intime).total_seconds() / 3600.0)


def _open_database(cohort: str, database_url: str):
    module = eicu if cohort == "eicu" else mimiciv
    return module.DatabaseClient(database_url)


def _fetch_selected_stays(
    cohort: str,
    database_url: str,
    mapping: Mapping[str, Any],
    source_examples: Sequence[Mapping[str, Any]],
) -> tuple[list[Any], dict[str, Any]]:
    module = eicu if cohort == "eicu" else mimiciv
    client = _open_database(cohort, database_url)
    try:
        schema = module.schema_check(client, mapping)
        if not schema.get("verified"):
            raise RuntimeError(f"{cohort} schema check failed: {schema.get('missing')}")
        if cohort == "eicu":
            queue, queue_metadata, _ = module._fetch_cohort(client, mapping)
            queue_description = queue_metadata
        else:
            queue, queue_counts, careunit_map = module._fetch_queue(client)
            queue_description = {"counts": queue_counts, "icu_type_mapping": careunit_map}
        by_id = {int(stay.stay_id): stay for stay in queue}
        if len(by_id) != len(queue):
            raise RuntimeError(f"{cohort} live eligible queue contains duplicate stay IDs")
        ids = [int(example["record_id"]) for example in source_examples]
        missing = [stay_id for stay_id in ids if stay_id not in by_id]
        if missing:
            raise RuntimeError(
                f"{cohort} frozen source stays missing from the live eligible queue: "
                f"count={len(missing)}, first={missing[:10]}"
            )
        selected = [by_id[stay_id] for stay_id in ids]
        identity_errors: list[str] = []
        for source, stay in zip(source_examples, selected):
            identity_errors.extend(_validate_identity(cohort, source, stay))
        if identity_errors:
            raise RuntimeError("\n".join(identity_errors[:20]))
        queue_description["selected_stays"] = len(selected)
        queue_description["eligible_queue_stays"] = len(queue)
        return selected, {"schema_check": schema, "queue": queue_description}
    finally:
        client.close()


def _extract_chunks(
    cohort: str,
    database_url: str,
    mapping: Mapping[str, Any],
    selected: Sequence[Any],
    *,
    map_sha256: str,
    chunk_size: int,
) -> tuple[list[dict[str, Any]], str, Path]:
    module = eicu if cohort == "eicu" else mimiciv
    work_root = PROCESSED / ".p1_landmark_extract_work" / cohort
    work_root.mkdir(parents=True, exist_ok=True)
    # Chunk contents depend on the cohort extractor and frozen extraction
    # parameters. Packaging/audit-only edits to this orchestrator do not change
    # those raw event rows, so completed chunks remain safely resumable.
    extraction_code_hashes = {"extractor": _sha256_file(Path(module.__file__).resolve())}
    signature = _sha256_json(
        {
            "cohort": cohort,
            "selected_stay_ids": [int(stay.stay_id) for stay in selected],
            "mapping_sha256": map_sha256,
            "mapping_fingerprint": module._map_fingerprint(mapping),
            "horizon_hours": HORIZON_HOURS,
            "respect_icu_outtime": True,
            "chunk_size": int(chunk_size),
            "code_hashes": extraction_code_hashes,
        }
    )
    state_path = work_root / "state.json"
    state_payload = {
        "signature": signature,
        "cohort": cohort,
        "selected_stay_ids_sha256": _sha256_json(
            [int(stay.stay_id) for stay in selected]
        ),
        "mapping_sha256": map_sha256,
        "mapping_fingerprint": module._map_fingerprint(mapping),
        "horizon_hours": HORIZON_HOURS,
        "respect_icu_outtime": True,
        "chunk_size": int(chunk_size),
        "code_hashes": extraction_code_hashes,
    }
    if state_path.exists():
        old_state = json.loads(state_path.read_text(encoding="utf-8"))
        if old_state.get("signature") != signature:
            old_code_hashes = old_state.get("code_hashes", {})
            old_signature = _sha256_json(
                {
                    "cohort": cohort,
                    "selected_stay_ids": [int(stay.stay_id) for stay in selected],
                    "mapping_sha256": map_sha256,
                    "mapping_fingerprint": module._map_fingerprint(mapping),
                    "horizon_hours": HORIZON_HOURS,
                    "respect_icu_outtime": True,
                    "chunk_size": int(chunk_size),
                    "code_hashes": old_code_hashes,
                }
            )
            static_match = all(
                old_state.get(key) == state_payload[key]
                for key in (
                    "cohort",
                    "selected_stay_ids_sha256",
                    "mapping_sha256",
                    "mapping_fingerprint",
                    "horizon_hours",
                    "respect_icu_outtime",
                    "chunk_size",
                )
            )
            if (
                old_signature != old_state.get("signature")
                or old_code_hashes.get("extractor") != extraction_code_hashes["extractor"]
                or not static_match
            ):
                raise RuntimeError(
                    f"existing P1 work directory belongs to a different extraction: {work_root}"
                )
            print(
                f"{cohort}: resuming completed raw event chunks; only packaging/audit code changed",
                flush=True,
            )
            _atomic_json(state_path, state_payload)
    else:
        _atomic_json(state_path, state_payload)

    records: list[dict[str, Any]] = []
    sql_hashes: list[str] = []
    n_chunks = (len(selected) + int(chunk_size) - 1) // int(chunk_size)
    client = _open_database(cohort, database_url)
    try:
        for chunk_index in range(n_chunks):
            start = chunk_index * int(chunk_size)
            chunk = selected[start : start + int(chunk_size)]
            expected_ids = [int(stay.stay_id) for stay in chunk]
            part_path = work_root / f"part-{chunk_index:05d}.pkl"
            if part_path.exists():
                with part_path.open("rb") as handle:
                    payload = pickle.load(handle)
                part_records = payload.get("records")
                query_sha256 = str(payload.get("query_sha256", ""))
                actual_ids = [int(record["record_id"]) for record in part_records or ()]
                if actual_ids != expected_ids:
                    raise RuntimeError(f"resumed extraction part does not match {part_path}")
                if not query_sha256:
                    raise RuntimeError(f"resumed extraction part has no SQL hash: {part_path}")
            else:
                if cohort == "eicu":
                    part_records, _quality, query_sha256 = module._extract_chunk(
                        client,
                        chunk,
                        mapping,
                        horizon_hours=HORIZON_HOURS,
                        respect_outtime=True,
                    )
                else:
                    part_records, _seen, _hits, sql_fragments = module._extract_chunk(
                        client,
                        chunk,
                        mapping,
                        horizon_hours=HORIZON_HOURS,
                        respect_outtime=True,
                    )
                    query_sha256 = hashlib.sha256(
                        "\n".join(sql_fragments).encode("utf-8")
                    ).hexdigest()
                payload = {"records": part_records, "query_sha256": query_sha256}
                _atomic_pickle(part_path, payload)
            if [int(record["record_id"]) for record in part_records] != expected_ids:
                raise RuntimeError(f"extractor changed stay order in {cohort} chunk {chunk_index}")
            records.extend(part_records)
            sql_hashes.append(query_sha256)
            _atomic_json(
                work_root / "progress.json",
                {"completed_chunks": chunk_index + 1, "total_chunks": n_chunks},
            )
            event_count = sum(len(record.get("events", ())) for record in part_records)
            print(
                f"{cohort}: chunk {chunk_index + 1}/{n_chunks}; "
                f"stays={len(chunk)}; events={event_count}",
                flush=True,
            )
    finally:
        client.close()
    expected_all_ids = [int(stay.stay_id) for stay in selected]
    actual_all_ids = [int(record["record_id"]) for record in records]
    if actual_all_ids != expected_all_ids:
        raise RuntimeError(f"{cohort} extraction lost, duplicated, or reordered stays")
    query_hash = hashlib.sha256("\n".join(sql_hashes).encode("utf-8")).hexdigest()
    return records, query_hash, work_root


def _recheck_extraction_bounds(
    cohort: str,
    records: Sequence[dict[str, Any]],
    selected_stays: Sequence[Any],
) -> dict[str, int]:
    """Apply half-open extraction bounds in the original relative-hour precision.

    eICU stores discharge offsets in minutes but the event helper checks hours
    after converting minutes to hours and back. This pass prevents a measurement
    at D from entering the P1 event stream.
    """
    counts = {
        "events_removed_at_or_after_outtime": 0,
        "events_at_or_after_192h": 0,
        "negative_time_events": 0,
    }
    if len(records) != len(selected_stays):
        raise RuntimeError(f"{cohort} records and selected stays differ in length")
    for record, stay in zip(records, selected_stays):
        outtime_hour = _outtime_hours(cohort, stay)
        kept: list[tuple[str, float, float]] = []
        stay_counts = {key: 0 for key in counts}
        for event in record.get("events", ()):
            name, time_hour, value = str(event[0]), float(event[1]), float(event[2])
            if time_hour < 0.0:
                stay_counts["negative_time_events"] += 1
            elif time_hour >= outtime_hour:
                stay_counts["events_removed_at_or_after_outtime"] += 1
            elif time_hour >= HORIZON_HOURS:
                stay_counts["events_at_or_after_192h"] += 1
            else:
                kept.append((name, time_hour, value))
        record["events"] = kept
        record.pop("events_with_delta", None)
        record["_p1_boundary_recheck"] = stay_counts
        for key, value in stay_counts.items():
            counts[key] += int(value)
    return counts


def _make_full_event_arrays(
    record: Mapping[str, Any],
    standardizer: EventStandardizer,
    full_example: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    raw_events = list(record.get("events", ()))
    observed = full_example["obs"]
    future = full_example["target"]
    arrays = {
        key: np.concatenate((np.asarray(observed[key]), np.asarray(future[key])))
        for key in ("var", "time", "value", "delta_t", "is_first")
    }
    raw_times = np.asarray([float(event[1]) for event in raw_events], dtype=np.float64)
    raw_values = np.asarray([float(event[2]) for event in raw_events], dtype=np.float64)
    raw_names = [str(event[0]) for event in raw_events]
    name_to_id = {name: index for index, name in enumerate(standardizer.variable_names)}
    raw_var = np.asarray([name_to_id[name] for name in raw_names], dtype=np.int64)
    if not np.array_equal(raw_var, arrays["var"]):
        raise AssertionError("full event variable order differs from make_event_example")
    if not np.array_equal(raw_times.astype(np.float32), arrays["time"]):
        raise AssertionError("full event time order differs from make_event_example")
    raw32 = raw_values.astype(np.float32)
    arrays["value"] = _frozen_cache_transform(standardizer, raw_var, raw32)
    arrays["time_raw"] = raw_times
    arrays["value_raw"] = raw_values
    arrays["event_order"] = np.arange(len(raw_events), dtype=np.int64)
    return arrays


def _frozen_cache_transform(
    standardizer: EventStandardizer,
    variable_ids: np.ndarray,
    raw_values: np.ndarray,
) -> np.ndarray:
    """Match the exact float32 cache values produced by refit_standardizer.py."""
    variable_ids = np.asarray(variable_ids, dtype=np.int64)
    # The frozen JBI caches first store raw values as float32, then promote them
    # to float64 for standardization and cast only after clipping. This order is
    # material at the last bit and must be preserved for P1 cache fidelity.
    raw64 = np.asarray(raw_values, dtype=np.float32).astype(np.float64)
    standardized = (raw64 - standardizer.median[variable_ids]) / (
        standardizer.iqr[variable_ids] + EPS
    )
    return np.clip(
        standardized,
        standardizer.lower_clip[variable_ids],
        standardizer.upper_clip[variable_ids],
    ).astype(np.float32)


def _event_multiset(events: Mapping[str, Any], selector: np.ndarray) -> list[tuple[int, int, int]]:
    variables = np.asarray(events["var"], dtype=np.int64)[selector]
    times = np.asarray(events["time"], dtype=np.float32)[selector]
    values = np.asarray(events["value"], dtype=np.float32)[selector]
    time_bits = times.view(np.uint32)
    value_bits = values.view(np.uint32)
    return sorted(
        (int(variable), int(time_bit), int(value_bit))
        for variable, time_bit, value_bit in zip(variables, time_bits, value_bits)
    )


def _array_group_exact(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    fields: Sequence[str],
) -> tuple[bool, dict[str, Any]]:
    report: dict[str, Any] = {}
    passed = True
    for field in fields:
        left = np.asarray(expected[field])
        right = np.asarray(actual[field])
        same = left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
        report[field] = {
            "pass": bool(same),
            "expected_n": int(left.size),
            "actual_n": int(right.size),
            "expected_dtype": str(left.dtype),
            "actual_dtype": str(right.dtype),
        }
        passed &= bool(same)
    return passed, report


def _audit_one_stay(
    cohort: str,
    source: Mapping[str, Any],
    new: Mapping[str, Any],
    outtime_hour: float,
) -> dict[str, Any]:
    errors: list[str] = []
    source_id = int(source["record_id"])
    events = new["events"]
    raw_time = np.asarray(events["time_raw"], dtype=np.float64)
    horizon_end = min(float(outtime_hour), HORIZON_HOURS)
    if np.any(raw_time < 0.0) or np.any(raw_time >= horizon_end):
        errors.append(f"{cohort} stay {source_id}: event outside [0,min(D,192))")
    if np.any(np.diff(raw_time) < 0.0):
        errors.append(f"{cohort} stay {source_id}: full event stream is not chronologically ordered")
    event_order = np.asarray(events["event_order"], dtype=np.int64)
    if not np.array_equal(event_order, np.arange(len(raw_time), dtype=np.int64)):
        errors.append(f"{cohort} stay {source_id}: stable event order is not contiguous")

    obs_selector = raw_time < 24.0
    tgt_selector = (raw_time >= 24.0) & (raw_time < min(48.0, float(outtime_hour)))
    old_obs = source["obs"]
    old_tgt = source["target"]
    old_obs_selector = np.ones(len(np.asarray(old_obs["var"])), dtype=bool)
    old_tgt_time = np.asarray(old_tgt["time"], dtype=np.float32).astype(np.float64)
    old_tgt_selector = (old_tgt_time >= 24.0) & (
        old_tgt_time < min(48.0, float(outtime_hour))
    )
    obs_actual = _event_multiset(events, obs_selector)
    obs_expected = _event_multiset(old_obs, old_obs_selector)
    tgt_actual = _event_multiset(events, tgt_selector)
    tgt_expected = _event_multiset(old_tgt, old_tgt_selector)
    obs_pass = obs_actual == obs_expected
    tgt_pass = tgt_actual == tgt_expected
    if not obs_pass:
        errors.append(
            f"{cohort} stay {source_id}: [0,24) event multiset differs "
            f"(new={len(obs_actual)}, old={len(obs_expected)})"
        )
    if not tgt_pass:
        errors.append(
            f"{cohort} stay {source_id}: [24,min(48,D)) event multiset differs "
            f"(new={len(tgt_actual)}, old={len(tgt_expected)})"
        )

    excluded_target = int(
        np.sum((old_tgt_time >= float(outtime_hour)) & (old_tgt_time < 48.0))
    ) if outtime_hour < 48.0 else 0
    if float(outtime_hour) >= 48.0:
        old_full_tgt = _event_multiset(
            old_tgt,
            (old_tgt_time >= 24.0) & (old_tgt_time < 48.0),
        )
        complete_48_pass = tgt_actual == old_full_tgt
    else:
        complete_48_pass = None

    # Compare the compatibility arrays as well as the required event multisets.
    legacy_fields = ("var", "time", "value", "value_raw")
    legacy_new = {key: np.asarray(new["target_48"][key]) for key in legacy_fields}
    old_target_shared = {
        key: np.asarray(old_tgt[key])[old_tgt_selector]
        for key in legacy_fields
    }
    if outtime_hour >= 48.0:
        target_exact, target_exact_fields = _array_group_exact(
            {key: old_tgt[key] for key in legacy_fields},
            {key: new["target_48"][key] for key in legacy_fields},
            legacy_fields,
        )
    else:
        target_exact, target_exact_fields = _array_group_exact(
            old_target_shared, legacy_new, legacy_fields
        )
    old_obs_all = {key: old_obs[key] for key in legacy_fields}
    new_obs_all = {key: new["obs"][key] for key in legacy_fields}
    obs_exact, obs_exact_fields = _array_group_exact(old_obs_all, new_obs_all, legacy_fields)
    if not obs_exact:
        errors.append(f"{cohort} stay {source_id}: obs arrays are not bitwise identical")
    if not target_exact:
        errors.append(f"{cohort} stay {source_id}: shared target arrays are not bitwise identical")

    return {
        "record_id": source_id,
        "outtime_hour": float(outtime_hour),
        "n_events_retained_in_0_192": int(len(raw_time)),
        "boundary_recheck": dict(new.get("boundary_recheck", {})),
        "obs_0_24_multiset_pass": bool(obs_pass),
        "obs_arrays_pass": bool(obs_exact),
        "obs_event_count": int(len(obs_expected)),
        "obs_field_checks": obs_exact_fields,
        "target_24_48_shared_multiset_pass": bool(tgt_pass),
        "target_shared_arrays_pass": bool(target_exact),
        "target_shared_event_count": int(len(tgt_expected)),
        "target_field_checks": target_exact_fields,
        "target_full_24_48_pass_when_D_ge_48": complete_48_pass,
        "old_target_events_excluded_at_or_after_D_before_48": excluded_target,
        "errors": errors,
    }


def _build_candidate(
    cohort: str,
    source_examples: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    selected_stays: Sequence[Any],
    standardizer_payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    standardizer = EventStandardizer.from_dict(standardizer_payload)
    output: list[dict[str, Any]] = []
    stay_audits: list[dict[str, Any]] = []
    d_lt_48_stays = 0
    excluded_old_target_events = 0
    total_events = 0
    boundary_recheck = {
        "events_removed_at_or_after_outtime": 0,
        "events_at_or_after_192h": 0,
        "negative_time_events": 0,
    }
    per_variable: dict[str, int] = {name: 0 for name in TIME_SERIES_VARIABLES}
    for source, record, stay in zip(source_examples, records, selected_stays):
        if int(source["record_id"]) != int(record["record_id"]):
            raise RuntimeError(f"{cohort} source and extracted record order differ")
        outtime_hour = _outtime_hours(cohort, stay)
        if not np.isfinite(outtime_hour) or outtime_hour <= 0:
            raise RuntimeError(f"{cohort} invalid ICU outtime for stay {stay.stay_id}")
        maker = (eicu if cohort == "eicu" else mimiciv).make_event_example
        legacy = maker(
            record,
            standardizer,
            t_obs=24.0,
            horizon=LEGACY_HORIZON_HOURS,
        )
        for part in ("obs", "target"):
            block = legacy[part]
            block["value"] = _frozen_cache_transform(
                standardizer,
                np.asarray(block["var"], dtype=np.int64),
                np.asarray(block["value_raw"], dtype=np.float32),
            )
        full = maker(
            record,
            standardizer,
            t_obs=24.0,
            horizon=HORIZON_HOURS,
        )
        example = dict(legacy)
        for field, value in source.items():
            if field not in {"obs", "target"}:
                example[field] = value
        example["obs"] = legacy["obs"]
        example["target"] = legacy["target"]
        example["events"] = _make_full_event_arrays(record, standardizer, full)
        example["outtime_hour"] = float(outtime_hour)
        example["full_event_range"] = [0.0, min(float(outtime_hour), HORIZON_HOURS)]
        stay_boundary = dict(record.get("_p1_boundary_recheck", {}))
        for key in boundary_recheck:
            boundary_recheck[key] += int(stay_boundary.get(key, 0))
        audit = _audit_one_stay(
            cohort,
            source,
            example | {"target_48": legacy["target"], "boundary_recheck": stay_boundary},
            outtime_hour,
        )
        stay_audits.append(audit)
        if outtime_hour < 48.0:
            d_lt_48_stays += 1
        excluded_old_target_events += int(audit["old_target_events_excluded_at_or_after_D_before_48"])
        total_events += len(example["events"]["var"])
        for variable_id in np.asarray(example["events"]["var"], dtype=np.int64):
            per_variable[TIME_SERIES_VARIABLES[int(variable_id)]] += 1
        output.append(example)

    error_rows = [row for row in stay_audits if row["errors"]]
    report = {
        "cohort": cohort,
        "n_source_stays": len(source_examples),
        "n_extracted_stays": len(records),
        "stay_order_matches": [int(row["record_id"]) for row in source_examples]
        == [int(record["record_id"]) for record in records],
        "d_lt_48_stays": int(d_lt_48_stays),
        "old_target_events_excluded_at_or_after_D_before_48": int(excluded_old_target_events),
        "boundary_recheck": boundary_recheck,
        "n_events_in_new_full_stream": int(total_events),
        "events_by_variable": per_variable,
        "n_stays_obs_multiset_pass": int(sum(row["obs_0_24_multiset_pass"] for row in stay_audits)),
        "n_stays_target_shared_multiset_pass": int(
            sum(row["target_24_48_shared_multiset_pass"] for row in stay_audits)
        ),
        "n_stays_obs_array_pass": int(sum(row["obs_arrays_pass"] for row in stay_audits)),
        "n_stays_target_shared_array_pass": int(
            sum(row["target_shared_arrays_pass"] for row in stay_audits)
        ),
        "n_stays_full_24_48_pass_when_D_ge_48": int(
            sum(row["target_full_24_48_pass_when_D_ge_48"] is True for row in stay_audits)
        ),
        "n_errors": int(sum(len(row["errors"]) for row in stay_audits)),
        "first_errors": [
            {"record_id": row["record_id"], "errors": row["errors"]}
            for row in error_rows[:20]
        ],
        "stay_details": stay_audits,
    }
    report["pass"] = (
        report["stay_order_matches"]
        and report["n_source_stays"] == 12_000
        and report["n_extracted_stays"] == 12_000
        and report["n_stays_obs_multiset_pass"] == 12_000
        and report["n_stays_target_shared_multiset_pass"] == 12_000
        and report["n_stays_obs_array_pass"] == 12_000
        and report["n_stays_target_shared_array_pass"] == 12_000
        and report["boundary_recheck"]["negative_time_events"] == 0
        and report["boundary_recheck"]["events_at_or_after_192h"] == 0
        and report["n_errors"] == 0
    )
    return output, report


def _split_audit(cohort: str, cache: Mapping[str, Any], split: Mapping[str, Any]) -> dict[str, Any]:
    examples = list(cache["examples"])
    group_column = FROZEN[cohort]["group_column"]
    split_groups = {
        name: {str(examples[int(index)][group_column]) for index in split[name]}
        for name in ("train", "val", "test")
    }
    overlaps = {
        f"{left}&{right}": len(split_groups[left] & split_groups[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    sizes = {name: len(split[name]) for name in ("train", "val", "test")}
    passed = (
        sizes == FROZEN[cohort]["split_sizes"]
        and all(value == 0 for value in overlaps.values())
        and len({int(example["record_id"]) for example in examples}) == 12_000
    )
    return {
        "split_unit": split.get("split_unit"),
        "group_column": group_column,
        "split_sizes": sizes,
        "n_patients": len(set().union(*split_groups.values())),
        "group_overlap": overlaps,
        "sample_order_matches_split": split.get("sample_ids")
        == [str(example["sample_id"]) for example in examples],
        "pass": bool(passed),
    }


def _db_name(cohort: str) -> str:
    return "eicu" if cohort == "eicu" else "mimiciv"


def _query_smoke() -> dict[str, Any]:
    """Check that dynamic horizon placeholders stay aligned with query params."""
    from datetime import datetime, timezone

    e_map = eicu._load_map(ROOT / FROZEN["eicu"]["map"])
    m_map = mimiciv._load_map(ROOT / FROZEN["mimiciv"]["map"])
    e_stay = eicu.Stay(1, 1, 1, 1, "unit-test-patient", "F", 50, 0, 240.0, None)
    m_stay = mimiciv.Stay(
        1,
        1,
        1,
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 1, 11, tzinfo=timezone.utc),
        0,
        "MICU",
        1,
    )
    checks: list[dict[str, Any]] = []
    for cohort, module, mapping, stay in (
        ("eicu", eicu, e_map, e_stay),
        ("mimiciv", mimiciv, m_map, m_stay),
    ):
        variables = mapping["variables"]
        specs = list(variables.items())
        numeric = next((item for item in specs if item[1]["source_kind"] == "numeric"), None)
        if numeric is not None:
            query, params = module._numeric_query([stay], [numeric], horizon_hours=HORIZON_HOURS)[:2]
            expected_horizon = HORIZON_HOURS * 60.0 if cohort == "eicu" else HORIZON_HOURS
            checks.append(_placeholder_check(cohort, "numeric", query, params, expected_horizon))
        lab_kind = "lab_exact" if cohort == "eicu" else "labevent_item"
        lab = next((item for item in specs if item[1]["source_kind"] == lab_kind), None)
        if lab is not None:
            if cohort == "eicu":
                query, params = module._lab_query(
                    [stay], lab[1], [str(lab[1]["labname"])], horizon_hours=HORIZON_HOURS
                )
            else:
                query, params = module._labevent_query(
                    [stay], lab[1], horizon_hours=HORIZON_HOURS
                )
            expected_horizon = HORIZON_HOURS * 60.0 if cohort == "eicu" else HORIZON_HOURS
            checks.append(_placeholder_check(cohort, "lab", query, params, expected_horizon))
        interval = next(
            (item for item in specs if item[1]["source_kind"] in {"interval_binary", "interval_start"}),
            None,
        )
        if interval is not None:
            query, params = module._interval_query(
                [stay], interval[1], horizon_hours=HORIZON_HOURS
            )
            expected_horizon = HORIZON_HOURS * 60.0 if cohort == "eicu" else HORIZON_HOURS
            checks.append(_placeholder_check(cohort, "interval", query, params, expected_horizon))
    failed = [item for item in checks if not item["pass"]]
    if failed:
        raise RuntimeError(f"dynamic horizon SQL preflight failed: {failed}")
    return {"horizon_hours": HORIZON_HOURS, "checks": checks, "pass": True}


def _placeholder_check(
    cohort: str,
    kind: str,
    query: str,
    params: Sequence[Any],
    expected_horizon_parameter: float,
) -> dict[str, Any]:
    placeholder_count = query.count("%s")
    horizon_present = any(
        float(value) == float(expected_horizon_parameter)
        for value in params
        if isinstance(value, (float, int))
    )
    return {
        "cohort": cohort,
        "query_kind": kind,
        "placeholder_count": placeholder_count,
        "parameter_count": len(params),
        "expected_horizon_parameter": float(expected_horizon_parameter),
        "horizon_parameter_present": bool(horizon_present),
        "pass": placeholder_count == len(params) and horizon_present,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    query_smoke = _query_smoke()
    print(f"P1 SQL parameter preflight: PASS ({len(query_smoke['checks'])} checks)", flush=True)
    if args.query_smoke_only:
        return {"query_smoke": query_smoke}

    result_fidelity = RESULTS / "p1_landmark_cache_fidelity.json"
    result_manifest = RESULTS / "p1_landmark_extraction_manifest.json"
    if result_fidelity.exists() or result_manifest.exists():
        raise FileExistsError(
            "P1 extraction outputs already exist; preserving them rather than overwriting"
        )

    cohorts = ("eicu", "mimiciv") if args.cohort == "all" else (args.cohort,)
    # This v1 extraction is all-cohort atomic at the audit gate.  Requiring both
    # cohorts avoids a misleading one-cohort final manifest or cache pair.
    if cohorts != ("eicu", "mimiciv"):
        raise ValueError("final P1 landmark caches must be extracted and audited for both cohorts")
    targets = [PROCESSED / f"{cohort}_events_landmark_v1.pkl" for cohort in cohorts]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite P1 cache(s): {existing}")

    candidates: dict[str, dict[str, Any]] = {}
    fidelity_cohorts: dict[str, Any] = {}
    cleanup_dirs: list[Path] = []
    code_files = {
        "orchestrator": Path(__file__).resolve(),
        "eicu_extractor": Path(eicu.__file__).resolve(),
        "mimiciv_extractor": Path(mimiciv.__file__).resolve(),
        "frozen_standardization_transform": ROOT / "scripts" / "refit_standardizer.py",
    }
    code_hashes = {name: _sha256_file(path) for name, path in code_files.items()}
    for cohort in cohorts:
        print(f"\n=== P1 {cohort}: verifying frozen inputs and selecting stays ===", flush=True)
        source_cache, split, standardizer_payload, input_hashes = _frozen_inputs(cohort)
        frozen = FROZEN[cohort]
        map_path = ROOT / frozen["map"]
        module = eicu if cohort == "eicu" else mimiciv
        mapping = module._load_map(map_path)
        standardizer = EventStandardizer.from_dict(standardizer_payload)
        database_url = args.eicu_database_url if cohort == "eicu" else args.mimiciv_database_url
        if not database_url:
            raise RuntimeError(
                "eICU PostgreSQL DSN is missing; set EICU_PG_DSN or pass --eicu-database-url"
                if cohort == "eicu"
                else "MIMIC-IV PostgreSQL DSN is missing"
            )
        selected, db_report = _fetch_selected_stays(
            cohort, database_url, mapping, source_cache["examples"]
        )
        records, query_hash, work_dir = _extract_chunks(
            cohort,
            database_url,
            mapping,
            selected,
            map_sha256=input_hashes[str(map_path.relative_to(ROOT))],
            chunk_size=args.chunk_size,
        )
        cleanup_dirs.append(work_dir)
        boundary_recheck = _recheck_extraction_bounds(cohort, records, selected)
        new_examples, cohort_fidelity = _build_candidate(
            cohort,
            source_cache["examples"],
            records,
            selected,
            standardizer_payload,
        )
        split_report = _split_audit(cohort, source_cache, split)
        plan_hash = _sha256_file(PLAN)
        if not split_report["pass"]:
            cohort_fidelity["pass"] = False
            cohort_fidelity.setdefault("errors", []).append("frozen patient split audit failed")
        cohort_fidelity["identity"] = {
            "selected_stay_count": len(selected),
            "source_stay_order_sha256": _sha256_json(
                [int(example["record_id"]) for example in source_cache["examples"]]
            ),
            "selected_database_stay_order_sha256": _sha256_json(
                [int(stay.stay_id) for stay in selected]
            ),
            "database_metadata_check": "pass",
        }
        cohort_fidelity["split"] = split_report
        cohort_fidelity["standardizer_sha256"] = _sha256_json(standardizer_payload)
        cohort_fidelity["mapping_sha256"] = input_hashes[str(map_path.relative_to(ROOT))]
        cohort_fidelity["mapping_fingerprint"] = module._map_fingerprint(mapping)
        cohort_fidelity["query_sql_sha256"] = query_hash
        cohort_fidelity["boundary_recheck"] = boundary_recheck
        cohort_fidelity["schema_check"] = db_report["schema_check"]
        cohort_fidelity["database_queue"] = db_report["queue"]
        cohort_fidelity["pass"] = bool(cohort_fidelity["pass"] and split_report["pass"])
        fidelity_cohorts[cohort] = cohort_fidelity
        if not cohort_fidelity["pass"]:
            failure_path = (
                RESULTS
                / "_work"
                / f"p1_landmark_cache_fidelity_failed_{cohort}_"
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
            )
            _atomic_json(
                failure_path,
                {
                    "analysis_plan": PLAN_VERSION,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "plan_sha256": plan_hash,
                    "query_smoke": query_smoke,
                    "overall_pass": False,
                    "cohorts": fidelity_cohorts,
                    "training_gate": "STOP: fidelity must pass before any P1 training",
                },
            )
            raise RuntimeError(
                f"{cohort} P1 fidelity gate failed; no new final cache was written. "
                f"Details: {failure_path}"
            )
        new_cache = dict(source_cache)
        new_cache["examples"] = new_examples
        new_cache["landmark_cache_version"] = PLAN_VERSION
        new_cache["full_event_stream_key"] = "events"
        new_cache["full_event_horizon_hours"] = HORIZON_HOURS
        new_cache["event_time_boundary_precision"] = "float64 time_raw; float32 model time"
        new_cache["source_cache_sha256"] = frozen["cache_sha256"]
        new_cache["source_split_sha256"] = frozen["split_sha256"]
        new_cache["analysis_plan_sha256"] = plan_hash
        candidates[cohort] = {
            "cache": new_cache,
            "input_hashes": input_hashes,
            "mapping": mapping,
            "db_report": db_report,
            "query_hash": query_hash,
            "new_examples": new_examples,
            "database_name": _db_name(cohort),
            "code_hashes": code_hashes,
        }
        print(
            f"{cohort}: fidelity PASS; stays={len(new_examples)}; "
            f"events={cohort_fidelity['n_events_in_new_full_stream']}; "
            f"D<48 stays={cohort_fidelity['d_lt_48_stays']}",
            flush=True,
        )

    # Promote only after both cohorts have passed. Keep a single frozen split
    # source; no replacement or regeneration occurs here.
    cache_hashes: dict[str, str] = {}
    staged_paths: list[Path] = []
    try:
        for cohort in cohorts:
            final_path = PROCESSED / f"{cohort}_events_landmark_v1.pkl"
            if final_path.exists():
                raise FileExistsError(f"refusing to overwrite P1 cache: {final_path}")
            staged_path = final_path.with_name(final_path.name + ".candidate.tmp")
            _atomic_pickle(staged_path, candidates[cohort]["cache"])
            staged_paths.append(staged_path)
            cache_hashes[cohort] = _sha256_file(staged_path)
        for cohort, staged_path in zip(cohorts, staged_paths):
            final_path = PROCESSED / f"{cohort}_events_landmark_v1.pkl"
            os.replace(staged_path, final_path)
    except Exception:
        for path in staged_paths:
            if path.exists():
                path.unlink()
        raise

    for cohort in cohorts:
        fidelity_cohorts[cohort]["new_cache_sha256"] = cache_hashes[cohort]
        fidelity_cohorts[cohort]["new_cache_path"] = str(
            (PROCESSED / f"{cohort}_events_landmark_v1.pkl").relative_to(ROOT)
        )
    fidelity_payload = {
        "analysis_plan": PLAN_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": _sha256_file(PLAN),
        "plan_first_commit": "05e2590",
        "query_smoke": query_smoke,
        "overall_pass": all(fidelity_cohorts[cohort]["pass"] for cohort in cohorts),
        "cohorts": fidelity_cohorts,
        "training_gate": "PASS" if all(fidelity_cohorts[c]["pass"] for c in cohorts) else "STOP",
    }
    if not fidelity_payload["overall_pass"]:
        raise RuntimeError("internal error: cache promotion reached with a failed fidelity audit")
    _atomic_json(result_fidelity, fidelity_payload)

    manifest_cohorts: dict[str, Any] = {}
    for cohort in cohorts:
        item = candidates[cohort]
        cache_path = PROCESSED / f"{cohort}_events_landmark_v1.pkl"
        split_path = PROCESSED / FROZEN[cohort]["split"]
        standardizer_hash = _sha256_json(item["cache"]["standardizer"])
        event_time = np.concatenate(
            [np.asarray(example["events"]["time_raw"], dtype=np.float64) for example in item["new_examples"]]
        )
        manifest_cohorts[cohort] = {
            "database": item["database_name"],
            "cache_path": str(cache_path.relative_to(ROOT)),
            "cache_sha256": cache_hashes[cohort],
            "source_cache_path": FROZEN[cohort]["cache"],
            "source_cache_sha256": item["input_hashes"][str((PROCESSED / FROZEN[cohort]["cache"]).relative_to(ROOT))],
            "split_path": str(split_path.relative_to(ROOT)),
            "split_sha256": item["input_hashes"][str(split_path.relative_to(ROOT))],
            "split_group_column": FROZEN[cohort]["group_column"],
            "split_sizes": FROZEN[cohort]["split_sizes"],
            "variable_map_path": FROZEN[cohort]["map"],
            "variable_map_sha256": item["input_hashes"][str((ROOT / FROZEN[cohort]["map"]).relative_to(ROOT))],
            "variable_map_fingerprint": item["mapping"].get("mapping_fingerprint")
            or (eicu if cohort == "eicu" else mimiciv)._map_fingerprint(item["mapping"]),
            "standardizer_sha256": standardizer_hash,
            "n_stays": len(item["new_examples"]),
            "n_events": int(sum(len(example["events"]["var"]) for example in item["new_examples"])),
            "n_events_by_variable": fidelity_cohorts[cohort]["events_by_variable"],
            "event_time_hours": {
                "min": float(np.min(event_time)) if event_time.size else None,
                "max": float(np.max(event_time)) if event_time.size else None,
                "range_rule": "0 <= s < min(icu_outtime_hour, 192)",
                "model_time_dtype": "float32",
                "boundary_time_dtype": "float64",
                "stable_order_key": "events.event_order (chronological extractor sort order)",
            },
            "icu_outtime_used_only_for_extraction_and_window_censoring": True,
            "query_sql_sha256": item["query_hash"],
            "schema_check": item["db_report"]["schema_check"],
            "database_queue": item["db_report"]["queue"],
            "frozen_input_hashes": item["input_hashes"],
        }
    manifest = {
        "analysis_plan": PLAN_VERSION,
        "analysis_plan_path": str(PLAN.relative_to(ROOT)),
        "analysis_plan_sha256": _sha256_file(PLAN),
        "analysis_plan_first_commit": "05e2590",
        "extraction_commit": _git_head(),
        "git_status_before_extraction": _git_status(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "database_credentials_recorded": False,
        "data_source": "local read-only PostgreSQL databases; only SELECT queries used",
        "extraction_range": "0 <= relative_icu_hour < min(icu_outtime_hour, 192)",
        "event_window_convention": "half-open intervals",
        "standardization_arithmetic": (
            "float32 raw cache value promoted to float64, median/IQR normalization, "
            "frozen clipping, final float32 cast; matched against scripts/refit_standardizer.py"
        ),
        "query_parameter_preflight": query_smoke,
        "code_sha256": {name: _sha256_file(path) for name, path in code_files.items()},
        "cohorts": manifest_cohorts,
        "cache_fidelity_report": str(result_fidelity.relative_to(ROOT)),
        "cache_fidelity_report_sha256": _sha256_file(result_fidelity),
    }
    _atomic_json(result_manifest, manifest)
    if not args.keep_work:
        for work_dir in cleanup_dirs:
            if work_dir.exists():
                import shutil

                shutil.rmtree(work_dir)
        work_parent = PROCESSED / ".p1_landmark_extract_work"
        if work_parent.exists() and not any(work_parent.iterdir()):
            work_parent.rmdir()
    print(f"\nP1 extraction and cache fidelity PASS: {result_manifest}", flush=True)
    return {"fidelity": fidelity_payload, "manifest": manifest}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("all", "eicu", "mimiciv"), default="all")
    parser.add_argument("--eicu-database-url", default=os.environ.get("EICU_PG_DSN"))
    parser.add_argument(
        "--mimiciv-database-url",
        default=os.environ.get("MIMICIV_DATABASE_URL", mimiciv.DEFAULT_DATABASE_URL),
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--query-smoke-only", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    return args


if __name__ == "__main__":
    run(_parse_args())
