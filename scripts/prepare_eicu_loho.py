#!/usr/bin/env python
"""Prepare the complete eICU cohort required by the LOHO protocol.

The random 12k cache is intentionally not accepted here.  This entry point
rechecks the live cohort, requires exactly the preregistered 145 hospitals
with at least 100 raw long-stay rows, and extracts every eligible stay from
those hospitals with the frozen MIMIC-IV standardizer.  A connection or cohort
contract failure produces auditable blocking evidence and never creates a
partial LOHO cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.eicu_extract import (  # noqa: E402
    DEFAULT_MIMIC_CACHE,
    DEFAULT_MAP,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_REFERENCE_RANGES,
    DEFAULT_RESULTS_DIR,
    DatabaseClient,
    _atomic_json,
    _fetch_cohort,
    _fetch_long_stay_hospital_counts,
    _load_map,
    extract,
    schema_check,
)


DEFAULT_DATABASE_URL = "postgresql://127.0.0.1:15434/eicu"
DEFAULT_MIN_HOSPITAL_STAYS = 100
DEFAULT_EXPECTED_HOSPITALS = 145


def _blocked(
    output_dir: Path,
    *,
    database_url: str,
    reason: str,
    detail: str,
    expected_hospitals: int,
    minimum_stays: int,
) -> int:
    next_step = (
        "restore the read-only eICU PostgreSQL tunnel and rerun this entry point"
        if reason == "database_unavailable"
        else "resolve the live E-T0/cohort contract mismatch and rerun this entry point"
    )
    payload = {
        "protocol": "E-LOHO",
        "status": "blocked",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "database": database_url.split("@")[-1],
        "reason": reason,
        "detail": detail,
        "required": {
            "expected_hospitals": int(expected_hospitals),
            "minimum_long_stay_rows_per_hospital": int(minimum_stays),
            "cohort": "first unit stay, age >=16, unitdischargeoffset >=2880, alive/expired",
            "cache": "complete selected-hospital cohort; never the 12k random sample",
        },
        "next_step": next_step,
    }
    _atomic_json(output_dir / "loho_blocked.json", payload)
    (output_dir / "loho_blocked.md").write_text(
        "# eICU E-LOHO blocked\n\n"
        f"- Reason: **{reason}**\n"
        f"- Detail: `{detail}`\n"
        f"- Required hospitals: **{expected_hospitals}** with at least **{minimum_stays}** raw long-stay rows each.\n"
        "- No LOHO cache or training result was accepted.\n"
        "- The 12k random eICU cache is not a substitute.\n\n"
        f"Next step: {next_step}.\n",
        encoding="utf-8",
    )
    print(output_dir / "loho_blocked.json")
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("EICU_PG_DSN", DEFAULT_DATABASE_URL))
    parser.add_argument("--map", dest="map_path", default=str(DEFAULT_MAP))
    parser.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--mimic-standardizer-cache", default=str(DEFAULT_MIMIC_CACHE))
    parser.add_argument("--reference-ranges", default=str(DEFAULT_REFERENCE_RANGES))
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--minimum-hospital-stays", type=int, default=DEFAULT_MIN_HOSPITAL_STAYS)
    parser.add_argument("--expected-hospitals", type=int, default=DEFAULT_EXPECTED_HOSPITALS)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    minimum = int(args.minimum_hospital_stays)
    expected_hospitals = int(args.expected_hospitals)
    if minimum < 1 or expected_hospitals < 1:
        raise SystemExit("minimum-hospital-stays and expected-hospitals must be positive")

    try:
        mapping = _load_map(args.map_path)
        client = DatabaseClient(args.database_url)
        try:
            schema = schema_check(client, mapping)
            if not schema["verified"]:
                return _blocked(
                    output_dir,
                    database_url=args.database_url,
                    reason="schema_contract_failed",
                    detail=str(schema.get("missing", [])),
                    expected_hospitals=expected_hospitals,
                    minimum_stays=minimum,
                )
            long_stay_hospital_counts = _fetch_long_stay_hospital_counts(client, mapping)
            stays, queue_metadata, _ = _fetch_cohort(client, mapping)
        finally:
            client.close()
    except Exception as exc:  # connection failure is an explicit protocol block
        return _blocked(
            output_dir,
            database_url=args.database_url,
            reason="database_unavailable",
            detail=f"{type(exc).__name__}: {exc}",
            expected_hospitals=expected_hospitals,
            minimum_stays=minimum,
        )

    counts = Counter(stay.hospital_id for stay in stays)
    selected_hospitals = sorted(
        hospital_id
        for hospital_id, count in long_stay_hospital_counts.items()
        if count >= minimum
    )
    if len(selected_hospitals) != expected_hospitals:
        return _blocked(
            output_dir,
            database_url=args.database_url,
            reason="cohort_contract_mismatch",
            detail=(
                f"live patient table has {len(selected_hospitals)} hospitals meeting "
                f"the >= {minimum} raw long-stay-row threshold, expected {expected_hospitals}"
            ),
            expected_hospitals=expected_hospitals,
            minimum_stays=minimum,
        )
    selected_hospital_set = set(selected_hospitals)
    missing_eligible = sorted(
        hospital_id for hospital_id in selected_hospital_set if hospital_id not in counts
    )
    if missing_eligible:
        return _blocked(
            output_dir,
            database_url=args.database_url,
            reason="selected_hospital_has_no_eligible_stays",
            detail=(
                "selected long-stay hospitals have no rows in the registered "
                f"forecast cohort: {missing_eligible}"
            ),
            expected_hospitals=expected_hospitals,
            minimum_stays=minimum,
        )
    selected = [stay for stay in stays if stay.hospital_id in selected_hospital_set]
    selected.sort(key=lambda stay: stay.stay_id)
    stay_ids = [int(stay.stay_id) for stay in selected]
    queue = {
        "protocol": "E-LOHO",
        "status": "ready_for_full_extraction",
        "cohort_contract": queue_metadata,
        "minimum_hospital_stays": minimum,
        "expected_hospitals": expected_hospitals,
        "selected_hospitals": selected_hospitals,
        "selected_hospital_long_stay_counts": {
            str(hospital): int(long_stay_hospital_counts[hospital])
            for hospital in selected_hospitals
        },
        "selected_hospital_eligible_stay_counts": {
            str(hospital): int(counts[hospital]) for hospital in selected_hospitals
        },
        "selected_stays": len(selected),
        "selected_stay_ids_sha256": hashlib.sha256(
            ",".join(str(value) for value in stay_ids).encode("utf-8")
        ).hexdigest(),
        "sampling": "all eligible stays from hospitals with >=100 raw long-stay rows",
        "split_seed": 42,
        "fold_seed": 42,
    }
    _atomic_json(output_dir / "loho_queue.json", queue)
    if args.check_only:
        print(output_dir / "loho_queue.json")
        return 0

    result = extract(
        map_path=args.map_path,
        database_url=args.database_url,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        mimic_standardizer_cache=args.mimic_standardizer_cache,
        sample_size=len(selected),
        sample_seed=42,
        split_seed=42,
        chunk_size=int(args.chunk_size),
        cache_name="eicu_loho_events.pkl",
        split_name="eicu_loho_split_indices.json",
        cohort_name="eicu_loho_cohort.csv",
        subject_map_name="eicu_loho_subject_map.csv",
        manifest_name="eicu_loho_manifest.json",
        reference_ranges=args.reference_ranges,
        work_dir=Path(args.processed_dir) / "eicu_loho_extract_work",
        update_map=False,
        keep_work=False,
        hospital_min_stays=minimum,
    )
    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    filter_report = manifest.get("queue", {}).get("filter", {})
    if int(manifest.get("examples", -1)) != len(selected) or int(
        filter_report.get("selected_hospitals", -1)
    ) != expected_hospitals:
        raise RuntimeError("LOHO extraction finalization failed the selected-hospital contract")
    for stale_path in (
        output_dir / "loho_blocked.json",
        output_dir / "loho_blocked.md",
    ):
        stale_path.unlink(missing_ok=True)
    queue["status"] = "extracted"
    queue["extraction"] = result
    _atomic_json(output_dir / "loho_queue.json", queue)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
