"""Extract a leakage-safe, P12-compatible event cache from MIMIC-IV.

The extractor is deliberately independent of the model and training code.  It
reads the auditable variable map, verifies the live PostgreSQL schema, samples
the adult first-ICU-stay cohort, and writes the same cache/split contract used
by :mod:`src.data.p12_events`.

Extraction is resumable at stay-id chunks.  A completed chunk is written by an
atomic rename, so a dropped database connection cannot leave a partially
accepted chunk in the final cache.  The chunk directory is removed after a
successful finalization unless ``--keep-work`` is explicitly requested.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from .p12_events import (
    TIME_SERIES_VARIABLES,
    EventStandardizer,
    annotate_events,
    make_event_example,
)

try:  # psycopg is installed in the specified icu_kan environment.
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - exercised only outside the target env
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP = ROOT / "configs/mimiciv_variable_map.yaml"
DEFAULT_DATABASE_URL = "postgresql://127.0.0.1:15433/mimiciv"
DEFAULT_PROCESSED_DIR = ROOT / "data/processed"
DEFAULT_RESULTS_DIR = ROOT / "results/mimiciv"
DEFAULT_REFERENCE_RANGES = DEFAULT_RESULTS_DIR / "p12_reference_ranges.csv"
DEFAULT_SAMPLE_SIZE = 12_000
DEFAULT_SAMPLE_SEED = 42
DEFAULT_SPLIT_SEED = 42
DEFAULT_T_OBS = 24.0
DEFAULT_HORIZON = 48.0
DEFAULT_CHUNK_SIZE = 500
DEFAULT_QUERY_TIMEOUT_MS = 300_000
STRUCTURAL_THRESHOLD = 100
IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


def stratified_group_split(
    labels: Sequence[int],
    groups: Sequence[object],
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    fractions: Sequence[float] = (0.70, 0.15, 0.15),
) -> dict[str, list[int]]:
    """Split records without separating records from the same patient.

    The unit assigned to a partition is ``groups`` (MIMIC-IV ``subject_id``),
    while the objective preserves record-level positive-label counts as closely
    as the indivisible groups allow.  A patient's records are never split, even
    when that patient has multiple ICU stays or mixed stay labels.  The greedy
    assignment is deterministic for a fixed seed and is deliberately kept as a
    pure function so its leakage contract can be mutation-tested without a
    database connection.
    """

    if len(labels) != len(groups) or not labels:
        raise ValueError("labels and groups must have the same non-zero length")
    if len(fractions) != 3 or any(float(value) <= 0.0 for value in fractions):
        raise ValueError("fractions must contain three positive values")
    total_fraction = float(sum(fractions))
    if not np.isclose(total_fraction, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("split fractions must sum to one")

    label_array = np.asarray(labels, dtype=np.int64)
    if not np.isin(label_array, (0, 1)).all():
        raise ValueError("labels must be binary")
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        if group is None:
            raise ValueError("group identifiers must not be missing")
        group_to_indices[str(group)].append(int(index))

    rng = np.random.default_rng(int(seed))
    group_records = []
    for group, indices in group_to_indices.items():
        group_records.append(
            {
                "group": group,
                "indices": indices,
                "n": len(indices),
                "positive": int(label_array[indices].sum()),
                "tie": float(rng.random()),
            }
        )
    # Large groups are placed first; the random tie-break makes equal-sized
    # groups seed-dependent without relying on hash iteration order.
    group_records.sort(key=lambda item: (-item["n"], -item["positive"], item["tie"]))

    target_n = np.asarray(fractions, dtype=float) * len(label_array)
    target_positive = np.asarray(fractions, dtype=float) * float(label_array.sum())
    assigned_n = np.zeros(3, dtype=float)
    assigned_positive = np.zeros(3, dtype=float)
    partitions: list[list[int]] = [[], [], []]
    partition_groups: list[list[str]] = [[], [], []]
    for item in group_records:
        scores = []
        for partition in range(3):
            new_n = assigned_n.copy()
            new_positive = assigned_positive.copy()
            new_n[partition] += item["n"]
            new_positive[partition] += item["positive"]
            size_loss = np.sum(((new_n - target_n) / np.maximum(target_n, 1.0)) ** 2)
            label_loss = np.sum(
                ((new_positive - target_positive) / np.maximum(target_positive, 1.0)) ** 2
            )
            scores.append(float(size_loss + 2.0 * label_loss))
        selected = int(np.argmin(scores))
        partitions[selected].extend(item["indices"])
        partition_groups[selected].append(str(item["group"]))
        assigned_n[selected] += item["n"]
        assigned_positive[selected] += item["positive"]

    if any(not partition for partition in partitions):
        if len(group_records) < 3:
            raise ValueError("at least three groups are required for a three-way split")
        for empty in [index for index, values in enumerate(partitions) if not values]:
            source = max(
                (
                    index
                    for index, values in enumerate(partition_groups)
                    if len(values) > 1
                ),
                key=lambda index: len(partition_groups[index]),
                default=None,
            )
            if source is None:
                raise ValueError("grouped split could not populate all partitions")
            moved_group = partition_groups[source].pop()
            moved_indices = group_to_indices[moved_group]
            partitions[source] = [
                index for index in partitions[source] if index not in moved_indices
            ]
            partitions[empty].extend(moved_indices)
            partition_groups[empty].append(moved_group)
    return {
        name: sorted(indices)
        for name, indices in zip(("train", "val", "test"), partitions)
    }


def _group_split_audit(
    groups: Sequence[object],
    labels: Sequence[int],
    split_indices: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    """Return and validate the identity-level split contract."""

    expected = list(range(len(groups)))
    observed = sorted(int(index) for values in split_indices.values() for index in values)
    if observed != expected:
        raise AssertionError("split indices do not cover each record exactly once")
    group_sets = {
        name: {str(groups[int(index)]) for index in split_indices[name]}
        for name in ("train", "val", "test")
    }
    overlaps = {
        f"{left}&{right}": len(group_sets[left] & group_sets[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    if any(overlaps.values()):
        raise AssertionError(f"group leakage detected: {overlaps}")
    label_array = np.asarray(labels, dtype=np.int64)
    return {
        "split_unit": "subject_id_grouped",
        "group_column": "subject_id",
        "n_groups": int(len(set(str(group) for group in groups))),
        "n_groups_by_split": {name: len(values) for name, values in group_sets.items()},
        "sizes": {name: len(split_indices[name]) for name in ("train", "val", "test")},
        "label_counts": {
            name: int(label_array[list(split_indices[name])].sum())
            for name in ("train", "val", "test")
        },
        "group_overlap": overlaps,
    }


@dataclass(frozen=True)
class Stay:
    """The queue fields required to assign observations to one ICU stay."""

    stay_id: int
    subject_id: int
    hadm_id: int
    icu_intime: datetime
    icu_outtime: datetime
    label: int
    first_careunit: str
    icu_type: int

    def as_query_tuple(self) -> tuple[int, int, datetime]:
        return self.stay_id, self.hadm_id, self.icu_intime


class DatabaseClient:
    """Small reconnecting read-only client for chunked extraction queries."""

    def __init__(
        self,
        database_url: str,
        *,
        timeout_ms: int = DEFAULT_QUERY_TIMEOUT_MS,
        retries: int = 5,
        retry_delay: float = 2.0,
    ) -> None:
        if psycopg is None:
            raise RuntimeError(
                "psycopg is required; install the pinned project environment"
            )
        self.database_url = database_url
        self.timeout_ms = int(timeout_ms)
        self.retries = int(retries)
        self.retry_delay = float(retry_delay)
        self.conn = self._connect()

    def _connect(self):
        connection = psycopg.connect(
            self.database_url,
            connect_timeout=15,
            row_factory=dict_row,
        )
        connection.autocommit = True
        connection.execute(f"SET statement_timeout = {self.timeout_ms}")
        connection.execute("SET lock_timeout = 15000")
        return connection

    def close(self) -> None:
        if self.conn is not None and not self.conn.closed:
            self.conn.close()

    def fetchall(self, query: str, params: Sequence[object] = ()) -> list[dict[str, Any]]:
        """Execute one read query, reconnecting only after database errors."""

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with self.conn.cursor() as cursor:
                    cursor.execute(query, params)
                    return list(cursor.fetchall())
            except (psycopg.Error, OSError) as exc:  # type: ignore[union-attr]
                last_error = exc
                self.close()
                if attempt >= self.retries:
                    break
                time.sleep(self.retry_delay * (attempt + 1))
                self.conn = self._connect()
        raise RuntimeError(
            f"MIMIC-IV query failed after {self.retries + 1} attempts"
        ) from last_error


def _qident(value: object) -> str:
    text = str(value)
    if not IDENTIFIER.fullmatch(text):
        raise ValueError(f"unsafe SQL identifier in variable map: {text!r}")
    return f'"{text}"'


def _load_map(path: str | Path = DEFAULT_MAP) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    names = tuple(str(name) for name in payload.get("variable_names", ()))
    if names != TIME_SERIES_VARIABLES:
        raise ValueError(
            "MIMIC-IV variable_names must exactly match P12 TIME_SERIES_VARIABLES"
        )
    variables = payload.get("variables")
    if not isinstance(variables, dict) or tuple(variables) != names:
        raise ValueError("variable map must contain exactly one entry for each P12 variable")
    for expected_id, name in enumerate(names):
        spec = variables[name]
        if not isinstance(spec, dict) or int(spec.get("var_id", -1)) != expected_id:
            raise ValueError(f"invalid var_id for {name}")
        if spec.get("verified") is not True:
            raise ValueError(f"mapping {name} is not verified; T0 must finish first")
        if not spec.get("source_schema") or not spec.get("source_table"):
            raise ValueError(f"mapping {name} has no source table")
        if spec.get("join_key") not in {"stay_id", "hadm_id"}:
            raise ValueError(f"mapping {name} has an invalid join_key")
    return payload


def _map_fingerprint(mapping: Mapping[str, Any]) -> str:
    canonical = json.dumps(mapping, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _required_columns(mapping: Mapping[str, Any]) -> dict[tuple[str, str], set[str]]:
    required: dict[tuple[str, str], set[str]] = defaultdict(set)
    for spec in mapping["variables"].values():
        key = (str(spec["source_schema"]), str(spec["source_table"]))
        required[key].add(str(spec.get("source_column", "")))
        for column in spec.get("source_columns", ()):  # FiO2 coalesce
            required[key].add(str(column))
        for field in ("start_column", "end_column", "status_column"):
            if spec.get(field):
                required[key].add(str(spec[field]))
        if spec.get("filter") == "specimen_art":
            required[key].add("specimen")
        if str(spec.get("source_kind")) == "labevent_item":
            required[key].update({"itemid", "valuenum", "charttime", "labevent_id"})
    required[("mimiciv_derived", "icustay_detail")].update(
        {
            "subject_id",
            "hadm_id",
            "stay_id",
            "hospital_expire_flag",
            "first_icu_stay",
            "admission_age",
            "los_icu",
            "icu_intime",
            "icu_outtime",
        }
    )
    required[("mimiciv_icu", "icustays")].update(
        {"stay_id", "first_careunit"}
    )
    required[("mimiciv_hosp", "d_labitems")].update(
        {"itemid", "label", "fluid", "category"}
    )
    return required


def schema_check(
    client: DatabaseClient,
    mapping: Mapping[str, Any],
) -> dict[str, Any]:
    """Assert every configured table, column, and lab item against live SQL."""

    rows = client.fetchall(
        """
        SELECT table_schema, table_name, column_name
        FROM information_schema.columns
        WHERE table_schema IN ('mimiciv_hosp', 'mimiciv_icu', 'mimiciv_derived')
        """
    )
    actual = {
        (str(row["table_schema"]), str(row["table_name"]), str(row["column_name"]))
        for row in rows
    }
    missing: list[str] = []
    table_report: dict[str, dict[str, Any]] = {}
    for (schema, table), columns in sorted(_required_columns(mapping).items()):
        missing_columns = sorted(
            column
            for column in columns
            if column and (schema, table, column) not in actual
        )
        key = f"{schema}.{table}"
        table_report[key] = {
            "required_columns": sorted(column for column in columns if column),
            "missing_columns": missing_columns,
            "verified": not missing_columns,
        }
        missing.extend(f"{key}.{column}" for column in missing_columns)

    item_specs: dict[int, tuple[str, str]] = {}
    for name, spec in mapping["variables"].items():
        for itemid, label in (spec.get("item_labels") or {}).items():
            item_specs[int(itemid)] = (name, str(label))
    item_report: dict[str, Any] = {}
    if item_specs:
        item_rows = client.fetchall(
            """
            SELECT itemid, label, fluid, category
            FROM mimiciv_hosp.d_labitems
            WHERE itemid = ANY(%s)
            """,
            (sorted(item_specs),),
        )
        by_id = {int(row["itemid"]): row for row in item_rows}
        for itemid, (variable, expected_label) in sorted(item_specs.items()):
            row = by_id.get(itemid)
            found_label = None if row is None else str(row["label"])
            verified = row is not None and found_label.casefold() == expected_label.casefold()
            item_report[str(itemid)] = {
                "variable": variable,
                "expected_label": expected_label,
                "found_label": found_label,
                "verified": verified,
            }
            if not verified:
                missing.append(f"mimiciv_hosp.d_labitems.itemid={itemid}")

    return {
        "verified": not missing,
        "missing": missing,
        "tables": table_report,
        "lab_items": item_report,
        "mapping_fingerprint": _map_fingerprint(mapping),
    }


def _fetch_queue(client: DatabaseClient) -> tuple[list[Stay], dict[str, int], dict[str, int]]:
    count_row = client.fetchall(
        """
        SELECT
            count(*) AS total_icustay_detail,
            count(*) FILTER (WHERE first_icu_stay IS TRUE) AS after_first_icu,
            count(*) FILTER (
                WHERE first_icu_stay IS TRUE AND los_icu >= 2.0
            ) AS after_los,
            count(*) FILTER (
                WHERE first_icu_stay IS TRUE
                  AND los_icu >= 2.0
                  AND admission_age >= 16
            ) AS after_age,
            count(*) FILTER (
                WHERE first_icu_stay IS TRUE
                  AND los_icu >= 2.0
                  AND admission_age >= 16
                  AND hospital_expire_flag IS NOT NULL
            ) AS after_label
        FROM mimiciv_derived.icustay_detail
        """
    )[0]
    rows = client.fetchall(
        """
        SELECT
            d.subject_id,
            d.hadm_id,
            d.stay_id,
            d.hospital_expire_flag,
            d.first_icu_stay,
            d.admission_age,
            d.los_icu,
            d.icu_intime,
            d.icu_outtime,
            i.first_careunit
        FROM mimiciv_derived.icustay_detail AS d
        JOIN mimiciv_icu.icustays AS i USING (stay_id)
        WHERE d.first_icu_stay IS TRUE
          AND d.los_icu >= 2.0
          AND d.admission_age >= 16
        ORDER BY d.stay_id
        """
    )
    if not rows:
        raise ValueError("MIMIC-IV adult first-ICU-stay queue is empty")
    careunits = sorted({str(row["first_careunit"]) for row in rows})
    if any(row["first_careunit"] is None for row in rows):
        raise ValueError("first_careunit contains NULL values in the eligible queue")
    careunit_map = {name: index + 1 for index, name in enumerate(careunits)}
    stays: list[Stay] = []
    for row in rows:
        label = int(row["hospital_expire_flag"])
        if label not in (0, 1):
            raise ValueError(f"hospital_expire_flag is not binary for stay {row['stay_id']}")
        intime = row["icu_intime"]
        outtime = row["icu_outtime"]
        if not isinstance(intime, datetime) or not isinstance(outtime, datetime):
            raise ValueError(f"missing ICU timestamps for stay {row['stay_id']}")
        careunit = str(row["first_careunit"])
        stays.append(
            Stay(
                stay_id=int(row["stay_id"]),
                subject_id=int(row["subject_id"]),
                hadm_id=int(row["hadm_id"]),
                icu_intime=intime,
                icu_outtime=outtime,
                label=label,
                first_careunit=careunit,
                icu_type=careunit_map[careunit],
            )
        )
    counts = {
        "total_icustay_detail": int(count_row["total_icustay_detail"]),
        "eligible_after_first_icu": int(count_row["after_first_icu"]),
        "eligible_after_los": int(count_row["after_los"]),
        "eligible_after_age": int(count_row["after_age"]),
        "eligible_with_label": int(count_row["after_label"]),
        "eligible_joined_with_careunit": len(stays),
    }
    return stays, counts, careunit_map


def _sample_stays(
    stays: Sequence[Stay],
    *,
    sample_size: int,
    seed: int,
) -> list[Stay]:
    if sample_size < 1 or sample_size > len(stays):
        raise ValueError(f"sample_size must be between 1 and {len(stays)}")
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(len(stays), size=int(sample_size), replace=False))
    selected = [stays[int(index)] for index in indices]
    if len({stay.stay_id for stay in selected}) != len(selected):
        raise AssertionError("simple random sampling produced duplicate stay_id values")
    return selected


def _values_cte(stays: Sequence[Stay]) -> tuple[str, list[object]]:
    if not stays:
        raise ValueError("cannot build a query for an empty stay chunk")
    placeholders = ",".join(["(%s,%s,%s)"] * len(stays))
    params: list[object] = []
    for stay in stays:
        params.extend(stay.as_query_tuple())
    return (
        "WITH selected(stay_id, hadm_id, icu_intime) AS "
        f"(VALUES {placeholders})",
        params,
    )


def _time_predicate(
    alias: str = "e", horizon_hours: float = DEFAULT_HORIZON
) -> str:
    if not math.isfinite(float(horizon_hours)) or float(horizon_hours) <= 0:
        raise ValueError("horizon_hours must be finite and positive")
    return (
        f"{alias}.charttime >= s.icu_intime "
        f"AND {alias}.charttime < s.icu_intime + (%s * INTERVAL '1 hour')"
    )


def _numeric_query(
    stays: Sequence[Stay],
    specs: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    horizon_hours: float = DEFAULT_HORIZON,
) -> tuple[str, list[object]]:
    cte, params = _values_cte(stays)
    params.append(float(horizon_hours))
    first = specs[0][1]
    schema = _qident(first["source_schema"])
    table = _qident(first["source_table"])
    select_columns = ["s.stay_id", "s.hadm_id", "e.charttime"]
    nonnull: list[str] = []
    for name, spec in specs:
        if spec.get("source_columns"):
            columns = [f"e.{_qident(column)}" for column in spec["source_columns"]]
            expression = f"COALESCE({','.join(columns)})"
        else:
            expression = f"e.{_qident(spec['source_column'])}"
        nonnull.append(f"({expression}) IS NOT NULL")
        if spec.get("filter") == "specimen_art":
            expression = (
                "CASE WHEN regexp_replace(upper(trim(coalesce(e.specimen, ''))), "
                "'[^A-Z]', '', 'g') = 'ART' "
                f"THEN {expression} END"
            )
            nonnull[-1] = f"({expression}) IS NOT NULL"
        select_columns.append(f'{expression} AS "{name}"')
    join_key = str(first["join_key"])
    query = (
        f"{cte} SELECT {', '.join(select_columns)} "
        f"FROM selected AS s JOIN {schema}.{table} AS e "
        f"ON e.{_qident(join_key)} = s.{_qident(join_key)} "
        f"WHERE {_time_predicate(horizon_hours=horizon_hours)} AND ({' OR '.join(nonnull)}) "
        "ORDER BY s.stay_id, e.charttime"
    )
    return query, params


def _labevent_query(
    stays: Sequence[Stay],
    spec: Mapping[str, Any],
    *,
    horizon_hours: float = DEFAULT_HORIZON,
) -> tuple[str, list[object]]:
    cte, params = _values_cte(stays)
    itemids = sorted(int(itemid) for itemid in spec.get("itemids", ()))
    if not itemids:
        raise ValueError(f"labevent mapping for {spec} has no itemids")
    params.append(itemids)
    params.append(float(horizon_hours))
    query = (
        f"{cte} SELECT s.stay_id, s.hadm_id, e.charttime, e.itemid, "
        "e.valuenum, e.labevent_id "
        f"FROM selected AS s JOIN {_qident(spec['source_schema'])}.{_qident(spec['source_table'])} AS e "
        f"ON e.{_qident(spec['join_key'])} = s.{_qident(spec['join_key'])} "
        f"WHERE e.itemid = ANY(%s) AND e.valuenum IS NOT NULL "
        f"AND {_time_predicate(horizon_hours=horizon_hours)} "
        "ORDER BY s.stay_id, e.charttime, e.labevent_id"
    )
    return query, params


def _interval_query(
    stays: Sequence[Stay],
    spec: Mapping[str, Any],
    *,
    horizon_hours: float = DEFAULT_HORIZON,
) -> tuple[str, list[object]]:
    if not math.isfinite(float(horizon_hours)) or float(horizon_hours) <= 0:
        raise ValueError("horizon_hours must be finite and positive")
    cte, params = _values_cte(stays)
    table = f"{_qident(spec['source_schema'])}.{_qident(spec['source_table'])}"
    join_key = _qident(spec["join_key"])
    start = _qident(spec["start_column"])
    if spec["source_kind"] == "interval_binary":
        end = _qident(spec["end_column"])
        status = _qident(spec["status_column"])
        params.append(float(horizon_hours))
        query = (
            f"{cte} SELECT s.stay_id, s.icu_intime, e.{start} AS starttime, "
            f"e.{end} AS endtime, e.{status} AS status "
            f"FROM selected AS s JOIN {table} AS e "
            f"ON e.{join_key} = s.{join_key} "
            f"WHERE upper(trim(e.{status})) = 'INVASIVEVENT' "
            f"AND e.{start} < s.icu_intime + (%s * INTERVAL '1 hour') "
            f"AND (e.{end} IS NULL OR e.{end} >= s.icu_intime) "
            "ORDER BY s.stay_id, e.starttime, e.endtime"
        )
    else:
        value = _qident(spec["source_column"])
        params.append(float(horizon_hours))
        query = (
            f"{cte} SELECT s.stay_id, s.icu_intime, e.{start} AS starttime, "
            f"e.{value} AS value "
            f"FROM selected AS s JOIN {table} AS e "
            f"ON e.{join_key} = s.{join_key} "
            f"WHERE e.{start} >= s.icu_intime "
            f"AND e.{start} < s.icu_intime + (%s * INTERVAL '1 hour') "
            f"AND e.{value} IS NOT NULL "
            "ORDER BY s.stay_id, e.starttime"
        )
    return query, params


def _hours_since(intime: datetime, timestamp: datetime) -> float:
    if not isinstance(timestamp, datetime):
        raise ValueError("database event timestamp is not a datetime")
    return (timestamp - intime).total_seconds() / 3600.0


def _convert_value(name: str, value: object, spec: Mapping[str, Any]) -> tuple[float, bool]:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value for {name}")
    rule = str(spec.get("unit_rule", "identity"))
    converted = False
    if rule == "fahrenheit_above_50_to_celsius":
        if number > 50.0:
            number = (number - 32.0) * 5.0 / 9.0
            converted = True
    elif rule == "fio2_percent_to_fraction":
        # P12 records FiO2 as a fraction: measured min 0.21, max 1.00, and zero
        # of its 95,163 raw records exceed 1.0.  The spec's original
        # "P12 is a percentage, multiply MIMIC by 100" rule was an unverified
        # assumption and is reversed here.  Robust standardisation makes the two
        # conventions equivalent to ~5e-6 relative, so this is about keeping
        # value_raw comparable across datasets, not about model behaviour.
        if number > 1.0:
            number /= 100.0
            converted = True
    elif rule in {
        "identity",
        "gcs_total_3_to_15",
        "arterial_specimen_only",
        "instantaneous_ml_event",
        "interval_start_kg_deduplicate",
        "invasive_vent_interval_boundaries",
    }:
        pass
    else:
        raise ValueError(f"unknown unit rule for {name}: {rule}")
    if name == "GCS" and not 3.0 <= number <= 15.0:
        raise ValueError(f"GCS outside [3, 15]: {number}")
    return number, converted


def _new_record(stay: Stay) -> dict[str, Any]:
    return {
        "record_id": stay.stay_id,
        "sample_id": f"mimiciv-{stay.stay_id}",
        "label": stay.label,
        "set_name": "mimiciv",
        # Keep the patient identity in the cache.  A stay-level cache without
        # this field cannot be audited or split without leakage at patient
        # level, even though the extraction queue already knows subject_id.
        "subject_id": stay.subject_id,
        "hadm_id": stay.hadm_id,
        "static": {"ICUType": stay.icu_type},
        "events": [],
        "_stay_id": stay.stay_id,
    }


def _append_event(
    record: dict[str, Any],
    name: str,
    timestamp: datetime,
    value: object,
    spec: Mapping[str, Any],
    stay: Stay,
    conversion_seen: dict[str, int],
    conversion_hits: dict[str, int],
    sort_key: tuple[object, ...],
    *,
    horizon_hours: float = DEFAULT_HORIZON,
    respect_outtime: bool = False,
) -> None:
    hour = _hours_since(stay.icu_intime, timestamp)
    if not 0.0 <= hour < float(horizon_hours):
        return
    if respect_outtime:
        stay_hours = (stay.icu_outtime - stay.icu_intime).total_seconds() / 3600.0
        if hour >= stay_hours:
            return
    converted_value, converted = _convert_value(name, value, spec)
    conversion_seen[name] += 1
    conversion_hits[name] += int(converted)
    record["events"].append(
        {
            "variable": name,
            "time": float(hour),
            "value": float(converted_value),
            "_sort_key": tuple(sort_key),
        }
    )


def _extract_chunk(
    client: DatabaseClient,
    stays: Sequence[Stay],
    mapping: Mapping[str, Any],
    *,
    horizon_hours: float = DEFAULT_HORIZON,
    respect_outtime: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int], list[str]]:
    if not math.isfinite(float(horizon_hours)) or float(horizon_hours) <= 0:
        raise ValueError("horizon_hours must be finite and positive")
    records = {stay.stay_id: _new_record(stay) for stay in stays}
    variables = mapping["variables"]
    stay_by_id = {stay.stay_id: stay for stay in stays}
    conversion_seen = {name: 0 for name in TIME_SERIES_VARIABLES}
    conversion_hits = {name: 0 for name in TIME_SERIES_VARIABLES}
    sql_fragments: list[str] = []

    grouped: dict[tuple[str, str, str, str], list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for name in TIME_SERIES_VARIABLES:
        spec = variables[name]
        kind = str(spec["source_kind"])
        if kind in {"numeric"}:
            key = (
                str(spec["source_schema"]),
                str(spec["source_table"]),
                str(spec["join_key"]),
                kind,
            )
            grouped[key].append((name, spec))

    for _, specs in sorted(grouped.items()):
        query, params = _numeric_query(
            stays, specs, horizon_hours=horizon_hours
        )
        sql_fragments.append(query)
        rows = client.fetchall(query, params)
        for row in rows:
            stay_id = int(row["stay_id"])
            stay = stay_by_id[stay_id]
            for name, spec in specs:
                value = row.get(name)
                if value is None:
                    continue
                timestamp = row["charttime"]
                _append_event(
                    records[stay_id],
                    name,
                    timestamp,
                    value,
                    spec,
                    stay,
                    conversion_seen,
                    conversion_hits,
                    (timestamp, name, float(value)),
                    horizon_hours=horizon_hours,
                    respect_outtime=respect_outtime,
                )

    lab_specs = [
        (name, mapping["variables"][name])
        for name in TIME_SERIES_VARIABLES
        if mapping["variables"][name]["source_kind"] == "labevent_item"
    ]
    if lab_specs:
        # All labevent variables share one indexed query; itemid resolves the
        # variable after retrieval and preserves the exact configured mapping.
        first_spec = lab_specs[0][1]
        item_to_variable = {
            int(itemid): name
            for name, spec in lab_specs
            for itemid in spec["itemids"]
        }
        all_itemids = sorted(
            {
                int(itemid)
                for _, spec in lab_specs
                for itemid in spec["itemids"]
            }
        )
        lab_query_spec = dict(first_spec)
        lab_query_spec["itemids"] = all_itemids
        query, params = _labevent_query(
            stays, lab_query_spec, horizon_hours=horizon_hours
        )
        sql_fragments.append(query)
        rows = client.fetchall(query, params)
        for row in rows:
            itemid = int(row["itemid"])
            name = item_to_variable.get(itemid)
            if name is None:
                raise AssertionError(f"unmapped configured lab itemid {itemid}")
            stay_id = int(row["stay_id"])
            stay = stay_by_id[stay_id]
            spec = variables[name]
            _append_event(
                records[stay_id],
                name,
                row["charttime"],
                row["valuenum"],
                spec,
                stay,
                conversion_seen,
                conversion_hits,
                (row["charttime"], name, int(row["labevent_id"])),
                horizon_hours=horizon_hours,
                respect_outtime=respect_outtime,
            )

    interval_specs = [
        (name, mapping["variables"][name])
        for name in TIME_SERIES_VARIABLES
        if mapping["variables"][name]["source_kind"]
        in {"interval_binary", "interval_start"}
    ]
    for name, spec in interval_specs:
        query, params = _interval_query(
            stays, spec, horizon_hours=horizon_hours
        )
        sql_fragments.append(query)
        rows = client.fetchall(query, params)
        by_stay = stay_by_id
        if spec["source_kind"] == "interval_binary":
            for row in rows:
                stay_id = int(row["stay_id"])
                stay = by_stay[stay_id]
                start = row["starttime"]
                end = row["endtime"]
                if end is not None and end < start:
                    raise ValueError(f"ventilation interval ends before start for stay {stay_id}")
                _append_event(
                    records[stay_id],
                    name,
                    max(start, stay.icu_intime),
                    1.0,
                    spec,
                    stay,
                    conversion_seen,
                    conversion_hits,
                    (max(start, stay.icu_intime), name, 1),
                    horizon_hours=horizon_hours,
                    respect_outtime=respect_outtime,
                )
                # P12's MechVent is a 1-only indicator: its measured
                # distribution is min = max = 1.0 with a single unique value,
                # which is why it sits in the degenerate-variable set.  Emitting
                # a 0 at extubation would make this a different variable from
                # P12's and change what the encoder sees, so the interval end is
                # deliberately not recorded.
        else:
            previous: dict[int, float] = {}
            for row in rows:
                stay_id = int(row["stay_id"])
                value = float(row["value"])
                if stay_id in previous and value == previous[stay_id]:
                    continue
                previous[stay_id] = value
                stay = by_stay[stay_id]
                _append_event(
                    records[stay_id],
                    name,
                    row["starttime"],
                    value,
                    spec,
                    stay,
                    conversion_seen,
                    conversion_hits,
                    (row["starttime"], name, value),
                    horizon_hours=horizon_hours,
                    respect_outtime=respect_outtime,
                )

    finalized: list[dict[str, Any]] = []
    for stay in stays:
        record = records[stay.stay_id]
        events = sorted(record.pop("events"), key=lambda event: event["_sort_key"])
        triples = [
            (str(event["variable"]), float(event["time"]), float(event["value"]))
            for event in events
        ]
        record["events"] = triples
        record["events_with_delta"] = annotate_events(triples)
        record.pop("_stay_id", None)
        finalized.append(record)
    return finalized, conversion_seen, conversion_hits, sql_fragments


def _atomic_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _part_payload(
    records: Sequence[dict[str, Any]],
    seen: Mapping[str, int],
    hits: Mapping[str, int],
    sql_fragments: Sequence[str],
) -> dict[str, Any]:
    return {
        "records": list(records),
        "conversion_seen": dict(seen),
        "conversion_hits": dict(hits),
        "sql_sha256": hashlib.sha256(
            "\n".join(sql_fragments).encode("utf-8")
        ).hexdigest(),
    }


def _load_or_extract_parts(
    client: DatabaseClient,
    selected: Sequence[Stay],
    mapping: Mapping[str, Any],
    *,
    work_dir: Path,
    chunk_size: int,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int], str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(
        json.dumps(
            {
                "stay_ids": [stay.stay_id for stay in selected],
                "mapping_fingerprint": _map_fingerprint(mapping),
                "chunk_size": int(chunk_size),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    state_path = work_dir / "selection_state.json"
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("signature") != signature:
            raise ValueError(
                f"resume directory {work_dir} belongs to a different selection or map"
            )
    else:
        _atomic_json(
            state_path,
            {
                "signature": signature,
                "stay_ids": [stay.stay_id for stay in selected],
                "mapping_fingerprint": _map_fingerprint(mapping),
                "chunk_size": int(chunk_size),
            },
        )

    records: list[dict[str, Any]] = []
    seen = {name: 0 for name in TIME_SERIES_VARIABLES}
    hits = {name: 0 for name in TIME_SERIES_VARIABLES}
    sql_hashes: list[str] = []
    stay_by_id = {stay.stay_id: stay for stay in selected}
    n_chunks = math.ceil(len(selected) / int(chunk_size))
    for chunk_index in range(n_chunks):
        chunk = selected[
            chunk_index * int(chunk_size) : (chunk_index + 1) * int(chunk_size)
        ]
        part_path = work_dir / f"part-{chunk_index:05d}.pkl"
        if part_path.exists():
            with part_path.open("rb") as handle:
                payload = pickle.load(handle)
            part_records = payload.get("records")
            part_ids = {int(record["record_id"]) for record in part_records or ()}
            if part_ids != {stay.stay_id for stay in chunk}:
                raise ValueError(f"part {part_path} does not match the current stay chunk")
        else:
            part_records, part_seen, part_hits, sql_fragments = _extract_chunk(
                client, chunk, mapping
            )
            payload = _part_payload(part_records, part_seen, part_hits, sql_fragments)
            _atomic_pickle(part_path, payload)
        records.extend(part_records)
        for name in TIME_SERIES_VARIABLES:
            seen[name] += int(payload.get("conversion_seen", {}).get(name, 0))
            hits[name] += int(payload.get("conversion_hits", {}).get(name, 0))
        if payload.get("sql_sha256"):
            sql_hashes.append(str(payload["sql_sha256"]))
        _atomic_json(
            work_dir / "progress.json",
            {"completed_chunks": chunk_index + 1, "total_chunks": n_chunks},
        )
        print(
            f"chunk {chunk_index + 1}/{n_chunks} "
            f"stays={len(chunk)} events={sum(len(record['events']) for record in part_records)}",
            flush=True,
        )
    records.sort(key=lambda record: int(record["record_id"]))
    if {int(record["record_id"]) for record in records} != set(stay_by_id):
        raise AssertionError("chunk assembly lost or duplicated stay records")
    query_hash = hashlib.sha256("\n".join(sorted(sql_hashes)).encode("utf-8")).hexdigest()
    return records, seen, hits, query_hash


def _event_counts(examples: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {name: 0 for name in TIME_SERIES_VARIABLES}
    for example in examples:
        for container in (example["obs"], example["target"]):
            for variable_id in np.asarray(container["var"], dtype=np.int64).tolist():
                counts[TIME_SERIES_VARIABLES[int(variable_id)]] += 1
    return counts


def _staleness_summary(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_variable: dict[str, list[float]] = {name: [] for name in TIME_SERIES_VARIABLES}
    overall: list[float] = []
    for example in examples:
        target = example["target"]
        variables = np.asarray(target["var"], dtype=np.int64)
        times = np.asarray(target["time"], dtype=float)
        for variable_id, timestamp in zip(variables.tolist(), times.tolist()):
            staleness = float(timestamp) - DEFAULT_T_OBS
            if staleness < 0.0:
                raise AssertionError("target event precedes t_obs")
            name = TIME_SERIES_VARIABLES[int(variable_id)]
            by_variable[name].append(staleness)
            overall.append(staleness)

    def summary(values: Sequence[float]) -> dict[str, Any]:
        array = np.asarray(values, dtype=float)
        if not len(array):
            return {"n": 0, "p25": None, "median": None, "p75": None, "p99": None}
        return {
            "n": int(len(array)),
            "p25": float(np.quantile(array, 0.25)),
            "median": float(np.median(array)),
            "p75": float(np.quantile(array, 0.75)),
            "p99": float(np.quantile(array, 0.99)),
        }

    return {
        "definition": "target_event_time_minus_t_obs_hours",
        "overall": summary(overall),
        "by_variable": {name: summary(values) for name, values in by_variable.items()},
    }


def _write_cache_and_split(
    examples: Sequence[dict[str, Any]],
    standardizer: EventStandardizer,
    selected: Sequence[Stay],
    split_indices: Mapping[str, Sequence[int]],
    *,
    processed_dir: Path,
    cache_name: str,
    split_name: str,
) -> tuple[Path, Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_path = processed_dir / cache_name
    split_path = processed_dir / split_name
    cache_payload = {
        "examples": list(examples),
        "standardizer": standardizer.as_dict(),
        "variable_names": list(standardizer.variable_names),
        "sets": ["mimiciv"],
        "t_obs": DEFAULT_T_OBS,
        "split_seed": DEFAULT_SPLIT_SEED,
    }
    _atomic_pickle(cache_path, cache_payload)
    labels = [int(example["label"]) for example in examples]
    split_audit = _group_split_audit(
        [example["subject_id"] for example in examples], labels, split_indices
    )
    split_payload = {
        "split_seed": DEFAULT_SPLIT_SEED,
        "t_obs": DEFAULT_T_OBS,
        "sets": ["mimiciv"],
        "n_samples": len(examples),
        "split_unit": split_audit["split_unit"],
        "group_column": split_audit["group_column"],
        "n_groups": split_audit["n_groups"],
        "n_groups_by_split": split_audit["n_groups_by_split"],
        "group_overlap": split_audit["group_overlap"],
        "train": [int(index) for index in split_indices["train"]],
        "val": [int(index) for index in split_indices["val"]],
        "test": [int(index) for index in split_indices["test"]],
        "sample_ids": [str(example["sample_id"]) for example in examples],
        "label_counts": split_audit["label_counts"],
    }
    _atomic_json(split_path, split_payload)
    return cache_path, split_path


def _sanity_quantiles(cache_path: Path) -> dict[str, dict[str, Any]]:
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    values: dict[str, list[float]] = {name: [] for name in TIME_SERIES_VARIABLES}
    for example in payload["examples"]:
        for container in (example["obs"], example["target"]):
            variables = np.asarray(container["var"], dtype=np.int64)
            raw = np.asarray(container["value_raw"], dtype=float)
            for variable_id, value in zip(variables.tolist(), raw.tolist()):
                values[TIME_SERIES_VARIABLES[int(variable_id)]].append(float(value))
    result: dict[str, dict[str, Any]] = {}
    for name, items in values.items():
        array = np.asarray(items, dtype=float)
        result[name] = {
            "n": int(len(array)),
            "p1": float(np.quantile(array, 0.01)) if len(array) else None,
            "p50": float(np.quantile(array, 0.50)) if len(array) else None,
            "p99": float(np.quantile(array, 0.99)) if len(array) else None,
            "min": float(array.min()) if len(array) else None,
            "max": float(array.max()) if len(array) else None,
            "n_unique": int(len(np.unique(array))) if len(array) else 0,
        }
    return result


def _load_reference_ranges(path: Path) -> dict[str, dict[str, Any]]:
    """Load and validate the measured P12 reference table used by T3."""

    required = {"var_id", "name", "n", "p1", "p50", "p99", "min", "max", "n_unique"}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"reference table {path} is missing columns "
                f"{sorted(required - set(reader.fieldnames or ()))}"
            )
        for raw in reader:
            name = str(raw["name"])
            var_id = int(raw["var_id"])
            if name in rows:
                raise ValueError(f"duplicate variable in reference table: {name}")
            rows[name] = {
                "var_id": var_id,
                "n": int(raw["n"]),
                "p1": None if raw["p1"] in {"", "nan", "NaN"} else float(raw["p1"]),
                "p50": None if raw["p50"] in {"", "nan", "NaN"} else float(raw["p50"]),
                "p99": None if raw["p99"] in {"", "nan", "NaN"} else float(raw["p99"]),
                "min": None if raw["min"] in {"", "nan", "NaN"} else float(raw["min"]),
                "max": None if raw["max"] in {"", "nan", "NaN"} else float(raw["max"]),
                "n_unique": int(raw["n_unique"]),
            }
    if tuple(rows) != TIME_SERIES_VARIABLES:
        raise ValueError("reference table variables must exactly match the P12 order")
    for expected_id, name in enumerate(TIME_SERIES_VARIABLES):
        if rows[name]["var_id"] != expected_id:
            raise ValueError(f"reference table var_id mismatch for {name}")
    return rows


def write_sanity_check(
    mimic_cache: Path,
    reference_ranges: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Write the T3 p1/p50/p99 comparison used for manual dimensional review."""

    mimic = _sanity_quantiles(mimic_cache)
    reference = _load_reference_ranges(reference_ranges)
    broad_ranges = {
        "GCS": (0.0, 20.0),
        "FiO2": (0.0, 1.1),
        "pH": (5.0, 9.0),
        "SaO2": (0.0, 110.0),
        "Temp": (20.0, 45.0),
        "MechVent": (-0.1, 1.1),
        "Weight": (1.0, 500.0),
    }
    structural_candidates = {"Cholesterol", "MechVent", "TroponinI"}
    rows: list[dict[str, Any]] = []
    auto_pass = True
    for name in TIME_SERIES_VARIABLES:
        values = mimic[name]
        low, high = broad_ranges.get(name, (-1e12, 1e12))
        missing_allowed = name in structural_candidates and values["n"] == 0
        range_ok = bool(
            values["p1"] is not None
            and values["p99"] is not None
            and values["p1"] >= low
            and values["p99"] <= high
        )
        dimension_ok = bool((values["n"] > 0 and range_ok) or missing_allowed)
        unit_flags: list[str] = []
        if name == "FiO2":
            p12_median = reference[name]["p50"]
            mimic_median = values["p50"]
            if (
                p12_median is not None
                and mimic_median is not None
                and ((p12_median < 2.0 < mimic_median) or (mimic_median < 2.0 < p12_median))
            ):
                unit_flags.append(
                    "P12 and MIMIC-IV are on opposite FiO2 scales (fraction vs percent)"
                )
        if name == "MechVent" and values["n"]:
            if values["min"] != 1.0 or values["max"] != 1.0:
                unit_flags.append("MechVent must contain only interval-start value 1 events")
        auto_pass = auto_pass and dimension_ok and not unit_flags
        rows.append(
            {
                "variable": name,
                "mimiciv_n": values["n"],
                "mimiciv_p1": values["p1"],
                "mimiciv_p50": values["p50"],
                "mimiciv_p99": values["p99"],
                "mimiciv_min": values["min"],
                "mimiciv_max": values["max"],
                "mimiciv_n_unique": values["n_unique"],
                "p12_n": reference[name]["n"],
                "p12_p1": reference[name]["p1"],
                "p12_p50": reference[name]["p50"],
                "p12_p99": reference[name]["p99"],
                "p12_min": reference[name]["min"],
                "p12_max": reference[name]["max"],
                "p12_n_unique": reference[name]["n_unique"],
                "automatic_dimension_check": dimension_ok,
                "unit_flags": unit_flags,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MIMIC-IV T3 sanity check",
        "",
        f"Automatic dimensional-range check: `{auto_pass}`.",
        "",
        f"Reference: `{reference_ranges}` (measured P12 raw-value ranges).",
        "Values are in the canonical P12 units after only the registered "
        "conversions in the MIMIC-IV variable map.",
        "",
        "| variable | MIMIC n | MIMIC p1 | MIMIC p50 | MIMIC p99 | P12 n | P12 p1 | P12 p50 | P12 p99 | range check | unit flags |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        def fmt(value: object) -> str:
            return "n/a" if value is None else f"{float(value):.6g}"

        lines.append(
            f"| {row['variable']} | {row['mimiciv_n']} | {fmt(row['mimiciv_p1'])} | "
            f"{fmt(row['mimiciv_p50'])} | {fmt(row['mimiciv_p99'])} | {row['p12_n']} | "
            f"{fmt(row['p12_p1'])} | {fmt(row['p12_p50'])} | {fmt(row['p12_p99'])} | "
            f"{row['automatic_dimension_check']} | {'; '.join(row['unit_flags']) or '—'} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"automatic_pass": auto_pass, "rows": rows}


def _update_map_status(
    map_path: Path,
    mapping: Mapping[str, Any],
    train_counts: Mapping[str, int],
) -> None:
    payload = json.loads(json.dumps(mapping))
    structural = set(payload.get("structural_candidates", ()))
    for name, spec in payload["variables"].items():
        count = int(train_counts.get(name, 0))
        spec["verified"] = True
        spec["train_event_count"] = count
        spec["structurally_missing"] = bool(name in structural and count < STRUCTURAL_THRESHOLD)
        if name not in structural and count == 0:
            raise ValueError(f"required variable {name} has no training events")
    temporary = map_path.with_name(map_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    os.replace(temporary, map_path)


def _manifest(
    *,
    mapping: Mapping[str, Any],
    schema: Mapping[str, Any],
    selected: Sequence[Stay],
    queue_counts: Mapping[str, int],
    sample_seed: int,
    split_indices: Mapping[str, Sequence[int]],
    examples: Sequence[Mapping[str, Any]],
    conversion_seen: Mapping[str, int],
    conversion_hits: Mapping[str, int],
    query_sql_sha256: str,
    standardizer: EventStandardizer,
    map_path: Path,
    database_url: str,
    careunit_map: Mapping[str, int],
) -> dict[str, Any]:
    total_counts = _event_counts(examples)
    train_examples = [examples[index] for index in split_indices["train"]]
    train_counts = _event_counts(train_examples)
    structural_candidates = set(mapping.get("structural_candidates", ()))
    structural_missing = [
        {
            "variable": name,
            "train_event_count": int(train_counts[name]),
            "threshold": STRUCTURAL_THRESHOLD,
            "reason": "training event count below structural-missing threshold",
        }
        for name in structural_candidates
        if train_counts[name] < STRUCTURAL_THRESHOLD
    ]
    conversion_rates = {
        name: {
            "checked": int(conversion_seen[name]),
            "converted": int(conversion_hits[name]),
            "rate": (
                float(conversion_hits[name] / conversion_seen[name])
                if conversion_seen[name]
                else 0.0
            ),
        }
        for name in TIME_SERIES_VARIABLES
    }
    labels = [int(example["label"]) for example in examples]
    split_audit = _group_split_audit(
        [example["subject_id"] for example in examples], labels, split_indices
    )
    return {
        "dataset": "mimiciv",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "database_url": database_url.split("@")[-1],
        "map_path": str(map_path),
        "mapping_fingerprint": _map_fingerprint(mapping),
        "query_sql_sha256": query_sql_sha256,
        "schema_check": schema,
        "queue": {
            **{str(key): int(value) for key, value in queue_counts.items()},
            "sample_size": len(selected),
            "sample_seed": int(sample_seed),
            "sampling": "simple_random_without_replacement",
            "selected_stay_ids_sha256": hashlib.sha256(
                ",".join(str(stay.stay_id) for stay in selected).encode("utf-8")
            ).hexdigest(),
        },
        "split": {
            "split_seed": DEFAULT_SPLIT_SEED,
            "fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
            **split_audit,
        },
        "icu_type_mapping": {
            str(name): int(value) for name, value in sorted(careunit_map.items())
        },
        "event_counts": {
            "all_windows": total_counts,
            "train_split": train_counts,
        },
        "unit_conversion": conversion_rates,
        "structurally_missing": structural_missing,
        "staleness_hours": _staleness_summary(examples),
        "standardizer": {
            "fit_split": "train",
            "variable_count": standardizer.n_variables,
            "clipping_ratio": standardizer.clipping_ratio.tolist(),
        },
        "examples": len(examples),
        "variable_names": list(TIME_SERIES_VARIABLES),
    }


def extract(
    *,
    map_path: str | Path = DEFAULT_MAP,
    database_url: str = DEFAULT_DATABASE_URL,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    split_seed: int = DEFAULT_SPLIT_SEED,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    cache_name: str = "mimiciv_events.pkl",
    split_name: str = "mimiciv_split_indices_12k.json",
    manifest_name: str = "mimiciv_manifest.json",
    work_dir: str | Path | None = None,
    reference_ranges: str | Path = DEFAULT_REFERENCE_RANGES,
    update_map: bool = False,
    keep_work: bool = False,
) -> dict[str, Any]:
    if int(split_seed) != DEFAULT_SPLIT_SEED:
        raise ValueError("the MIMIC-IV split_seed is preregistered as 42")
    if float(DEFAULT_T_OBS) != 24.0:
        raise AssertionError("unexpected t_obs constant")
    map_path = Path(map_path)
    processed_dir = Path(processed_dir)
    results_dir = Path(results_dir)
    mapping = _load_map(map_path)
    client = DatabaseClient(database_url)
    work_path = Path(work_dir) if work_dir is not None else processed_dir / f"mimiciv_extract_work_{int(sample_size)}"
    try:
        schema = schema_check(client, mapping)
        if not schema["verified"]:
            raise RuntimeError(f"T0 schema check failed: {schema['missing']}")
        stays, queue_counts, careunit_map = _fetch_queue(client)
        queue_counts = dict(queue_counts)
        queue_counts["eligible_total_before_sample"] = len(stays)
        selected = _sample_stays(stays, sample_size=sample_size, seed=sample_seed)
        records, conversion_seen, conversion_hits, query_hash = _load_or_extract_parts(
            client,
            selected,
            mapping,
            work_dir=work_path,
            chunk_size=chunk_size,
        )
    finally:
        client.close()

    labels = [int(record["label"]) for record in records]
    split_indices = stratified_group_split(
        labels,
        [record["subject_id"] for record in records],
        seed=split_seed,
    )
    train_records = [records[index] for index in split_indices["train"]]
    standardizer = EventStandardizer.fit(train_records)
    examples = [make_event_example(record, standardizer, t_obs=DEFAULT_T_OBS) for record in records]
    # ``make_event_example`` intentionally keeps the model-facing payload
    # compact.  Identity fields are copied back here for patient-level audit
    # and future grouped splitting; they are never passed as model features.
    for example, record in zip(examples, records):
        example["subject_id"] = int(record["subject_id"])
        example["hadm_id"] = int(record["hadm_id"])
    if len(examples) != int(sample_size):
        raise AssertionError("cache example count does not equal requested sample size")
    cache_path, split_path = _write_cache_and_split(
        examples,
        standardizer,
        selected,
        split_indices,
        processed_dir=processed_dir,
        cache_name=cache_name,
        split_name=split_name,
    )
    train_counts = _event_counts([examples[index] for index in split_indices["train"]])
    if update_map:
        _update_map_status(map_path, mapping, train_counts)
        mapping = _load_map(map_path)
    manifest = _manifest(
        mapping=mapping,
        schema=schema,
        selected=selected,
        queue_counts=queue_counts,
        sample_seed=sample_seed,
        split_indices=split_indices,
        examples=examples,
        conversion_seen=conversion_seen,
        conversion_hits=conversion_hits,
        query_sql_sha256=query_hash,
        standardizer=standardizer,
        map_path=map_path,
        database_url=database_url,
        careunit_map=careunit_map,
    )
    manifest_path = results_dir / manifest_name
    _atomic_json(manifest_path, manifest)
    sanity = None
    reference_ranges_path = Path(reference_ranges)
    if reference_ranges_path.exists():
        sanity = write_sanity_check(
            cache_path,
            reference_ranges_path,
            results_dir / "sanity_check.md",
        )
        manifest["sanity_check"] = {
            "path": str(results_dir / "sanity_check.md"),
            "reference_ranges": str(reference_ranges_path),
            "automatic_dimension_check": sanity["automatic_pass"],
        }
        _atomic_json(manifest_path, manifest)
    if not keep_work:
        shutil.rmtree(work_path, ignore_errors=True)
    return {
        "cache_path": str(cache_path),
        "split_path": str(split_path),
        "manifest_path": str(manifest_path),
        "n_samples": len(examples),
        "split_sizes": {name: len(split_indices[name]) for name in split_indices},
        "structurally_missing": manifest["structurally_missing"],
        "sanity_check": sanity["automatic_pass"] if sanity else None,
        "icu_type_mapping": careunit_map,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", dest="map_path", default=str(DEFAULT_MAP))
    parser.add_argument("--database-url", default=os.environ.get("MIMICIV_DATABASE_URL", DEFAULT_DATABASE_URL))
    parser.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--cache-name", default="mimiciv_events.pkl")
    parser.add_argument("--split-name", default="mimiciv_split_indices_12k.json")
    parser.add_argument("--manifest-name", default="mimiciv_manifest.json")
    parser.add_argument("--work-dir")
    parser.add_argument("--reference-ranges", default=str(DEFAULT_REFERENCE_RANGES))
    parser.add_argument("--update-map", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--schema-check", action="store_true", help="run T0 and print the live schema assertion")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    mapping = _load_map(args.map_path)
    if args.schema_check:
        client = DatabaseClient(args.database_url)
        try:
            report = schema_check(client, mapping)
        finally:
            client.close()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["verified"]:
            raise SystemExit(1)
        return
    result = extract(
        map_path=args.map_path,
        database_url=args.database_url,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        split_seed=args.split_seed,
        chunk_size=args.chunk_size,
        cache_name=args.cache_name,
        split_name=args.split_name,
        manifest_name=args.manifest_name,
        work_dir=args.work_dir,
        reference_ranges=args.reference_ranges,
        update_map=args.update_map,
        keep_work=args.keep_work,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
