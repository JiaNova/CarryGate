"""Extract the frozen 37-variable eICU 48-hour forecasting cache.

The extractor is deliberately independent of model and training code.  It
uses the live ``eicu_crd`` schema, keeps the P12 variable IDs unchanged, and
normalizes values with the already-fitted MIMIC-IV train standardizer.  A
selected stay chunk is written to an atomically renamed pickle, so a dropped
database connection can be resumed without accepting a partial chunk.  The
work directory is removed after successful finalization.
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
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .p12_events import (
    TIME_SERIES_VARIABLES,
    EventStandardizer,
    annotate_events,
    make_event_example,
    stratified_event_split,
)

try:  # psycopg is installed in the specified icu_kan environment.
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - exercised outside the target env
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP = ROOT / "configs/eicu_variable_map.yaml"
DEFAULT_PROCESSED_DIR = ROOT / "data/processed"
DEFAULT_RESULTS_DIR = ROOT / "results/phase3_eicu"
DEFAULT_MIMIC_CACHE = DEFAULT_PROCESSED_DIR / "mimiciv_events.pkl"
DEFAULT_REFERENCE_RANGES = ROOT / "results/mimiciv/p12_reference_ranges.csv"
DEFAULT_SAMPLE_SIZE = 12_000
DEFAULT_SAMPLE_SEED = 42
DEFAULT_SPLIT_SEED = 42
DEFAULT_T_OBS = 24.0
DEFAULT_HORIZON = 48.0
DEFAULT_CHUNK_SIZE = 500
DEFAULT_QUERY_TIMEOUT_MS = 300_000
STRUCTURAL_THRESHOLD = 100
IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")
EXCLUDED_LABNAMES = frozenset(
    {"urinary creatinine", "glucose - CSF", "bedside glucose"}
)
PIVOTED_TABLES = (
    "pivoted_bg",
    "pivoted_infusion",
    "pivoted_lab",
    "pivoted_med",
    "pivoted_o2",
    "pivoted_score",
    "pivoted_treatment_vasopressor",
    "pivoted_uo",
    "pivoted_vital",
    "pivoted_vital_other",
)


@dataclass(frozen=True)
class Stay:
    """The cohort fields needed by extraction, stratification, and LOHO."""

    stay_id: int
    subject_id: int
    hadm_id: int
    hospital_id: int
    uniquepid: str
    gender: str
    anchor_age: int
    label: int
    outtime: float
    admission_weight: float | None


class DatabaseClient:
    """Small reconnecting client for parameterized chunk extraction queries."""

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

    def fetchall(
        self, query: str, params: Sequence[object] = ()
    ) -> list[dict[str, Any]]:
        """Execute one query and reconnect only after a database error."""

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
            f"eICU query failed after {self.retries + 1} attempts"
        ) from last_error


def _qident(value: object) -> str:
    """Quote a map-controlled SQL identifier after a strict allow-list check."""

    text = str(value)
    if not IDENTIFIER.fullmatch(text):
        raise ValueError(f"unsafe SQL identifier in eICU map: {text!r}")
    return f'"{text}"'


def _qualified(schema: str, table: str) -> str:
    return f"{_qident(schema)}.{_qident(table)}"


def _load_map(path: str | Path = DEFAULT_MAP) -> dict[str, Any]:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = tuple(str(name) for name in payload.get("variable_names", ()))
    if names != TIME_SERIES_VARIABLES:
        raise ValueError("eICU variable_names must exactly match the P12 order")
    variables = payload.get("variables")
    if not isinstance(variables, dict) or tuple(variables) != names:
        raise ValueError("eICU map must contain exactly one entry per P12 variable")
    if set(payload.get("excluded_variable_ids", ())) != {6, 19, 32}:
        raise ValueError("eICU exclusion set must remain (6, 19, 32)")
    for expected_id, name in enumerate(names):
        spec = variables[name]
        if not isinstance(spec, dict):
            raise ValueError(f"invalid mapping for {name}")
        if int(spec.get("var_id", -1)) != expected_id:
            raise ValueError(f"var_id mismatch for {name}")
        if spec.get("verified") is not True:
            raise ValueError(f"mapping {name} is not verified; T-E1 must finish first")
        source_schema = str(spec.get("source_schema", ""))
        source_table = str(spec.get("source_table", ""))
        _qident(source_schema)
        _qident(source_table)
        if str(spec.get("join_key", "")) != "patientunitstayid":
            raise ValueError(f"invalid join key for {name}")
        for field in (
            "source_column",
            "offset_column",
            "start_column",
            "end_column",
            "status_column",
        ):
            if spec.get(field):
                _qident(spec[field])
        if str(spec.get("source_kind")) == "lab_exact" and not spec.get("labname"):
            raise ValueError(f"lab_exact mapping {name} has no exact labname")
    return payload


def _map_fingerprint(mapping: Mapping[str, Any]) -> str:
    canonical = json.dumps(mapping, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _required_columns(mapping: Mapping[str, Any]) -> dict[tuple[str, str], set[str]]:
    required: dict[tuple[str, str], set[str]] = defaultdict(set)
    for spec in mapping["variables"].values():
        key = (str(spec["source_schema"]), str(spec["source_table"]))
        required[key].add(str(spec["join_key"]))
        for field in (
            "source_column",
            "offset_column",
            "start_column",
            "end_column",
            "status_column",
        ):
            if spec.get(field):
                required[key].add(str(spec[field]))
        if str(spec.get("source_kind")) == "lab_exact":
            required[key].update({"labid", "labname", "labresult", "labresultoffset"})
    required[(str(mapping.get("schema", "eicu_crd")), "patient")].update(
        {
            "patientunitstayid",
            "uniquepid",
            "patienthealthsystemstayid",
            "hospitalid",
            "gender",
            "age",
            "admissionweight",
            "unitvisitnumber",
            "unitdischargeoffset",
            "unitdischargestatus",
        }
    )
    return required


def schema_check(
    client: DatabaseClient,
    mapping: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify mapped tables, columns, exact lab names, and airway statuses."""

    schema = str(mapping.get("schema", "eicu_crd"))
    rows = client.fetchall(
        """
        SELECT table_schema, table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = %s
        """,
        (schema,),
    )
    actual = {
        (str(row["table_schema"]), str(row["table_name"]), str(row["column_name"]))
        for row in rows
    }
    missing: list[str] = []
    table_report: dict[str, Any] = {}
    for (table_schema, table), columns in sorted(_required_columns(mapping).items()):
        missing_columns = sorted(
            column
            for column in columns
            if (table_schema, table, column) not in actual
        )
        key = f"{table_schema}.{table}"
        table_report[key] = {
            "required_columns": sorted(columns),
            "missing_columns": missing_columns,
            "verified": not missing_columns,
        }
        missing.extend(f"{key}.{column}" for column in missing_columns)

    lab_specs = [
        (name, spec)
        for name, spec in mapping["variables"].items()
        if str(spec.get("source_kind")) == "lab_exact"
    ]
    lab_report: dict[str, Any] = {}
    lab_table = next((str(spec["source_table"]) for _, spec in lab_specs), "lab")
    if lab_specs:
        lab_names = [str(spec["labname"]) for _, spec in lab_specs]
        lab_rows = client.fetchall(
            f"""
            SELECT labname, count(*) AS rows
            FROM {_qualified(schema, lab_table)}
            WHERE labname = ANY(%s)
            GROUP BY labname
            ORDER BY labname
            """,
            (lab_names,),
        )
        found = {str(row["labname"]): int(row["rows"]) for row in lab_rows}
        for name, spec in lab_specs:
            labname = str(spec["labname"])
            verified = labname in found
            lab_report[name] = {
                "labname": labname,
                "rows": found.get(labname, 0),
                "exact_equality": True,
                "verified": verified,
            }
            if not verified:
                missing.append(f"{schema}.{lab_table}.labname={labname!r}")
        trap_rows = client.fetchall(
            f"""
            SELECT labname, count(*) AS rows
            FROM {_qualified(schema, lab_table)}
            WHERE labname = ANY(%s)
            GROUP BY labname
            ORDER BY labname
            """,
            (sorted(EXCLUDED_LABNAMES),),
        )
        trap_report = {
            str(row["labname"]): int(row["rows"]) for row in trap_rows
        }
    else:
        trap_report = {}

    interval_specs = [
        spec
        for spec in mapping["variables"].values()
        if str(spec.get("source_kind")) == "interval_binary"
    ]
    status_report: dict[str, Any] = {}
    for spec in interval_specs:
        table = str(spec["source_table"])
        column = _qident(spec["status_column"])
        expected = [str(value) for value in spec.get("status_values", ())]
        status_rows = client.fetchall(
            f"""
            SELECT {column} AS status, count(*) AS rows
            FROM {_qualified(schema, table)}
            WHERE {column} = ANY(%s)
            GROUP BY {column}
            ORDER BY {column}
            """,
            (expected,),
        )
        found = {str(row["status"]): int(row["rows"]) for row in status_rows}
        status_report[f"{table}.{spec['status_column']}"] = {
            "expected_values": expected,
            "rows_by_value": found,
            "verified": all(value in found for value in expected),
        }
        if not all(value in found for value in expected):
            missing.append(f"{schema}.{table}.{spec['status_column']} status value")

    pivoted_bg_columns = {
        column
        for table_schema, table, column in actual
        if table_schema == schema and table == "pivoted_bg"
    }
    blood_gas_report = {
        "source": "pivoted_bg",
        "specimen_column_present": "specimen" in pivoted_bg_columns,
        "verified_derivative_source": "specimen" not in pivoted_bg_columns,
        "note": (
            "Live pivoted_bg has no specimen column; gas fields are consumed from "
            "the registered derivative table without an unverifiable post-hoc filter."
        ),
    }
    mapping_verified = not missing and all(
        bool(spec.get("verified") is True) for spec in mapping["variables"].values()
    )
    return {
        "verified": bool(mapping_verified),
        "missing": missing,
        "tables": table_report,
        "lab_exact": lab_report,
        "excluded_labname_audit": {
            "names": sorted(EXCLUDED_LABNAMES),
            "rows_by_name": trap_report,
            "extraction_rule": "exact labname equality; none are selected",
        },
        "interval_status": status_report,
        "blood_gas": blood_gas_report,
        "mapping_fingerprint": _map_fingerprint(mapping),
    }


def _age_expression(alias: str = "p") -> str:
    return (
        f"CASE WHEN trim({alias}.age) = '> 89' THEN 90 "
        f"WHEN trim({alias}.age) ~ '^[0-9]+$' THEN trim({alias}.age)::int "
        "ELSE NULL END"
    )


def _fetch_cohort(
    client: DatabaseClient,
    mapping: Mapping[str, Any],
) -> tuple[list[Stay], dict[str, Any], dict[str, int]]:
    schema = str(mapping.get("schema", "eicu_crd"))
    cohort_cfg = mapping["cohort"]
    patient = _qualified(schema, "patient")
    query = f"""
        SELECT p.patientunitstayid,
               p.uniquepid,
               p.patienthealthsystemstayid,
               p.hospitalid,
               p.gender,
               p.age,
               p.admissionweight,
               p.unitdischargeoffset,
               p.unitdischargestatus,
               p.unitvisitnumber
        FROM {patient} AS p
        WHERE p.unitvisitnumber = %s
          AND p.unitdischargeoffset >= %s
          AND ({_age_expression()}) >= %s
          AND p.unitdischargestatus = ANY(%s)
        ORDER BY p.patientunitstayid
    """
    rows = client.fetchall(
        query,
        (
            int(cohort_cfg["first_unit_stay_value"]),
            int(cohort_cfg["min_unit_discharge_offset_minutes"]),
            int(cohort_cfg["min_age_years"]),
            [str(cohort_cfg["alive_label"]), str(cohort_cfg["expired_label"])],
        ),
    )
    if not rows:
        raise ValueError("eICU eligible cohort is empty")
    unique_ids = sorted({str(row["uniquepid"]) for row in rows})
    subject_map = {value: index + 1 for index, value in enumerate(unique_ids)}
    stays: list[Stay] = []
    for row in rows:
        stay_id = int(row["patientunitstayid"])
        if row["hospitalid"] is None:
            raise ValueError(f"eligible stay {stay_id} has NULL hospitalid")
        age_value = row["age"]
        age_text = "" if age_value is None else str(age_value).strip()
        age = 90 if age_text == "> 89" else int(age_text)
        status = str(row["unitdischargestatus"])
        if status not in {
            str(cohort_cfg["alive_label"]),
            str(cohort_cfg["expired_label"]),
        }:
            raise ValueError(f"non-binary discharge status for stay {stay_id}")
        weight = None if row["admissionweight"] is None else float(row["admissionweight"])
        if weight is not None and not math.isfinite(weight):
            weight = None
        gender = "" if row["gender"] is None else str(row["gender"]).strip()
        stays.append(
            Stay(
                stay_id=stay_id,
                subject_id=int(subject_map[str(row["uniquepid"])]),
                hadm_id=int(row["patienthealthsystemstayid"]),
                hospital_id=int(row["hospitalid"]),
                uniquepid=str(row["uniquepid"]),
                gender=gender,
                anchor_age=age,
                label=int(status == str(cohort_cfg["expired_label"])),
                outtime=float(row["unitdischargeoffset"]) / 60.0,
                admission_weight=weight,
            )
        )
    counts = {
        "eligible_first_unit_age_los_label": len(stays),
        "eligible_subjects": len(subject_map),
        "eligible_hospitals": len({stay.hospital_id for stay in stays}),
        "expired": sum(stay.label for stay in stays),
        "alive": sum(1 - stay.label for stay in stays),
        "age_gt_89_mapped_to_90": sum(stay.anchor_age == 90 for stay in stays),
        "admission_weight_nonnull": sum(stay.admission_weight is not None for stay in stays),
    }
    metadata = {
        "contract": "unitvisitnumber=1, unitdischargeoffset>=2880, age>=16, binary status",
        "status_labels": {
            "alive": str(cohort_cfg["alive_label"]),
            "expired": str(cohort_cfg["expired_label"]),
        },
        "counts": counts,
    }
    return stays, metadata, subject_map


def _fetch_long_stay_hospital_counts(
    client: DatabaseClient,
    mapping: Mapping[str, Any],
) -> dict[int, int]:
    """Count hospitals using the preregistered raw long-stay T0 criterion.

    The 145-hospital LOHO frame is defined by all patient rows with
    ``unitdischargeoffset >= 2880`` (the same fact audited in E-T0).  The
    forecasting cache still applies the stricter first-unit/age/status cohort
    contract; those eligible stays are filtered after the hospital frame is
    fixed.
    """

    schema = str(mapping.get("schema", "eicu_crd"))
    cohort_cfg = mapping["cohort"]
    rows = client.fetchall(
        f"""
        SELECT hospitalid, count(*) AS n_long_stays
        FROM {_qualified(schema, 'patient')}
        WHERE hospitalid IS NOT NULL
          AND unitdischargeoffset >= %s
        GROUP BY hospitalid
        ORDER BY hospitalid
        """,
        (int(cohort_cfg["min_unit_discharge_offset_minutes"]),),
    )
    return {
        int(row["hospitalid"]): int(row["n_long_stays"])
        for row in rows
    }


def _stratified_sample(
    stays: Sequence[Stay], *, sample_size: int, seed: int
) -> list[Stay]:
    if sample_size < 1 or sample_size > len(stays):
        raise ValueError(f"sample_size must be between 1 and {len(stays)}")
    labels = np.asarray([stay.label for stay in stays], dtype=np.int64)
    unique, counts = np.unique(labels, return_counts=True)
    quotas = {int(label): int(np.floor(sample_size * count / len(stays))) for label, count in zip(unique, counts)}
    remaining = int(sample_size - sum(quotas.values()))
    fractional = sorted(
        (
            sample_size * int(count) / len(stays) - quotas[int(label)],
            int(label),
        )
        for label, count in zip(unique, counts)
    )
    for _, label in sorted(fractional, reverse=True)[:remaining]:
        quotas[label] += 1
    rng = np.random.default_rng(int(seed))
    selected_indices: list[int] = []
    for label in sorted(quotas):
        candidates = np.flatnonzero(labels == label)
        selected_indices.extend(
            int(value)
            for value in rng.choice(candidates, size=quotas[label], replace=False)
        )
    selected = [stays[index] for index in sorted(selected_indices)]
    if len(selected) != sample_size or len({stay.stay_id for stay in selected}) != sample_size:
        raise AssertionError("stratified eICU sampling lost or duplicated stays")
    return sorted(selected, key=lambda stay: stay.stay_id)


def _values_cte(stays: Sequence[Stay]) -> tuple[str, list[object]]:
    if not stays:
        raise ValueError("cannot build a query for an empty stay chunk")
    placeholders = ",".join("(%s)" for _ in stays)
    return (
        f"WITH selected(stay_id) AS (VALUES {placeholders})",
        [int(stay.stay_id) for stay in stays],
    )


def _numeric_query(
    stays: Sequence[Stay],
    specs: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    horizon_hours: float = DEFAULT_HORIZON,
) -> tuple[str, list[object], dict[str, str]]:
    if not math.isfinite(float(horizon_hours)) or float(horizon_hours) <= 0:
        raise ValueError("horizon_hours must be finite and positive")
    cte, params = _values_cte(stays)
    params.append(float(horizon_hours) * 60.0)
    first = specs[0][1]
    schema = str(first["source_schema"])
    table = str(first["source_table"])
    offset = _qident(first["offset_column"])
    select_columns = ["s.stay_id", f"e.{offset} AS event_offset"]
    aliases: dict[str, str] = {}
    for name, spec in specs:
        alias = f"v{int(spec['var_id'])}"
        aliases[alias] = name
        select_columns.append(f"e.{_qident(spec['source_column'])} AS {alias}")
    query = (
        f"{cte} SELECT {', '.join(select_columns)} "
        f"FROM selected AS s JOIN {_qualified(schema, table)} AS e "
        f"ON e.{_qident(first['join_key'])} = s.stay_id "
        f"WHERE e.{offset} >= 0 AND e.{offset} < %s "
        "ORDER BY s.stay_id, event_offset"
    )
    return query, params, aliases


def _lab_query(
    stays: Sequence[Stay],
    spec: Mapping[str, Any],
    labnames: Sequence[str],
    *,
    horizon_hours: float = DEFAULT_HORIZON,
) -> tuple[str, list[object]]:
    if not math.isfinite(float(horizon_hours)) or float(horizon_hours) <= 0:
        raise ValueError("horizon_hours must be finite and positive")
    cte, params = _values_cte(stays)
    params.append(list(labnames))
    params.append(float(horizon_hours) * 60.0)
    schema = str(spec["source_schema"])
    table = str(spec["source_table"])
    query = (
        f"{cte} SELECT s.stay_id, e.labresultoffset AS event_offset, "
        "e.labid, e.labname, e.labresult AS value "
        f"FROM selected AS s JOIN {_qualified(schema, table)} AS e "
        f"ON e.{_qident(spec['join_key'])} = s.stay_id "
        "WHERE e.labname = ANY(%s) AND e.labresult IS NOT NULL "
        "AND e.labresultoffset >= 0 AND e.labresultoffset < %s "
        "ORDER BY s.stay_id, e.labresultoffset, e.labid"
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
    schema = str(spec["source_schema"])
    table = str(spec["source_table"])
    start = _qident(spec["start_column"])
    end = _qident(spec["end_column"])
    status = _qident(spec["status_column"])
    params.append([str(value) for value in spec.get("status_values", ())])
    params.append(float(horizon_hours) * 60.0)
    query = (
        f"{cte} SELECT s.stay_id, e.{start} AS start_offset, "
        f"e.{end} AS end_offset, e.{status} AS status, e.ctid::text AS row_id "
        f"FROM selected AS s JOIN {_qualified(schema, table)} AS e "
        f"ON e.{_qident(spec['join_key'])} = s.stay_id "
        f"WHERE e.{status} = ANY(%s) "
        f"AND e.{start} < %s AND (e.{end} IS NULL OR e.{end} >= 0) "
        f"ORDER BY s.stay_id, e.{start}, e.{end}"
    )
    return query, params


def _convert_value(
    name: str,
    value: object,
    spec: Mapping[str, Any],
    quality: Counter[str],
) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        quality["non_numeric_values_dropped"] += 1
        return None
    if not math.isfinite(number):
        quality["nonfinite_values_dropped"] += 1
        return None
    rule = str(spec.get("unit_rule", "identity"))
    if rule == "fraction_or_percent":
        quality["fio2_values_seen"] += 1
        if number > 1.0:
            number /= 100.0
            quality["fio2_percent_to_fraction"] += 1
    elif rule == "celsius_verified":
        quality["temperature_values_seen"] += 1
        if not 25.0 <= number <= 45.0:
            quality["temperature_out_of_range_dropped"] += 1
            return None
    elif rule == "gcs_3_to_15":
        if not 3.0 <= number <= 15.0:
            quality["gcs_out_of_range_dropped"] += 1
            return None
    elif rule in {
        "identity",
        "arterial_derivative_verified",
        "one_at_vent_start",
        "admission_single_point",
    }:
        pass
    else:
        raise ValueError(f"unknown eICU unit rule for {name}: {rule}")
    return number


def _append_event(
    record: dict[str, Any],
    name: str,
    offset_minutes: object,
    value: object,
    spec: Mapping[str, Any],
    quality: Counter[str],
    *,
    source: str,
    tie: object,
    horizon_hours: float = DEFAULT_HORIZON,
    respect_outtime: bool = False,
) -> None:
    if offset_minutes is None:
        return
    offset = float(offset_minutes)
    if not math.isfinite(offset) or offset < 0.0 or offset >= float(horizon_hours) * 60.0:
        return
    if respect_outtime and offset >= float(record["outtime"]) * 60.0:
        return
    converted = _convert_value(name, value, spec, quality)
    if converted is None:
        return
    hour = offset / 60.0
    variable_id = int(spec["var_id"])
    record["_event_rows"].append(
        {
            "sort_key": (hour, variable_id, str(source), str(tie)),
            "variable": name,
            "time": hour,
            "value": converted,
        }
    )
    quality[f"events_{name}"] += 1
    quality[f"source_{source}"] += 1


def _new_record(stay: Stay) -> dict[str, Any]:
    return {
        "record_id": stay.stay_id,
        "sample_id": f"eicu-{stay.stay_id}",
        "label": stay.label,
        "set_name": "eicu",
        "hospital_id": stay.hospital_id,
        "subject_id": stay.subject_id,
        "hadm_id": stay.hadm_id,
        "uniquepid": stay.uniquepid,
        "gender": stay.gender,
        "anchor_age": stay.anchor_age,
        "outtime": stay.outtime,
        "_event_rows": [],
    }


def _extract_chunk(
    client: DatabaseClient,
    stays: Sequence[Stay],
    mapping: Mapping[str, Any],
    *,
    horizon_hours: float = DEFAULT_HORIZON,
    respect_outtime: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int], str]:
    if not math.isfinite(float(horizon_hours)) or float(horizon_hours) <= 0:
        raise ValueError("horizon_hours must be finite and positive")
    records = {stay.stay_id: _new_record(stay) for stay in stays}
    stay_by_id = {stay.stay_id: stay for stay in stays}
    quality: Counter[str] = Counter()
    sql_fragments: list[str] = []
    variables = mapping["variables"]

    for stay in stays:
        if stay.admission_weight is not None:
            _append_event(
                records[stay.stay_id],
                "Weight",
                0.0,
                stay.admission_weight,
                variables["Weight"],
                quality,
                source="patient.admissionweight",
                tie="0",
            )
            quality["weight_rows_seen"] += 1

    grouped: dict[tuple[str, str, str, str], list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for name in TIME_SERIES_VARIABLES:
        spec = variables[name]
        if str(spec.get("source_kind")) != "numeric":
            continue
        key = (
            str(spec["source_schema"]),
            str(spec["source_table"]),
            str(spec["offset_column"]),
            str(spec["join_key"]),
        )
        grouped[key].append((name, spec))
    for _, specs in sorted(grouped.items()):
        specs.sort(key=lambda item: int(item[1]["var_id"]))
        query, params, aliases = _numeric_query(
            stays, specs, horizon_hours=horizon_hours
        )
        sql_fragments.append(query)
        for row in client.fetchall(query, params):
            stay_id = int(row["stay_id"])
            for alias, name in aliases.items():
                value = row.get(alias)
                if value is None:
                    continue
                spec = variables[name]
                _append_event(
                    records[stay_id],
                    name,
                    row["event_offset"],
                    value,
                    spec,
                    quality,
                    source=f"{spec['source_table']}.{spec['source_column']}",
                    tie=f"{row['event_offset']}",
                    horizon_hours=horizon_hours,
                    respect_outtime=respect_outtime,
                )

    lab_specs = [
        (name, variables[name])
        for name in TIME_SERIES_VARIABLES
        if str(variables[name].get("source_kind")) == "lab_exact"
    ]
    if lab_specs:
        labnames = sorted({str(spec["labname"]) for _, spec in lab_specs})
        query, params = _lab_query(
            stays, lab_specs[0][1], labnames, horizon_hours=horizon_hours
        )
        sql_fragments.append(query)
        name_by_labname = {str(spec["labname"]): name for name, spec in lab_specs}
        for row in client.fetchall(query, params):
            labname = str(row["labname"])
            if labname in EXCLUDED_LABNAMES:
                raise AssertionError(f"excluded labname reached event stream: {labname}")
            name = name_by_labname.get(labname)
            if name is None:
                raise AssertionError(f"unmapped exact labname reached event stream: {labname}")
            _append_event(
                records[int(row["stay_id"])],
                name,
                row["event_offset"],
                row["value"],
                variables[name],
                quality,
                source=f"lab.{labname}",
                tie=str(row["labid"]),
                horizon_hours=horizon_hours,
                respect_outtime=respect_outtime,
            )

    interval_specs = [
        (name, variables[name])
        for name in TIME_SERIES_VARIABLES
        if str(variables[name].get("source_kind")) == "interval_binary"
    ]
    for name, spec in interval_specs:
        query, params = _interval_query(
            stays, spec, horizon_hours=horizon_hours
        )
        sql_fragments.append(query)
        for row in client.fetchall(query, params):
            start = float(row["start_offset"])
            end = row["end_offset"]
            if end is not None and float(end) < start:
                # eICU uses zero as an incomplete/unknown end offset in some
                # respiratory-care rows.  The frozen contract is a 1-only
                # start-event indicator, so the malformed end is audited but
                # must not create an artificial zero or discard the start.
                quality["mechvent_end_before_start_audited"] += 1
            _append_event(
                records[int(row["stay_id"])],
                name,
                max(start, 0.0),
                1.0,
                spec,
                quality,
                source=f"{spec['source_table']}.{spec['status_column']}",
                tie=str(row["row_id"]),
                horizon_hours=horizon_hours,
                respect_outtime=respect_outtime,
            )

    finalized: list[dict[str, Any]] = []
    for stay in stays:
        record = records[stay.stay_id]
        event_rows = sorted(record.pop("_event_rows"), key=lambda item: item["sort_key"])
        triples = [
            (str(item["variable"]), float(item["time"]), float(item["value"]))
            for item in event_rows
        ]
        record["events"] = triples
        record["events_with_delta"] = annotate_events(triples)
        finalized.append(record)
    query_hash = hashlib.sha256(
        "\n".join(sql_fragments).encode("utf-8")
    ).hexdigest()
    return finalized, dict(quality), query_hash


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


def _load_or_extract_parts(
    client: DatabaseClient,
    selected: Sequence[Stay],
    mapping: Mapping[str, Any],
    *,
    work_dir: Path,
    chunk_size: int,
) -> tuple[list[dict[str, Any]], dict[str, int], str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
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
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("signature") != signature:
            raise ValueError(f"resume directory belongs to another eICU selection: {work_dir}")
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
    quality: Counter[str] = Counter()
    query_hashes: list[str] = []
    total_chunks = math.ceil(len(selected) / int(chunk_size))
    for chunk_index in range(total_chunks):
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
                raise ValueError(f"part {part_path} does not match the selected chunk")
        else:
            part_records, part_quality, query_hash = _extract_chunk(
                client, chunk, mapping
            )
            payload = {
                "records": part_records,
                "quality": part_quality,
                "query_hash": query_hash,
            }
            _atomic_pickle(part_path, payload)
        records.extend(part_records)
        for key, value in payload.get("quality", {}).items():
            quality[str(key)] += int(value)
        if payload.get("query_hash"):
            query_hashes.append(str(payload["query_hash"]))
        _atomic_json(
            work_dir / "progress.json",
            {"completed_chunks": chunk_index + 1, "total_chunks": total_chunks},
        )
        print(
            f"chunk {chunk_index + 1}/{total_chunks} stays={len(chunk)} "
            f"events={sum(len(record['events']) for record in part_records)}",
            flush=True,
        )
    records.sort(key=lambda record: int(record["record_id"]))
    if {int(record["record_id"]) for record in records} != {
        stay.stay_id for stay in selected
    }:
        raise AssertionError("eICU chunk assembly lost or duplicated stays")
    query_hash = hashlib.sha256(
        "\n".join(sorted(query_hashes)).encode("utf-8")
    ).hexdigest()
    return records, dict(quality), query_hash


def _load_frozen_standardizer(path: Path) -> EventStandardizer:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    standardizer = payload.get("standardizer")
    if isinstance(standardizer, EventStandardizer):
        result = standardizer
    elif isinstance(standardizer, Mapping):
        result = EventStandardizer.from_dict(standardizer)
    else:
        raise ValueError(f"MIMIC cache has no usable standardizer: {path}")
    if tuple(result.variable_names) != TIME_SERIES_VARIABLES:
        raise ValueError("MIMIC standardizer variable order does not match P12")
    return result


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


def _sanity_quantiles(cache_path: Path) -> dict[str, dict[str, Any]]:
    """Summarize raw canonical-unit values from both forecasting windows."""

    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    values: dict[str, list[float]] = {name: [] for name in TIME_SERIES_VARIABLES}
    for example in payload["examples"]:
        for container_name in ("obs", "target"):
            container = example[container_name]
            variable_ids = np.asarray(container["var"], dtype=np.int64)
            raw_values = np.asarray(container["value_raw"], dtype=np.float64)
            if variable_ids.shape != raw_values.shape:
                raise ValueError(
                    f"{container_name} variable/value shape mismatch in {cache_path}"
                )
            for variable_id, value in zip(variable_ids.tolist(), raw_values.tolist()):
                if not 0 <= int(variable_id) < len(TIME_SERIES_VARIABLES):
                    raise ValueError(f"invalid eICU variable id {variable_id}")
                if math.isfinite(float(value)):
                    values[TIME_SERIES_VARIABLES[int(variable_id)]].append(float(value))

    result: dict[str, dict[str, Any]] = {}
    for name in TIME_SERIES_VARIABLES:
        array = np.asarray(values[name], dtype=np.float64)
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
    """Load the measured P12 table using pandas' exact float parser."""

    import pandas as pd

    frame = pd.read_csv(path, float_precision="round_trip")
    required = {"var_id", "name", "n", "p1", "p50", "p99", "min", "max", "n_unique"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"reference table {path} is missing columns: {missing}")
    if frame["name"].duplicated().any():
        raise ValueError(f"reference table {path} contains duplicate variables")
    names = tuple(str(value) for value in frame["name"].tolist())
    if names != TIME_SERIES_VARIABLES:
        raise ValueError("reference table variables must exactly match the P12 order")
    rows: dict[str, dict[str, Any]] = {}
    for expected_id, name in enumerate(TIME_SERIES_VARIABLES):
        row = frame.iloc[expected_id]
        if int(row["var_id"]) != expected_id:
            raise ValueError(f"reference table var_id mismatch for {name}")
        rows[name] = {
            "var_id": expected_id,
            "n": int(row["n"]),
            **{
                key: (None if pd.isna(row[key]) else float(row[key]))
                for key in ("p1", "p50", "p99", "min", "max")
            },
            "n_unique": int(row["n_unique"]),
        }
    return rows


def write_sanity_check(
    cache_path: Path,
    reference_ranges: Path,
    mapping: Mapping[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> dict[str, Any]:
    """Compare eICU raw values with P12 and apply the registered unit gates.

    The P12 table is a measured reference, so its quantiles are reported for
    every variable.  The automatic decision is a dimensional/unit decision,
    not a claim that two hospitals have identical clinical distributions.
    """

    eicu = _sanity_quantiles(cache_path)
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
    structural_candidates = set(mapping.get("structural_candidates", ()))
    rows: list[dict[str, Any]] = []
    for name in TIME_SERIES_VARIABLES:
        values = eicu[name]
        reference_values = reference[name]
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
        if name == "FiO2" and values["n"]:
            if values["min"] < 0.0 or values["max"] > 1.1:
                unit_flags.append("FiO2 is outside canonical fraction scale")
        if name == "MechVent" and values["n"]:
            if values["min"] != 1.0 or values["max"] != 1.0:
                unit_flags.append("MechVent must contain only interval-start value 1 events")
        rows.append(
            {
                "variable": name,
                "var_id": int(values.get("var_id", TIME_SERIES_VARIABLES.index(name))),
                "eicu_n": values["n"],
                "eicu_p1": values["p1"],
                "eicu_p50": values["p50"],
                "eicu_p99": values["p99"],
                "eicu_min": values["min"],
                "eicu_max": values["max"],
                "eicu_n_unique": values["n_unique"],
                "p12_n": reference_values["n"],
                "p12_p1": reference_values["p1"],
                "p12_p50": reference_values["p50"],
                "p12_p99": reference_values["p99"],
                "p12_min": reference_values["min"],
                "p12_max": reference_values["max"],
                "p12_n_unique": reference_values["n_unique"],
                "automatic_dimension_check": dimension_ok,
                "unit_flags": unit_flags,
                "structural_candidate": name in structural_candidates,
                "missing_allowed": missing_allowed,
            }
        )
    automatic_pass = bool(
        len(rows) == len(TIME_SERIES_VARIABLES)
        and all(row["automatic_dimension_check"] and not row["unit_flags"] for row in rows)
    )
    report = {
        "dataset": "eicu",
        "cache": str(cache_path),
        "reference_ranges": str(reference_ranges),
        "automatic_pass": automatic_pass,
        "n_variables": len(rows),
        "n_dimension_pass": int(sum(row["automatic_dimension_check"] for row in rows)),
        "interpretation": (
            "P12 quantiles are a measured dimensional reference; distributional "
            "differences across institutions are descriptive and do not invalidate "
            "the unit gate."
        ),
        "rows": rows,
    }
    _atomic_json(json_path, report)

    def fmt(value: object) -> str:
        return "n/a" if value is None else f"{float(value):.6g}"

    lines = [
        "# eICU T3 raw-value and unit audit",
        "",
        f"Automatic dimensional/unit check: `{automatic_pass}` ({report['n_dimension_pass']}/{len(rows)} variables).",
        "",
        f"Measured P12 reference: `{reference_ranges}`.",
        "The P12 quantiles are reported as the registered dimensional reference; "
        "they are not an equality test for cross-institution clinical distributions.",
        "",
        "| variable | eICU n | eICU p1 | eICU p50 | eICU p99 | P12 n | P12 p1 | P12 p50 | P12 p99 | range/unit pass | flags |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variable']} | {row['eicu_n']} | {fmt(row['eicu_p1'])} | "
            f"{fmt(row['eicu_p50'])} | {fmt(row['eicu_p99'])} | {row['p12_n']} | "
            f"{fmt(row['p12_p1'])} | {fmt(row['p12_p50'])} | {fmt(row['p12_p99'])} | "
            f"{row['automatic_dimension_check'] and not row['unit_flags']} | "
            f"{'; '.join(row['unit_flags']) or '—'} |"
        )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _write_cache_and_split(
    examples: Sequence[dict[str, Any]],
    standardizer: EventStandardizer,
    split_indices: Mapping[str, Sequence[int]],
    *,
    processed_dir: Path,
    cache_name: str,
    split_name: str,
) -> tuple[Path, Path]:
    cache_path = processed_dir / cache_name
    split_path = processed_dir / split_name
    payload = {
        "examples": list(examples),
        "standardizer": standardizer.as_dict(),
        "variable_names": list(standardizer.variable_names),
        "sets": ["eicu"],
        "t_obs": DEFAULT_T_OBS,
        "split_seed": DEFAULT_SPLIT_SEED,
    }
    _atomic_pickle(cache_path, payload)
    labels = [int(example["label"]) for example in examples]
    split_payload = {
        "split_seed": DEFAULT_SPLIT_SEED,
        "t_obs": DEFAULT_T_OBS,
        "sets": ["eicu"],
        "n_samples": len(examples),
        "train": [int(index) for index in split_indices["train"]],
        "val": [int(index) for index in split_indices["val"]],
        "test": [int(index) for index in split_indices["test"]],
        "sample_ids": [str(example["sample_id"]) for example in examples],
        "label_counts": {
            name: int(sum(labels[index] for index in split_indices[name]))
            for name in ("train", "val", "test")
        },
    }
    _atomic_json(split_path, split_payload)
    return cache_path, split_path


def _cohort_frame(stays: Sequence[Stay]):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "subject_id": stay.subject_id,
                "hadm_id": stay.hadm_id,
                "stay_id": stay.stay_id,
                "hospital_id": stay.hospital_id,
                "uniquepid": stay.uniquepid,
                "gender": stay.gender,
                "anchor_age": stay.anchor_age,
                "intime": 0.0,
                "outtime": stay.outtime,
                "label": stay.label,
                "split": "external",
            }
            for stay in sorted(stays, key=lambda value: value.stay_id)
        ]
    )


def _write_cohort_files(
    stays: Sequence[Stay],
    subject_map: Mapping[str, int],
    *,
    processed_dir: Path,
    cohort_name: str,
    subject_map_name: str,
) -> tuple[Path, Path]:
    import pandas as pd

    processed_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = processed_dir / cohort_name
    subject_map_path = processed_dir / subject_map_name
    _cohort_frame(stays).to_csv(cohort_path, index=False)
    pd.DataFrame(
        {
            "uniquepid": list(sorted(subject_map)),
            "subject_id": [subject_map[key] for key in sorted(subject_map)],
        }
    ).to_csv(subject_map_path, index=False)
    return cohort_path, subject_map_path


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
            raise ValueError(f"required eICU variable {name} has no training events")
    temporary = map_path.with_name(map_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    os.replace(temporary, map_path)


def _manifest(
    *,
    mapping: Mapping[str, Any],
    schema: Mapping[str, Any],
    selected: Sequence[Stay],
    queue_metadata: Mapping[str, Any],
    sample_size: int,
    sample_seed: int,
    split_indices: Mapping[str, Sequence[int]],
    examples: Sequence[Mapping[str, Any]],
    quality: Mapping[str, int],
    query_hash: str,
    standardizer_path: Path,
    map_path: Path,
    database_url: str,
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
    labels = [int(example["label"]) for example in examples]
    conversion = {
        "FiO2": {
            "checked": int(quality.get("fio2_values_seen", 0)),
            "converted_percent_to_fraction": int(quality.get("fio2_percent_to_fraction", 0)),
            "rate": float(
                quality.get("fio2_percent_to_fraction", 0)
                / max(1, quality.get("fio2_values_seen", 0))
            ),
        },
        "Temp": {
            "checked": int(quality.get("temperature_values_seen", 0)),
            "out_of_range_dropped": int(quality.get("temperature_out_of_range_dropped", 0)),
        },
    }
    queue_payload = {
        **dict(queue_metadata.get("counts", {})),
        "sample_size": int(sample_size),
        "sample_seed": int(sample_seed),
        "sampling": "mortality_stratified_without_replacement",
        "selected_stay_ids_sha256": hashlib.sha256(
            ",".join(str(stay.stay_id) for stay in selected).encode("utf-8")
        ).hexdigest(),
        "selected_hospitals": int(len({stay.hospital_id for stay in selected})),
    }
    if "filter" in queue_metadata:
        queue_payload["filter"] = dict(queue_metadata["filter"])
    return {
        "dataset": "eicu",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "database_url": database_url.split("@")[-1],
        "map_path": str(map_path),
        "mapping_fingerprint": _map_fingerprint(mapping),
        "query_sql_sha256": query_hash,
        "schema_check": schema,
        "queue": queue_payload,
        "split": {
            "split_seed": DEFAULT_SPLIT_SEED,
            "fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
            "sizes": {name: len(split_indices[name]) for name in ("train", "val", "test")},
            "label_counts": {
                name: int(sum(labels[index] for index in split_indices[name]))
                for name in ("train", "val", "test")
            },
        },
        "event_counts": {"all_windows": total_counts, "train_split": train_counts},
        "unit_conversion": conversion,
        "quality": {str(key): int(value) for key, value in sorted(quality.items())},
        "structurally_missing": structural_missing,
        "excluded_variable_ids": [6, 19, 32],
        "staleness_hours": _staleness_summary(examples),
        "normalization": {
            "fit_source": "MIMIC-IV train split",
            "standardizer_cache": str(standardizer_path),
            "eicu_refit": False,
        },
        "examples": len(examples),
        "variable_names": list(TIME_SERIES_VARIABLES),
        "hospital_id_in_example": True,
    }


def _t0_facts(client: DatabaseClient, schema: str) -> dict[str, Any]:
    patient = _qualified(schema, "patient")
    patient_row = client.fetchall(
        f"""
        SELECT count(*) AS patient_rows,
               count(DISTINCT hospitalid) AS hospitals,
               count(*) FILTER (WHERE unitdischargeoffset >= 2880) AS long_stays,
               count(*) FILTER (WHERE trim(age) = '> 89') AS age_gt_89
        FROM {patient}
        """
    )[0]
    ge100 = client.fetchall(
        f"""
        SELECT count(*) AS hospitals_ge_100_long_stays
        FROM (
            SELECT hospitalid
            FROM {patient}
            WHERE unitdischargeoffset >= 2880
            GROUP BY hospitalid
            HAVING count(*) >= 100
        ) AS grouped
        """
    )[0]
    table_counts: dict[str, int] = {}
    for table in PIVOTED_TABLES:
        table_counts[table] = int(
            client.fetchall(f"SELECT count(*) AS rows FROM {_qualified(schema, table)}")[0]["rows"]
        )
    interval = client.fetchall(
        f"""
        WITH distinct_offsets AS (
            SELECT DISTINCT patientunitstayid, chartoffset
            FROM {_qualified(schema, 'pivoted_vital')}
            WHERE chartoffset IS NOT NULL
        ), ordered AS (
            SELECT chartoffset,
                   lag(chartoffset) OVER (
                       PARTITION BY patientunitstayid ORDER BY chartoffset
                   ) AS previous_offset
            FROM distinct_offsets
        ), intervals AS (
            SELECT chartoffset - previous_offset AS interval_minutes
            FROM ordered
            WHERE previous_offset IS NOT NULL AND chartoffset > previous_offset
        )
        SELECT count(*) AS positive_intervals,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY interval_minutes)
                   AS median_interval_minutes
        FROM intervals
        """
    )[0]
    table_row = client.fetchall(
        """
        SELECT count(*) AS total_tables
        FROM information_schema.tables
        WHERE table_schema = %s
        """,
        (schema,),
    )[0]
    index_rows = client.fetchall(
        """
        SELECT tablename, indexname
        FROM pg_indexes
        WHERE schemaname = %s
        ORDER BY tablename, indexname
        """,
        (schema,),
    )
    target_tables = sorted(
        {
            "patient",
            "vitalperiodic",
            "vitalaperiodic",
            "lab",
            "nursecharting",
            "respiratorycare",
            "treatment",
            *PIVOTED_TABLES,
        }
    )
    analyze_rows = client.fetchall(
        """
        SELECT relname AS table_name, last_analyze
        FROM pg_stat_user_tables
        WHERE schemaname = %s AND relname = ANY(%s)
        ORDER BY relname
        """,
        (schema, target_tables),
    )
    return {
        "patient_rows": int(patient_row["patient_rows"]),
        "hospitals": int(patient_row["hospitals"]),
        "long_stays": int(patient_row["long_stays"]),
        "age_gt_89": int(patient_row["age_gt_89"]),
        "hospitals_ge_100_long_stays": int(ge100["hospitals_ge_100_long_stays"]),
        "pivoted_table_rows": table_counts,
        "pivoted_vital_record_interval": {
            "method": "median positive interval between distinct chartoffset values per stay",
            "positive_intervals": int(interval["positive_intervals"]),
            "median_interval_minutes": float(interval["median_interval_minutes"]),
        },
        "total_tables": int(table_row["total_tables"]),
        "indexes": [
            {"table": str(row["tablename"]), "index": str(row["indexname"])}
            for row in index_rows
        ],
        "analyze": {
            str(row["table_name"]): row["last_analyze"] is not None
            for row in analyze_rows
        },
    }


def write_t0_report(
    client: DatabaseClient,
    mapping: Mapping[str, Any],
    *,
    output_dir: Path,
    map_path: Path,
) -> dict[str, Any]:
    schema = str(mapping.get("schema", "eicu_crd"))
    schema_report = schema_check(client, mapping)
    facts = _t0_facts(client, schema)
    facts["schema"] = schema
    facts["mapping_fingerprint"] = _map_fingerprint(mapping)
    facts["schema_report"] = schema_report
    expected = {
        "patient_rows": 200_859,
        "hospitals": 208,
        "long_stays": 77_832,
        "hospitals_ge_100_long_stays": 145,
        "pivoted_vital_rows": 21_032_656,
        "pivoted_lab_rows": 5_312_963,
        "pivoted_vital_interval_minutes": 15,
    }
    observed = {
        **{key: facts[key] for key in ("patient_rows", "hospitals", "long_stays", "hospitals_ge_100_long_stays")},
        "pivoted_vital_rows": facts["pivoted_table_rows"]["pivoted_vital"],
        "pivoted_lab_rows": facts["pivoted_table_rows"]["pivoted_lab"],
        "pivoted_vital_interval_minutes": facts["pivoted_vital_record_interval"]["median_interval_minutes"],
    }
    facts["documented_values"] = expected
    facts["observed_values"] = observed
    facts["documented_value_match"] = {
        key: bool(observed[key] == value)
        for key, value in expected.items()
    }
    facts["proceed"] = bool(schema_report["verified"] and all(facts["analyze"].values()))
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "eicu_t0_report.json", facts)
    lines = [
        "# eICU E-T0 live schema and index audit",
        "",
        f"Schema mapping verified: `{schema_report['verified']}`.",
        f"All analyzed target tables: `{all(facts['analyze'].values())}`.",
        f"E-T0 proceed: **{facts['proceed']}**.",
        "",
        "| fact | documented | observed | exact match |",
        "|---|---:|---:|:---:|",
    ]
    labels = {
        "patient_rows": "patient rows",
        "hospitals": "hospitals",
        "long_stays": "long stays >=48h",
        "hospitals_ge_100_long_stays": "hospitals with >=100 long stays",
        "pivoted_vital_rows": "pivoted_vital rows",
        "pivoted_lab_rows": "pivoted_lab rows",
        "pivoted_vital_interval_minutes": "pivoted_vital median interval (min)",
    }
    for key, label in labels.items():
        lines.append(
            f"| {label} | {expected[key]} | {observed[key]} | "
            f"{facts['documented_value_match'][key]} |"
        )
    lines.extend(
        [
            "",
            "The two pivoted row counts and the interval are retained as live measurements; they are not replaced by the historical 2026-08-27 snapshot.",
            "",
            f"Mapping file: `{map_path}`.",
            "",
        ]
    )
    (output_dir / "eicu_t0_report.md").write_text("\n".join(lines), encoding="utf-8")
    return facts


def extract(
    *,
    map_path: str | Path = DEFAULT_MAP,
    database_url: str,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    mimic_standardizer_cache: str | Path = DEFAULT_MIMIC_CACHE,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    split_seed: int = DEFAULT_SPLIT_SEED,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    cache_name: str = "eicu_events.pkl",
    split_name: str = "eicu_split_indices_12k.json",
    cohort_name: str = "eicu_cohort.csv",
    subject_map_name: str = "eicu_subject_map.csv",
    manifest_name: str = "eicu_manifest.json",
    reference_ranges: str | Path = DEFAULT_REFERENCE_RANGES,
    work_dir: str | Path | None = None,
    update_map: bool = False,
    keep_work: bool = False,
    hospital_min_stays: int | None = None,
) -> dict[str, Any]:
    if int(sample_seed) != DEFAULT_SAMPLE_SEED:
        raise ValueError("eICU sample_seed is preregistered as 42")
    if int(split_seed) != DEFAULT_SPLIT_SEED:
        raise ValueError("eICU split_seed is preregistered as 42")
    if float(DEFAULT_T_OBS) != 24.0 or float(DEFAULT_HORIZON) != 48.0:
        raise AssertionError("unexpected eICU window constants")
    map_path = Path(map_path)
    processed_dir = Path(processed_dir)
    results_dir = Path(results_dir)
    mapping = _load_map(map_path)
    client = DatabaseClient(database_url)
    work_path = (
        Path(work_dir)
        if work_dir is not None
        else processed_dir / f"eicu_extract_work_{int(sample_size)}"
    )
    try:
        schema_report = schema_check(client, mapping)
        if not schema_report["verified"]:
            raise RuntimeError(f"E-T0 schema check failed: {schema_report['missing']}")
        all_stays, queue_metadata, subject_map = _fetch_cohort(client, mapping)
        if hospital_min_stays is not None:
            minimum = int(hospital_min_stays)
            if minimum < 1:
                raise ValueError("hospital_min_stays must be positive")
            long_stay_hospital_counts = _fetch_long_stay_hospital_counts(client, mapping)
            selected_hospitals = {
                hospital_id
                for hospital_id, count in long_stay_hospital_counts.items()
                if count >= minimum
            }
            eligible_hospital_counts = Counter(stay.hospital_id for stay in all_stays)
            hospitals_without_eligible_stays = sorted(
                hospital_id
                for hospital_id in selected_hospitals
                if hospital_id not in eligible_hospital_counts
            )
            if hospitals_without_eligible_stays:
                raise ValueError(
                    "a selected long-stay hospital has no stay in the registered "
                    f"forecast cohort: {hospitals_without_eligible_stays}"
                )
            all_stays = [
                stay for stay in all_stays if stay.hospital_id in selected_hospitals
            ]
            all_stays.sort(key=lambda stay: stay.stay_id)
            if int(sample_size) != len(all_stays):
                raise ValueError(
                    "a hospital-filtered eICU extraction must use every selected stay; "
                    f"sample_size={sample_size}, selected_stays={len(all_stays)}"
                )
            if not all_stays:
                raise ValueError("hospital filter produced an empty eICU cohort")
            filtered_counts = dict(queue_metadata.get("counts", {}))
            filtered_counts.update(
                {
                    "eligible_first_unit_age_los_label": len(all_stays),
                    "eligible_subjects": len({stay.uniquepid for stay in all_stays}),
                    "eligible_hospitals": len(selected_hospitals),
                    "expired": sum(stay.label for stay in all_stays),
                    "alive": sum(1 - stay.label for stay in all_stays),
                    "age_gt_89_mapped_to_90": sum(
                        stay.anchor_age == 90 for stay in all_stays
                    ),
                    "admission_weight_nonnull": sum(
                        stay.admission_weight is not None for stay in all_stays
                    ),
                }
            )
            queue_metadata = dict(queue_metadata)
            queue_metadata["counts"] = filtered_counts
            queue_metadata["filter"] = {
                "hospital_min_stays": minimum,
                "hospital_count_basis": (
                    "all patient rows with unitdischargeoffset >= 2880; "
                    "same as E-T0 hospitals_ge_100_long_stays"
                ),
                "selected_hospitals": len(selected_hospitals),
                "selected_hospital_ids": sorted(int(value) for value in selected_hospitals),
                "long_stay_hospital_counts": {
                    str(hospital_id): int(long_stay_hospital_counts[hospital_id])
                    for hospital_id in sorted(selected_hospitals)
                },
                "eligible_stay_counts": {
                    str(hospital_id): int(eligible_hospital_counts[hospital_id])
                    for hospital_id in sorted(selected_hospitals)
                },
                "selection": "all forecast-cohort eligible stays from hospitals with >=100 raw long-stay rows",
            }
        selected = _stratified_sample(
            all_stays, sample_size=int(sample_size), seed=int(sample_seed)
        )
        records, quality, query_hash = _load_or_extract_parts(
            client,
            selected,
            mapping,
            work_dir=work_path,
            chunk_size=int(chunk_size),
        )
    finally:
        client.close()

    labels = [int(record["label"]) for record in records]
    split_indices = stratified_event_split(labels, seed=int(split_seed))
    standardizer_path = Path(mimic_standardizer_cache)
    standardizer = _load_frozen_standardizer(standardizer_path)
    examples: list[dict[str, Any]] = []
    stay_by_id = {stay.stay_id: stay for stay in selected}
    for record in records:
        example = make_event_example(record, standardizer, t_obs=DEFAULT_T_OBS)
        stay = stay_by_id[int(record["record_id"])]
        example.update(
            {
                "hospital_id": stay.hospital_id,
                "subject_id": stay.subject_id,
                "hadm_id": stay.hadm_id,
                "uniquepid": stay.uniquepid,
                "gender": stay.gender,
                "anchor_age": stay.anchor_age,
            }
        )
        examples.append(example)
    if len(examples) != int(sample_size):
        raise AssertionError("eICU cache example count does not equal sample_size")
    cache_path, split_path = _write_cache_and_split(
        examples,
        standardizer,
        split_indices,
        processed_dir=processed_dir,
        cache_name=cache_name,
        split_name=split_name,
    )
    cohort_path, subject_map_path = _write_cohort_files(
        selected,
        subject_map,
        processed_dir=processed_dir,
        cohort_name=cohort_name,
        subject_map_name=subject_map_name,
    )
    train_counts = _event_counts(
        [examples[index] for index in split_indices["train"]]
    )
    if update_map:
        _update_map_status(map_path, mapping, train_counts)
        mapping = _load_map(map_path)
    manifest = _manifest(
        mapping=mapping,
        schema=schema_report,
        selected=selected,
        queue_metadata=queue_metadata,
        sample_size=sample_size,
        sample_seed=sample_seed,
        split_indices=split_indices,
        examples=examples,
        quality=quality,
        query_hash=query_hash,
        standardizer_path=standardizer_path,
        map_path=map_path,
        database_url=database_url,
    )
    manifest_path = results_dir / manifest_name
    reference_path = Path(reference_ranges)
    if not reference_path.exists():
        raise FileNotFoundError(f"registered P12 reference table is missing: {reference_path}")
    manifest_stem = Path(manifest_name).stem
    sanity_json_path = results_dir / f"{manifest_stem}_t3_sanity.json"
    sanity_markdown_path = results_dir / f"{manifest_stem}_t3_sanity.md"
    sanity = write_sanity_check(
        cache_path,
        reference_path,
        mapping,
        sanity_json_path,
        sanity_markdown_path,
    )
    manifest["sanity_check"] = {
        "path": str(sanity_markdown_path),
        "json_path": str(sanity_json_path),
        "reference_ranges": str(reference_path),
        "automatic_dimension_check": bool(sanity["automatic_pass"]),
        "n_variables": int(sanity["n_variables"]),
        "n_dimension_pass": int(sanity["n_dimension_pass"]),
    }
    _atomic_json(manifest_path, manifest)
    if not keep_work:
        shutil.rmtree(work_path, ignore_errors=True)
    return {
        "cache_path": str(cache_path),
        "split_path": str(split_path),
        "cohort_path": str(cohort_path),
        "subject_map_path": str(subject_map_path),
        "manifest_path": str(manifest_path),
        "n_samples": len(examples),
        "split_sizes": {name: len(value) for name, value in split_indices.items()},
        "structurally_missing": manifest["structurally_missing"],
        "event_counts": manifest["event_counts"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", dest="map_path", default=str(DEFAULT_MAP))
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--schema", default=None, help="reserved for live map schema; use EICU_SCHEMA in the environment")
    parser.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--mimic-standardizer-cache", default=str(DEFAULT_MIMIC_CACHE))
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--cache-name", default="eicu_events.pkl")
    parser.add_argument("--split-name", default="eicu_split_indices_12k.json")
    parser.add_argument("--cohort-name", default="eicu_cohort.csv")
    parser.add_argument("--subject-map-name", default="eicu_subject_map.csv")
    parser.add_argument("--manifest-name", default="eicu_manifest.json")
    parser.add_argument("--reference-ranges", default=str(DEFAULT_REFERENCE_RANGES))
    parser.add_argument("--work-dir")
    parser.add_argument("--update-map", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--schema-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    database_url = args.database_url or os.environ.get("EICU_PG_DSN")
    if not database_url:
        raise SystemExit("EICU_PG_DSN is required")
    map_path = Path(args.map_path)
    mapping = _load_map(map_path)
    requested_schema = args.schema or os.environ.get("EICU_SCHEMA")
    if requested_schema and str(requested_schema) != str(mapping.get("schema", "eicu_crd")):
        raise SystemExit("--schema/EICU_SCHEMA must match configs/eicu_variable_map.yaml")
    if args.schema_check:
        client = DatabaseClient(database_url)
        try:
            report = write_t0_report(
                client,
                mapping,
                output_dir=Path(args.results_dir),
                map_path=map_path,
            )
        finally:
            client.close()
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        if not report["proceed"]:
            raise SystemExit(1)
        return
    result = extract(
        map_path=map_path,
        database_url=database_url,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        mimic_standardizer_cache=args.mimic_standardizer_cache,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        split_seed=args.split_seed,
        chunk_size=args.chunk_size,
        cache_name=args.cache_name,
        split_name=args.split_name,
        cohort_name=args.cohort_name,
        subject_map_name=args.subject_map_name,
        manifest_name=args.manifest_name,
        reference_ranges=args.reference_ranges,
        work_dir=args.work_dir,
        update_map=args.update_map,
        keep_work=args.keep_work,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
