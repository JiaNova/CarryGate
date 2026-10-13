"""Event-level parser for the PhysioNet/CinC Challenge 2012 data."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


STATIC_FEATURES: Tuple[str, ...] = ("Age", "Gender", "Height", "ICUType", "Weight")

# The official challenge page calls these 37 time-series variables.  The
# downloaded Set-A files use TroponinI/TroponinT; older challenge prose and
# some third-party loaders abbreviate them as TropI/TropT.  We canonicalize
# both spellings below while retaining one stable internal vocabulary.
TIME_SERIES_VARIABLES: Tuple[str, ...] = (
    "Albumin",
    "ALP",
    "ALT",
    "AST",
    "BUN",
    "Bilirubin",
    "Cholesterol",
    "Creatinine",
    "DiasABP",
    "FiO2",
    "GCS",
    "Glucose",
    "HCO3",
    "HCT",
    "HR",
    "K",
    "Lactate",
    "Mg",
    "MAP",
    "MechVent",
    "Na",
    "NIDiasABP",
    "NIMAP",
    "NISysABP",
    "PaCO2",
    "PaO2",
    "pH",
    "Platelets",
    "RespRate",
    "SaO2",
    "SysABP",
    "Temp",
    "TroponinI",
    "TroponinT",
    "Urine",
    "WBC",
    "Weight",
)

TIME_SERIES_SET = frozenset(TIME_SERIES_VARIABLES)
STATIC_SET = frozenset(STATIC_FEATURES)
KNOWN_PARAMETERS = frozenset({"RecordID", *STATIC_FEATURES, *TIME_SERIES_VARIABLES})
PARAMETER_ALIASES = {"TropI": "TroponinI", "TropT": "TroponinT"}


def annotate_events(
    events: Iterable[Tuple[str, float, float]],
) -> List[Dict[str, object]]:
    """Add per-variable time-gap metadata to raw event triples.

    The legacy ``events`` field remains a list of ``(variable, time, value)``
    triples so the Phase 0 hourly pipeline stays compatible.  Phase 1 uses
    the returned records (also exposed by :func:`parse_record` as
    ``events_with_delta``) and therefore never has to recompute gaps in a
    later, potentially leaky, stage.
    """

    ordered = list(enumerate(events))
    ordered.sort(key=lambda item: (float(item[1][1]), item[0]))
    last_time: Dict[str, float] = {}
    annotated: List[Dict[str, object]] = []
    for _, (variable, time, value) in ordered:
        variable = canonicalize_parameter(variable)
        time = float(time)
        value = float(value)
        previous = last_time.get(variable)
        is_first = previous is None
        delta = time if is_first else max(0.0, time - previous)
        annotated.append(
            {
                "variable": variable,
                "time": time,
                "value": value,
                "delta_t": delta,
                "is_first_observation": bool(is_first),
            }
        )
        # Records are read in chronological order in the official files, but
        # taking the maximum makes the invariant explicit for synthetic and
        # hand-written records used by tests.
        last_time[variable] = max(time, previous) if previous is not None else time
    return annotated


def canonicalize_parameter(parameter: str) -> str:
    """Normalize known historical spellings without silently dropping data."""

    return PARAMETER_ALIASES.get(parameter.strip(), parameter.strip())


def parse_time(value: str) -> float:
    """Parse an official ``HH:MM`` timestamp into elapsed hours."""

    text = str(value).strip()
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"invalid P12 time {value!r}; expected HH:MM")
    hour, minute = int(parts[0]), int(parts[1])
    if hour < 0 or minute < 0 or minute >= 60:
        raise ValueError(f"invalid P12 time {value!r}")
    return hour + minute / 60.0


def _parse_value(value: str) -> float:
    number = float(str(value).strip())
    if not math.isfinite(number):
        raise ValueError(f"non-finite P12 value: {value!r}")
    return number


def parse_record(path: str | Path) -> dict:
    """Parse one record while retaining raw event-level observations.

    Returns
    -------
    dict
        ``record_id`` (int), ``static`` (mapping), ``events`` (list of
        ``(variable, hour_float, value)``), and ``unknown_parameters``.

    ``-1`` is the official missing/unknown sentinel.  Missing static values are
    represented as ``None`` and missing time-series values are not emitted as
    events.  The first occurrence of each general descriptor at 00:00 is read
    as static; the time-varying ``Weight`` observations after that remain
    events.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    static: MutableMapping[str, float | None] = {name: None for name in STATIC_FEATURES}
    events: List[Tuple[str, float, float]] = []
    unknown = set()
    record_id: int | None = None

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        first = next(reader, None)
        if first is None:
            raise ValueError(f"empty P12 record: {path}")
        header = [item.strip() for item in first]
        if header[:3] == ["Time", "Parameter", "Value"]:
            rows = reader
        else:
            rows = iter([first, *reader])

        for row_number, row in enumerate(rows, start=2):
            if len(row) < 3:
                raise ValueError(f"malformed row {row_number} in {path}: {row!r}")
            time_text, raw_parameter, value_text = (str(item).strip() for item in row[:3])
            parameter = canonicalize_parameter(raw_parameter)
            hour = parse_time(time_text)
            value = _parse_value(value_text)

            if parameter == "RecordID":
                if value < 0 or int(value) != value:
                    raise ValueError(f"invalid RecordID in {path}: {value}")
                candidate = int(value)
                if record_id is not None and candidate != record_id:
                    raise ValueError(f"multiple RecordID values in {path}")
                record_id = candidate
                continue

            if parameter not in KNOWN_PARAMETERS:
                unknown.add(parameter)

            # General descriptors occur once at 00:00 in official records. A
            # second Weight row is a legitimate time-series observation.
            if (
                parameter in STATIC_SET
                and hour == 0.0
                and static[parameter] is None
            ):
                static[parameter] = None if value == -1 else value
                continue

            if parameter in TIME_SERIES_SET and value >= 0:
                events.append((parameter, hour, value))

    if record_id is None:
        try:
            record_id = int(path.stem)
        except ValueError as exc:
            raise ValueError(f"record {path} has no RecordID") from exc
    events_with_delta = annotate_events(events)
    return {
        "record_id": record_id,
        "static": dict(static),
        "events": events,
        "events_with_delta": events_with_delta,
        # Array aliases make the Phase 1 metadata explicit without changing
        # the legacy triple representation consumed by Phase 0.
        "delta_t": [float(event["delta_t"]) for event in events_with_delta],
        "is_first_observation": [
            bool(event["is_first_observation"]) for event in events_with_delta
        ],
        "unknown_parameters": sorted(unknown),
    }


def iter_record_paths(raw_dir: str | Path, set_name: str = "set-a") -> List[Path]:
    """Return numerically sorted record paths from an extracted P12 set."""

    directory = Path(raw_dir) / set_name
    if not directory.is_dir():
        raise FileNotFoundError(
            f"missing extracted {set_name} directory: {directory}; run p12_download first"
        )
    paths = [path for path in directory.glob("*.txt") if path.stem.isdigit()]
    return sorted(paths, key=lambda path: int(path.stem))


def parse_records(paths: Iterable[str | Path]) -> List[dict]:
    """Parse paths and fail loudly if RecordIDs are duplicated."""

    records = [parse_record(path) for path in paths]
    ids = [record["record_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate RecordID detected in P12 records")
    return records


def load_outcomes(path: str | Path) -> Dict[int, int]:
    """Load and validate official ``Outcomes-a.txt`` labels."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"outcome file has no header: {path}")
        fields = {field.strip() for field in reader.fieldnames}
        required = {"RecordID", "In-hospital_death"}
        if not required.issubset(fields):
            raise ValueError(
                f"{path} must contain {sorted(required)}, found {reader.fieldnames}"
            )
        outcomes: Dict[int, int] = {}
        for row_number, row in enumerate(reader, start=2):
            record_id = int(float(row["RecordID"]))
            label = int(float(row["In-hospital_death"]))
            if label not in (0, 1):
                raise ValueError(f"non-binary death label at row {row_number}: {label}")
            if record_id in outcomes:
                raise ValueError(f"duplicate outcome RecordID: {record_id}")
            outcomes[record_id] = label
    if not outcomes:
        raise ValueError(f"no outcomes found in {path}")
    return outcomes
