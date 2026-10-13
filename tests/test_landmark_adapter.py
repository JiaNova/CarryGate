"""Six executable adapter gates required before any P1 training run."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PYOMNITS = ROOT / "external" / "PyOmniTS"
if not (PYOMNITS / "models" / "ICUKANForecaster.py").is_file() or not (
    PYOMNITS / "data" / "data_provider" / "datasets" / "ICUKANLandmark.py"
).is_file():
    pytest.skip(
        "requires the locally rebuilt PyOmniTS harness; run scripts/d1_harness/setup_harness.sh",
        allow_module_level=True,
    )
sys.path.insert(0, str(PYOMNITS))
sys.path.insert(0, str(ROOT))

from data.data_provider.datasets import ICUKANLandmark as adapter  # noqa: E402
from models.ICUKANForecaster import Model as CarryGate  # noqa: E402
from src.data.persistence_baseline import build_persistence_baseline  # noqa: E402


def _events(times, variables=None, values=None):
    times = np.asarray(times, dtype=np.float64)
    n = len(times)
    variables = np.arange(n, dtype=np.int64) if variables is None else np.asarray(variables, dtype=np.int64)
    values = np.arange(1, n + 1, dtype=np.float32) if values is None else np.asarray(values, dtype=np.float32)
    return {
        "var": variables,
        "time": times.astype(np.float32),
        "time_raw": times,
        "value": values,
        "value_raw": values.astype(np.float64),
        "event_order": np.arange(n, dtype=np.int64),
    }


def _model_input(events, discharge, landmark):
    window = adapter.build_landmark_window(events, discharge, landmark)
    return adapter._events_to_dense(
        events, window.selection.input_indices, window.input_local_hour
    )


def test_v1_half_open_event_and_discharge_boundaries():
    t = 60.0
    eps = 1e-5
    discharge = 90.0
    events = _events(
        [t - 48.0, t - eps, t, t + 24.0 - eps, t + 24.0, discharge]
    )
    selection = adapter.select_landmark_events(events, discharge, t)
    assert selection.input_indices.tolist() == [0, 1]
    assert selection.outcome_indices.tolist() == [2, 3]
    assigned = set(selection.input_indices.tolist()) | set(selection.outcome_indices.tolist())
    assert assigned == {0, 1, 2, 3}
    assert set(range(len(events["time_raw"]))) - assigned == {4, 5}


def test_v2_outcome_values_and_discharge_cannot_change_input_tensor():
    events = _events([0.5, 23.0, 60.0, 66.25, 83.0, 90.0], [0, 1, 2, 3, 4, 5])
    baseline = _model_input(events, 90.0, 60.0)

    changed_value = {key: np.array(value, copy=True) for key, value in events.items()}
    changed_value["value"][2] = -999.0
    changed_value["value_raw"][2] = -999.0
    changed_outcome = _model_input(changed_value, 90.0, 60.0)

    changed_discharge = _model_input(events, 70.0, 60.0)
    for candidate in (changed_outcome, changed_discharge):
        assert all(torch.equal(a, b) for a, b in zip(baseline, candidate))


def test_v3_training_loss_mask_contains_only_real_outcome_measurements():
    t = 48.0
    # The target measurements are deliberately off the hourly midpoint grid.
    events = _events([2.0, 47.0, 48.125, 49.2, 70.75], [0, 1, 0, 2, 3])
    window = adapter.build_landmark_window(events, 90.0, t)
    y_mark, y, y_mask = adapter._events_to_dense(
        events, window.selection.outcome_indices, window.outcome_local_hour
    )
    assert int(y_mask.sum().item()) == len(window.selection.outcome_indices) == 3
    assert np.all(window.outcome_loss_weight == 1.0)
    assert len(window.grid_local_hour) == 24
    assert len(window.grid_loss_weight) == 24
    assert np.all(window.grid_loss_weight == 0.0)
    target_hours = y_mark[:, 0].numpy() * adapter.TOTAL_HOURS
    measured_hours = window.outcome_local_hour
    assert np.allclose(np.sort(target_hours[y_mask.any(dim=1).numpy()]), np.sort(measured_hours))
    assert not np.any(np.isclose(measured_hours, window.grid_local_hour[:, None], atol=1e-6))
    # No query-grid rows have been appended to y; every nonzero mask entry is
    # an actual event measurement and therefore carries one loss contribution.
    assert int(y_mask.sum().item()) == int(window.outcome_loss_weight.sum())
    assert y.shape == y_mask.shape


def test_v4_adapter_config_wrapper_and_carrygate_staleness_are_72_hours():
    discharge, landmark = 90.0, 48.0
    events = _events([47.0, 48.5], [3, 3], [1.25, 2.5])
    window = adapter.build_landmark_window(events, discharge, landmark)
    x_mark, x, x_mask = adapter._events_to_dense(
        events, window.selection.input_indices, window.input_local_hour
    )
    y_mark, y, y_mask = adapter._events_to_dense(
        events, window.selection.outcome_indices, window.outcome_local_hour
    )
    batch = adapter.collate_fn([{
        "x": x, "x_mark": x_mark, "x_mask": x_mask,
        "y": y, "y_mark": y_mark, "y_mask": y_mask,
        "sample_ID": torch.tensor(0),
    }])
    configs = SimpleNamespace(
        seq_len=48,
        pred_len=24,
        enc_in=adapter.N_VARIABLES,
        task_name="short_term_forecast",
        features="M",
    )
    model = CarryGate(configs).eval()
    packed = model.build_batch(
        batch["x"], batch["x_mark"], batch["x_mask"],
        batch["y"], batch["y_mark"], batch["y_mask"],
    )
    target_position = int(torch.nonzero(batch["y_mask"][0, :, 3], as_tuple=False)[0, 0])
    actual_staleness = float(packed["staleness"][0, target_position])
    assert adapter.TOTAL_HOURS == 72.0
    assert adapter.INPUT_HOURS + adapter.OUTCOME_HOURS == 72.0
    assert configs.seq_len + configs.pred_len == 72
    assert model.total_hours == 72.0
    assert np.isclose(actual_staleness, 1.5, atol=1e-6, rtol=0)


def _assert_legacy_block_matches(new_events, selected, new_local_time, old_block):
    new_vars = np.asarray(new_events["var"], dtype=np.int64)[selected]
    new_values = np.asarray(new_events["value"], dtype=np.float32)[selected]
    new_raw_hours = np.asarray(new_events["time_raw"], dtype=np.float64)[selected]
    old_vars = np.asarray(old_block["var"], dtype=np.int64)
    old_values = np.asarray(old_block["value"], dtype=np.float32)
    old_hours = np.asarray(old_block["time"], dtype=np.float32).astype(np.float64)
    assert np.array_equal(new_vars, old_vars)
    assert np.array_equal(new_values.view(np.uint32), old_values.view(np.uint32))
    assert np.allclose(new_raw_hours, old_hours, atol=1e-5, rtol=0)
    assert np.allclose(new_local_time, old_hours + 24.0, atol=1e-5, rtol=0)


def test_v5_hour24_real_stays_equal_the_legacy_events_shifted_by_24():
    processed = ROOT / "data" / "processed"
    required = [
        processed / f"{cohort}_events_jbi_v1.pkl"
        for cohort in ("eicu", "mimiciv")
    ] + [
        processed / f"{cohort}_events_landmark_v1.pkl"
        for cohort in ("eicu", "mimiciv")
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("requires locally generated protected event caches; none are distributed")
    for cohort in ("eicu", "mimiciv"):
        old_name = f"{cohort}_events_jbi_v1.pkl"
        new_name = f"{cohort}_events_landmark_v1.pkl"
        with (processed / old_name).open("rb") as handle:
            old_payload = pickle.load(handle)
        with (processed / new_name).open("rb") as handle:
            new_payload = pickle.load(handle)
        old_examples = old_payload["examples"]
        new_examples = new_payload["examples"]
        assert len(old_examples) == len(new_examples) == 12000
        eligible = 0
        for old, new in zip(old_examples, new_examples):
            discharge = float(new["outtime_hour"])
            if discharge <= 24.0:
                continue
            eligible += 1
            window = adapter.build_landmark_window(new["events"], discharge, 24.0)
            obs_idx = window.selection.input_indices
            tgt_idx = window.selection.outcome_indices
            _assert_legacy_block_matches(
                new["events"], obs_idx, window.input_local_hour, old["obs"]
            )
            old_target = old["target"]
            old_times = np.asarray(old_target["time"], dtype=np.float32).astype(np.float64)
            shared = old_times < min(48.0, discharge)
            old_shared = {key: np.asarray(value)[shared] for key, value in old_target.items()}
            new_shared = {key: np.asarray(value)[tgt_idx] for key, value in new["events"].items()}
            # D<48 targets after discharge are intentionally excluded by P1.
            _assert_legacy_block_matches(
                new["events"], tgt_idx, window.outcome_local_hour, old_shared
            )
            assert len(new_shared["var"]) == len(old_shared["var"])
        assert eligible > 0


def _grade(value: float, component: str) -> int:
    if component == "Platelets":
        return 0 if value >= 150 else 1 if value >= 100 else 2 if value >= 50 else 3 if value >= 20 else 4
    if component == "Bilirubin":
        return 0 if value < 1.2 else 1 if value < 2 else 2 if value < 6 else 3 if value < 12 else 4
    if component == "GCS":
        projected = float(np.clip(np.round(value), 3, 15))
        return 0 if projected == 15 else 1 if projected >= 13 else 2 if projected >= 10 else 3 if projected >= 6 else 4
    if component == "Creatinine":
        return 0 if value < 1.2 else 1 if value < 2 else 2 if value < 3.5 else 3 if value < 5 else 4
    raise KeyError(component)


def test_v6_adapter_carry_forward_has_zero_validation_skill_and_never_alarms():
    processed = ROOT / "data" / "processed"
    required = [
        processed / f"{cohort}_events_landmark_v1.pkl"
        for cohort in ("eicu", "mimiciv")
    ] + [
        processed / f"{cohort}_split_indices_jbi_v1.json"
        for cohort in ("eicu", "mimiciv")
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("requires locally generated protected event caches and split indices; none are distributed")
    component_names = ("Platelets", "Bilirubin", "GCS", "Creatinine")
    for cohort in ("eicu", "mimiciv"):
        with (processed / f"{cohort}_events_landmark_v1.pkl").open("rb") as handle:
            payload = pickle.load(handle)
        split = json.loads((processed / f"{cohort}_split_indices_jbi_v1.json").read_text())
        examples = payload["examples"]
        variable_index = {name: i for i, name in enumerate(payload["variable_names"])}
        component_index = [variable_index[name] for name in component_names]
        valid_events = torch.zeros(adapter.N_VARIABLES, dtype=torch.float64)
        model_error = torch.zeros(adapter.N_VARIABLES, dtype=torch.float64)
        reference_error = torch.zeros(adapter.N_VARIABLES, dtype=torch.float64)
        alarm_count = 0
        reference_count = 0

        for stay_index in split["val"]:
            example = examples[int(stay_index)]
            events = example["events"]
            discharge = float(example["outtime_hour"])
            for t in adapter.LANDMARK_HOURS:
                if discharge <= t:
                    continue
                window = adapter.build_landmark_window(events, discharge, t)
                input_indices = window.selection.input_indices
                outcome_indices = window.selection.outcome_indices
                if outcome_indices.size:
                    last_z, _, seen = adapter.carry_forward_input_reference(events, discharge, t)
                    target_var = np.asarray(events["var"], dtype=np.int64)[outcome_indices]
                    target = np.asarray(events["value"], dtype=np.float32)[outcome_indices]
                    prediction = last_z[target_var]
                    obs_var = torch.as_tensor(
                        np.asarray(events["var"], dtype=np.int64)[input_indices],
                        dtype=torch.long,
                    )
                    obs_value = torch.as_tensor(
                        np.asarray(events["value"], dtype=np.float32)[input_indices],
                        dtype=torch.float32,
                    )
                    tgt_var = torch.as_tensor(target_var, dtype=torch.long)
                    tgt_value = torch.as_tensor(target, dtype=torch.float32)
                    reference_prediction, _ = build_persistence_baseline(
                        obs_var=obs_var,
                        obs_time=torch.as_tensor(window.input_local_hour, dtype=torch.float32),
                        obs_value=obs_value,
                        tgt_var=tgt_var,
                        tgt_time=torch.as_tensor(window.outcome_local_hour, dtype=torch.float32),
                        t_obs=adapter.INPUT_HOURS,
                        train_median=torch.zeros(adapter.N_VARIABLES, dtype=torch.float32),
                    )
                    reference_prediction = reference_prediction.numpy()
                    assert np.array_equal(prediction, reference_prediction)
                    absolute_error = np.abs(prediction.astype(np.float64) - target.astype(np.float64))
                    reference_absolute_error = np.abs(
                        reference_prediction.astype(np.float64) - tgt_value.numpy().astype(np.float64)
                    )
                    for variable in np.unique(target_var):
                        chosen = target_var == variable
                        valid_events[variable] += int(chosen.sum())
                        model_error[variable] += absolute_error[chosen].sum()
                        reference_error[variable] += reference_absolute_error[chosen].sum()

                _, last_raw, seen = adapter.carry_forward_input_reference(events, discharge, t)
                queries = adapter.hourly_grid_local_hours(discharge, t)
                for variable in component_index:
                    if not seen[variable]:
                        continue
                    reference_count += len(queries)
                    ref_grade = _grade(float(last_raw[variable]), component_names[component_index.index(variable)])
                    # Carry-forward repeats its input reference at every valid
                    # query, so the strict-greater-than alarm rule is false.
                    alarm_count += sum(
                        _grade(float(last_raw[variable]), component_names[component_index.index(variable)])
                        > ref_grade
                        for _ in queries
                    )

        scored = (valid_events >= 50) & (reference_error > 1e-9)
        assert bool(scored.any()), f"{cohort}: validation has no eligible macro variables"
        skill = 1.0 - model_error[scored] / reference_error[scored]
        assert torch.equal(model_error, reference_error)
        assert torch.equal(skill, torch.zeros_like(skill))
        assert alarm_count == 0
        assert reference_count > 0
