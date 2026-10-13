#!/usr/bin/env python
"""Gate D3b: the PyOmniTS wrapper must rebuild the project's own batch exactly.

The wrapper in ``ICUKANForecaster.py`` reconstructs observation triplets,
per-variable inter-arrival times and the carry-forward anchor from a dense grid.
If any of that differs from what ``src/data/p12_forecast_dataset.py`` builds,
the benchmark number the wrapper produces would describe a different model from
the one the paper reports.

The comparison runs on this project's own cohort, where both paths are defined:
the dense grid comes from the D1 dataset adapter, and the reference comes from
the project's own loader. Observation ORDER is allowed to differ, because the
backbone is permutation-invariant over observations; the multiset of
(variable, value, delta_t) triples is not allowed to differ.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.p12_forecast_dataset import load_forecast_splits, make_forecast_loaders  # noqa: E402
from src.data.persistence_baseline import build_persistence_baseline  # noqa: E402

N_VARIABLES = 37
L_TOTAL = 48.0


def dense_from_triplets(var, time, value, n_rows_hint=None):
    """The D1 adapter's layout: repeats go on extra rows at the same timestamp."""
    order = np.lexsort((var, time))
    var, time, value = var[order], time[order], value[order]
    _, inverse, counts = np.unique(np.stack([time, var]), axis=1,
                                   return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    group_start = np.searchsorted(inverse, np.arange(counts.size))
    rank = np.arange(inverse.size) - group_start[inverse]
    uniq_time, time_index = np.unique(time, return_inverse=True)
    rows_per_time = np.zeros(uniq_time.size, dtype=np.int64)
    np.maximum.at(rows_per_time, time_index, rank + 1)
    row_start = np.concatenate([[0], np.cumsum(rows_per_time)[:-1]])
    row_of = row_start[time_index] + rank
    n_rows = int(rows_per_time.sum())
    dense = np.full((n_rows, N_VARIABLES), np.nan, dtype=np.float32)
    dense[row_of, var] = value
    t = np.repeat(uniq_time, rows_per_time)
    return t, dense


def main() -> None:
    bundle = load_forecast_splits(
        str(ROOT / "data" / "processed" / "p12_events.pkl"),
        str(ROOT / "data" / "processed" / "split_indices_12k.json"))
    loaders = make_forecast_loaders(bundle, batch_size=16)
    reference = next(iter(loaders["test"]))
    if "x_last" not in reference:
        raise SystemExit(
            "the reference loader did not attach the persistence anchor; "
            "load_forecast_splits must be built with a train median")

    # Build the dense grid the harness would hand the wrapper, for these samples.
    examples = bundle.cache["examples"]

    batch_size = reference["obs_var"].shape[0]
    max_obs_rows = max_tgt_rows = 0
    built = []
    for i in range(batch_size):
        sample_id = int(reference["record_id"][i])
        example = next(e for e in examples if int(e["record_id"]) == sample_id)
        obs, tgt = example["obs"], example["target"]
        to, xo = dense_from_triplets(np.asarray(obs["var"]), np.asarray(obs["time"]),
                                     np.asarray(obs["value"]))
        tt, xt = dense_from_triplets(np.asarray(tgt["var"]), np.asarray(tgt["time"]),
                                     np.asarray(tgt["value"]))
        built.append((to, xo, tt, xt))
        max_obs_rows = max(max_obs_rows, xo.shape[0])
        max_tgt_rows = max(max_tgt_rows, xt.shape[0])

    x = torch.zeros(batch_size, max_obs_rows, N_VARIABLES)
    x_mask = torch.zeros_like(x)
    x_mark = torch.zeros(batch_size, max_obs_rows, 1)
    y = torch.zeros(batch_size, max_tgt_rows, N_VARIABLES)
    y_mask = torch.zeros_like(y)
    y_mark = torch.zeros(batch_size, max_tgt_rows, 1)
    for i, (to, xo, tt, xt) in enumerate(built):
        m = np.isfinite(xo)
        x[i, :xo.shape[0]][torch.as_tensor(m)] = torch.as_tensor(xo[m])
        x_mask[i, :xo.shape[0]] = torch.as_tensor(m.astype(np.float32))
        x_mark[i, :len(to), 0] = torch.as_tensor(to / L_TOTAL, dtype=torch.float32)
        m2 = np.isfinite(xt)
        y[i, :xt.shape[0]][torch.as_tensor(m2)] = torch.as_tensor(xt[m2])
        y_mask[i, :xt.shape[0]] = torch.as_tensor(m2.astype(np.float32))
        y_mark[i, :len(tt), 0] = torch.as_tensor(tt / L_TOTAL, dtype=torch.float32)

    sys.path.insert(0, str(ROOT / "external" / "PyOmniTS"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "icukan_forecaster", ROOT / "scripts" / "d1_harness" / "ICUKANForecaster.py")

    class FakeConfigs:
        enc_in = N_VARIABLES
        seq_len = 24
        pred_len = 24
        task_name = "short_term_forecast"
        features = "M"

    module = importlib.util.module_from_spec(spec)
    sys.modules["icukan_forecaster"] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise SystemExit(f"cannot import the wrapper without the harness on the path: {exc}")

    model = module.Model(FakeConfigs())
    batch = model.build_batch(x, x_mark, x_mask, y, y_mark, y_mask)

    failures = []
    for i in range(batch_size):
        ref_n = int(reference["obs_mask"][i].sum())
        got_n = int(batch["obs_mask"][i].sum())
        if ref_n != got_n:
            failures.append(f"sample {i}: {got_n} observations, reference has {ref_n}")
            continue
        ref = sorted(zip(reference["obs_var"][i][:ref_n].tolist(),
                         np.round(reference["obs_value"][i][:ref_n].numpy(), 5).tolist(),
                         np.round(reference["obs_delta_t"][i][:ref_n].numpy(), 4).tolist()))
        got = sorted(zip(batch["obs_var"][i][:got_n].tolist(),
                         np.round(batch["obs_value"][i][:got_n].numpy(), 5).tolist(),
                         np.round(batch["obs_delta_t"][i][:got_n].numpy(), 4).tolist()))
        if ref != got:
            mismatch = [(a, b) for a, b in zip(ref, got) if a != b][:3]
            failures.append(f"sample {i}: observation triples differ, e.g. {mismatch}")

        ref_t = int(reference["target_var"][i].numel())
        ref_tn = int((reference.get("target_loss_mask")[i]).numel())
        n_t = int(batch["target_var"][i].shape[0])
        rx = sorted(zip(reference["target_var"][i].tolist(),
                        np.round(reference["x_last"][i].numpy(), 5).tolist()))
        gx = sorted(zip(batch["target_var"][i].tolist(),
                        np.round(batch["x_last"][i].numpy(), 5).tolist()))
        if len(rx) == len(gx) and rx != gx:
            bad = [(a, b) for a, b in zip(rx, gx) if a != b][:3]
            failures.append(f"sample {i}: carry-forward differs, e.g. {bad}")

    if failures:
        print("GATE D3b FAILED")
        for f in failures[:10]:
            print("  " + f)
        raise SystemExit(1)
    print(f"GATE D3b PASSED on {batch_size} samples: observation triples and "
          f"carry-forward anchors are identical to the project's own batch")


if __name__ == "__main__":
    main()
