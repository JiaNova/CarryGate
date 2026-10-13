"""Collect every D1c run into one table.

Walks the harness result tree, scores each completed run with score_run, and
writes results/d1_external/d1c_matrix.csv -- one row per (model, cohort, seed).

Usage:
    python scripts/d1_harness/collect_d1c.py [--tag d1c]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "d1_harness"))

from score_run import VARIANT_CACHES, score as score_run  # noqa: E402
from rescore_macro import score as score_macro  # noqa: E402

HARNESS = ROOT / "external" / "PyOmniTS" / "storage" / "results" / "ICUKAN"
COHORTS = ("p12", "mimiciv", "eicu")


def _data_provenance(eval_dir: Path, cohort: str) -> dict[str, object]:
    config_path = eval_dir.parent / "configs.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    variant = str((config or {}).get("dataset_subset_name") or "registered").lower()
    if variant not in VARIANT_CACHES:
        raise ValueError(f"unknown data variant in {config_path}: {variant}")
    events_name, split_name = VARIANT_CACHES[variant][cohort]
    cache_path = (ROOT / "data" / "processed" / events_name).resolve()
    split_path = (ROOT / "data" / "processed" / split_name).resolve()
    if not cache_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(f"missing data source: {cache_path} / {split_path}")
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    return {
        "data_source": f"ICU-KAN/{cohort}",
        "data_variant": variant,
        "cache_path": str(cache_path),
        "split_path": str(split_path),
        "split_unit": str(split_payload.get("split_unit", "registered_stay_level")),
        "group_column": split_payload.get("group_column"),
    }


def main(tag: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for cohort in COHORTS:
        base = HARNESS / cohort
        if not base.exists():
            continue
        for eval_dir in sorted(base.glob(f"*/{tag}_{cohort}/24_24/*/iter*/eval_*")):
            if not (eval_dir / "output_pred.npy").exists():
                continue
            model = eval_dir.parts[eval_dir.parts.index(cohort) + 1]
            iteration = int(re.search(r"iter(\d+)", str(eval_dir)).group(1))
            provenance = _data_provenance(eval_dir, cohort)
            try:
                result = score_run(
                    eval_dir, cohort, data_variant=str(provenance["data_variant"])
                )
                macro_result = score_macro(
                    eval_dir, cohort, data_variant=str(provenance["data_variant"])
                )
            except AssertionError as error:
                print(f"[skip] {model}/{cohort}/iter{iteration}: {error}")
                continue
            harness_metric = json.loads((eval_dir / "metric.json").read_text())
            rows.append({
                "model": model,
                "cohort": cohort,
                "seed": 2024 + iteration,
                "iteration": iteration,
                "n_scored_events": result["n_scored_events"],
                "mse_scored": round(result["MSE"], 6),
                "mae_scored": round(result["MAE"], 6),
                "persistence_skill_pooled": round(macro_result["persistence_skill_pooled"], 6),
                "persistence_skill_macro": round(macro_result["persistence_skill_macro"], 6),
                "mse_harness_all_vars": round(harness_metric["MSE"], 6),
                "mae_harness_all_vars": round(harness_metric["MAE"], 6),
                "eval_dir": str(eval_dir.relative_to(ROOT)),
                **provenance,
            })

    if not rows:
        print("no completed runs found")
        return
    rows.sort(key=lambda r: (r["cohort"], r["model"], r["seed"]))
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")

    # Compact per-arm summary for reading at a glance.
    import statistics as st
    seen = {}
    for row in rows:
        seen.setdefault((row["cohort"], row["model"]), []).append(row["persistence_skill_macro"])
    print(f"\n{'cohort':10s} {'model':12s} {'n':>2s}  persistence skill")
    for (cohort, model), skills in sorted(seen.items()):
        spread = f" +- {st.stdev(skills):.4f}" if len(skills) > 1 else ""
        print(f"{cohort:10s} {model:12s} {len(skills):2d}  {st.mean(skills):+.4f}{spread}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="d1c")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "d1_external" / "d1c_matrix.csv",
    )
    args = parser.parse_args()
    main(args.tag, args.out)
