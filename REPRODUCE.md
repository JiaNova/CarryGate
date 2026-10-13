# Reproducing the analyses and paper assets

This guide describes how to regenerate analyses and figures from locally obtained data and locally produced outputs. The repository distributes code only: it contains no raw or processed data, split indices, event caches, predictions, labels, checkpoints, metrics, figures, or manuscript files. Every generated artifact remains in ignored local directories.

## Data and provenance

Use PhysioNet Challenge 2012 version 1.0.0, MIMIC-IV version 2.2, and eICU-CRD version 2.0. MIMIC-IV and eICU-CRD require credentialing, training, and data-use agreements. Keep source data under `data/raw/`; load MIMIC-IV and eICU-CRD into local PostgreSQL databases for extraction. See `README.md` for the data locations and environment setup.

The supplement reports SHA256 digests of the selected stay identifiers. Compare them with the identifiers from your local extraction to check that the cohort extraction matches. Do not publish identifier lists or extraction manifests.

### Plan provenance

The records below are internal planning and execution records. They are not included in this release and do not constitute public prospective registrations. The commit values identify their historical internal provenance; this repository has a new, single-commit history.

| Analysis | Internal record | Historical commit |
|---|---|---|
| P1 patient-grouped landmark evaluation | P1 analysis-plan record | `05e2590` |
| P2 leave-hospitals-out evaluation | P2 analysis-plan record | `28947c8` |
| R2 and R4–R7 sensitivity/reporting rules | Revision sensitivity-plan record | `3689926` |
| P1 execution deviation | P1 execution-deviation record | `888e52d` |
| P2 test-inference memory deviation | P2 execution-deviation record | `8370d5c` |

The P1/P2 entry points verify locally supplied plan records before training. Some post hoc scripts also read the local plan record and its generated registration manifest. Keep those files outside version control; the internal records themselves are not part of the public release.

## Environment and external baseline

```bash
conda env create -f environment.yml
conda activate icu_kan
bash scripts/d1_harness/setup_harness.sh
```

The harness script checks out the PyOmniTS commit listed in `THIRD_PARTY.md`, applies the project-owned patch, and copies only the project adapters into the ignored `external/` directory.

## Rebuild order

Run commands from the repository root. First extract data and construct cohorts and patient-grouped splits as shown in `README.md`. Then train and evaluate the P1, P2, and external-baseline models. Training and test-evaluation entry points expose their fixed roster and required arguments through `--help`:

```bash
python scripts/jbi/42_train_p1_landmark.py --help
python scripts/jbi/51_run_p1_test_once.py --help
python scripts/jbi/52_train_p2_loho.py --help
python scripts/jbi/53_run_p2_test_once.py --help
python scripts/d1_harness/run_d1c.py --help
```

Run the post hoc analyses after the relevant locally generated evaluation products exist. The preflight commands validate inputs before calculation.

### Post hoc analyses

R2:

```bash
export ICU_KAN_P1_PLAN_SOURCE=local-private-copy/p1-analysis-plan.md
python scripts/jbi/58_ascertainment_tipping_point.py --refresh-generated
```

`ICU_KAN_P1_PLAN_SOURCE` must point to the exact internal plan bytes whose SHA256 is recorded in the local inference ledger. The plan itself is not distributed.

R4, R5, and R6 each require a successful preflight:

```bash
python scripts/jbi/59_objective_metric_matrix.py --preflight-only --refresh-generated
python scripts/jbi/59_objective_metric_matrix.py --run --refresh-generated

python scripts/jbi/60_variable_uncertainty_threshold_sensitivity.py --preflight-only --refresh-generated
python scripts/jbi/60_variable_uncertainty_threshold_sensitivity.py --run --refresh-generated

python scripts/jbi/61_hospital_heterogeneity.py --preflight-only --refresh-generated
python scripts/jbi/61_hospital_heterogeneity.py --run --refresh-generated
```

R7 uses the fixed P1 windows. It is a descriptive comparator and does not alter I1 or I2:

```bash
export ICU_KAN_R7_P1_FAILED_ATTEMPT=local-private-copy/p1-failed-attempt
export ICU_KAN_R7_FAILED_ATTEMPT_ARCHIVE=local-private-copy/r7-archive-manifest.json
python scripts/jbi/62_simple_trend_warning_baseline.py preflight --refresh-generated
python scripts/jbi/62_simple_trend_warning_baseline.py run --refresh-generated
```

R7 also needs the author-held revision/P1 plan documents and local P1 provenance outputs listed by its preflight. The two environment variables select the failed-attempt records without assuming the original server's run directory.

R3's analytic checks are in `tests/test_decision_curve.py` and `tests/test_sofa_truth_projection.py`. Checks that read locally generated aggregate results skip when those files are absent.

Supplementary Figure S6 is redrawn from locally generated P1 metrics:

```bash
python scripts/jbi/63_redraw_s6_sensitivity_burden.py --refresh-generated
```

### Figures and tables

After the analysis outputs exist, generate the figure and table families:

```bash
python scripts/figures/make_all.py --refresh-generated
python scripts/jbi/18_plot_paired_aggregation.py
python scripts/make_paper_assets.py --tables-only --refresh-generated
python scripts/make_d1d2_assets.py --refresh-generated
python scripts/jbi/30_deterioration_decision_frontier.py --refresh-generated
python scripts/jbi/32_make_chil_main_figures.py --refresh-generated
python scripts/jbi/33_make_aligned_table.py --refresh-generated
python scripts/jbi/35_gcs_correction_table.py --refresh-generated
python scripts/jbi/36_d1a_reproduction_table.py --refresh-generated
python scripts/jbi/54_grading_gap_decomposition.py --refresh-generated
python scripts/jbi/57_make_p2_assets.py --refresh-generated
python scripts/jbi/67_package_jamia_figures.py --refresh-generated
```

The GCS correction table (`35_gcs_correction_table.py`) also reads historical aggregate CSVs that are absent from this clean history. Set `ICU_KAN_GCS_HISTORY_ROOT` to a local archived checkout or directory containing the historical `results/` tree before running that command. The script verifies the expected row counts and does not use stay-level inputs.

These commands write only to local ignored output directories. The manuscript-linked claim audit, word count, and clinical-review packet are outside this code-only release because their manuscript and review materials are not distributed.

## Tests

```bash
pytest -q
```

Tests that need the locally rebuilt PyOmniTS harness skip with a reason until it is installed. Tests that read protected event caches or generated aggregate outputs skip when those local inputs are absent. The synthetic unit tests run without protected data.
