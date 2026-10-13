# CarryGate

CarryGate, implemented for the external-baseline harness as `ICUKANForecaster`, forecasts future ICU measurements from irregular, partially observed time series. It combines a triplet-based time-series encoder with a recency-conditioned gate and a carry-forward reference, so the model can use both the observed value and how long that value has been stale.

This repository contains analysis and training code only. It contains no clinical data, trained weights, predictions, metrics, figures, manuscript files, or external baseline source tree. Generated files belong under the ignored `data/`, `results/`, `checkpoints/`, and `external/` directories.

## Data access

Obtain the source data yourself, complete the required credentialing and data-use agreements, and do not redistribute the restricted datasets:

- PhysioNet Challenge 2012, version 1.0.0: [dataset page](https://physionet.org/content/challenge-2012/1.0.0/)
- MIMIC-IV, version 2.2: [dataset page](https://physionet.org/content/mimiciv/2.2/)
- eICU-CRD, version 2.0: [dataset page](https://physionet.org/content/eicu-crd/2.0/)

Keep downloaded files locally under `data/raw/physionet2012/`, `data/raw/mimiciv/2.2/`, and `data/raw/eicu-crd/2.0/`. The PhysioNet downloader writes to `data/raw/` by default. The MIMIC-IV and eICU extractors query local PostgreSQL databases; load the credentialed source tables into those databases and set `MIMICIV_DATABASE_URL` and `EICU_PG_DSN` in your shell. Do not put database credentials in tracked files.

The paper supplement gives SHA256 digests of selected stay identifiers. Compare those digests with the corresponding locally generated extraction manifest to verify that your extraction matches. Keep all identifiers and manifests local.

## Environment

Create the pinned environment and activate it:

```bash
conda env create -f environment.yml
conda activate icu_kan
```

`environment.yml` records Python 3.10.20 and the project package versions. `requirements.txt` provides the corresponding pip requirements, including the pinned PyTorch CUDA 12.4 wheel.

## Rebuild the external-baseline harness

The upstream PyOmniTS source is not vendored. Rebuild the harness at its recorded commit and apply the project-owned patch and adapters:

```bash
bash scripts/d1_harness/setup_harness.sh
```

The script creates `external/PyOmniTS/`, which is ignored by Git. Set `HARNESS_PYTHON` if the active `python` is not the interpreter to use. The repository URL, commit, and upstream license are recorded in [THIRD_PARTY.md](THIRD_PARTY.md).

## End-to-end order

1. **Extract data.** Download PhysioNet-2012 and build its event cache. Load the credentialed MIMIC-IV and eICU tables into local PostgreSQL, then run their extractors. MIMIC-IV extraction produces the standardizer and reference ranges used by the eICU pipeline.

   ```bash
   python src/data/p12_download.py --out data/raw --sets a,b,c
   python -m src.data.p12_events --raw data/raw --processed-dir data/processed --sets a,b,c
   python -m src.data.mimiciv_extract
   python -m src.data.eicu_extract
   python scripts/prepare_eicu_loho.py
   ```

   Use `--help` on the extractors to select local input, output, and database options. The extractors and cohort preparation run on CPU.

2. **Build cohorts and patient-grouped splits.** Run `scripts/jbi/00_build_patient_splits.py` for grouped splits, `scripts/jbi/39_extract_landmark_events.py` for the landmark event caches, `scripts/jbi/43_prepare_p2_loho.py` for P2 folds, and `scripts/run_eicu_loho.py` for the eICU leave-one-hospital-group-out cohort. Run `scripts/jbi/40_check_landmark_model_queries.py` before training to check the adapter/model query contract.

3. **Train.** Rebuild the PyOmniTS harness first. Use `scripts/jbi/42_train_p1_landmark.py` for P1, `scripts/jbi/52_train_p2_loho.py` for P2, `scripts/jbi/07_run_patient_em1_k16.py` for the patient-grouped EM1-K16 analysis, and `scripts/d1_harness/run_d1c.py` for external baselines. The training entry points show their required cohort, model, seed, and device arguments with `--help`.

4. **Evaluate.** Run the corresponding one-pass evaluation entry points, `scripts/jbi/51_run_p1_test_once.py` and `scripts/jbi/53_run_p2_test_once.py`, then run `scripts/run_eicu_loho.py` with the selected cohort configuration. These commands write protected predictions and evaluation products locally; none are in this repository.

5. **Run post hoc analyses.** Follow the ordered commands in [REPRODUCE.md](REPRODUCE.md) after the local training and evaluation outputs exist.

6. **Build figures and tables.** Run the figure and table commands in [REPRODUCE.md](REPRODUCE.md) last. They write generated assets to ignored local output directories.

P1/P2 training and some plan-bound sensitivity analyses check author-held internal provenance records. Those records are intentionally not distributed; keep them local and outside Git. The provenance table in `REPRODUCE.md` identifies them as internal records, not public registrations.

The R2 sensitivity analysis needs the exact historical P1 plan bytes named by the local inference ledger. Set `ICU_KAN_P1_PLAN_SOURCE` to that local file when running it from this new-history repository. The GCS correction table also compares against historical aggregate CSVs that are not distributed; set `ICU_KAN_GCS_HISTORY_ROOT` to a local archived checkout or directory containing the historical `results/` tree. Both scripts retain their original Git-history lookup when these variables are unset.

R7 also validates author-held provenance and failed-attempt records that are not included here. Set `ICU_KAN_R7_P1_FAILED_ATTEMPT` to the local P1 failed-attempt directory and `ICU_KAN_R7_FAILED_ATTEMPT_ARCHIVE` to the local R7 archive manifest before running that analysis.

## Hardware

Data extraction, cohort and split preparation, post hoc statistics, and figure/table generation run on CPU. Training CarryGate and the external baselines requires a CUDA-capable GPU for the paper-scale runs. The microbatch validation and GPU profiling helpers also require the rebuilt harness and a GPU. The test suite runs on CPU; the adapter tests skip until the harness is built, and tests that read protected caches or generated results skip until those local files exist.

Run tests with:

```bash
pytest -q
```
