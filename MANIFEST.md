# Release file manifest

Each row names one shipped file and its role in the reproduction dependency trace.

| File | Reason included |
|---|---|
| `MANIFEST.md` | Inventories every shipped file and records source files excluded from this code-only release. |
| `.gitignore` | Blocks accidental commits of restricted data, generated analyses, checkpoints and caches. |
| `README.md` | Release entry point: data access, environment, harness rebuild and ordered reproduction workflow. |
| `REPRODUCE.md` | Paper reproduction guide for provenance, post hoc analyses, figures and tables. |
| `THIRD_PARTY.md` | Records the upstream baseline repository, pinned commit, license and local patch workflow. |
| `configs/eicu_jbi_patient_m1k16.yaml` | Fixed eICU patient-grouped model settings consumed by the JBI training/evaluation entry points (MAE/MSE). |
| `configs/eicu_jbi_patient_m1k16_native_mse.yaml` | Fixed eICU patient-grouped model settings consumed by the JBI training/evaluation entry points (native MSE). |
| `configs/eicu_loho.yaml` | Loaded by scripts/run_eicu_loho.py for the eICU grouped-hospital evaluation. |
| `configs/eicu_variable_map.yaml` | Supplies eICU source schema and variable mappings to the extractor and landmark cohort pipeline; unused generated event-count aggregates were removed. |
| `configs/mimiciv_jbi_patient_m1k16.yaml` | Fixed MIMIC-IV patient-grouped model settings consumed by the JBI training/evaluation entry points (MAE/MSE). |
| `configs/mimiciv_jbi_patient_m1k16_native_mse.yaml` | Fixed MIMIC-IV patient-grouped model settings consumed by the JBI training/evaluation entry points (native MSE). |
| `configs/mimiciv_variable_map.yaml` | Supplies MIMIC-IV source schema and variable mappings to the extractor and landmark cohort pipeline; unused generated event-count aggregates were removed. |
| `environment.yml` | Pinned conda environment copied from the submission environment specification. |
| `pytest.ini` | Configures pytest discovery for the shipped tests. |
| `requirements.txt` | Pinned pip dependencies used by the data, model, training and analysis entry points. |
| `scripts/analyze_sofa_grades.py` | Computes the SOFA grading outputs consumed by the GCS correction and paper analyses. |
| `scripts/d1_harness/ICUKAN.py` | Project-owned dataset adapter copied into the pinned PyOmniTS harness by setup_harness.sh. |
| `scripts/d1_harness/ICUKANForecaster.py` | CarryGate forecaster used by the external-baseline training and evaluation entry point. |
| `scripts/d1_harness/ICUKAN_Naive.py` | Carry-forward comparator model installed in the rebuilt PyOmniTS harness. |
| `scripts/d1_harness/apply_task_aligned_patch.py` | Applies the project-owned task-alignment patch to the pinned upstream harness. |
| `scripts/d1_harness/check_adapter_fidelity.py` | Checks the project adapter against the pinned upstream data-provider behavior. |
| `scripts/d1_harness/check_forecaster_fidelity.py` | Checks CarryGate forecaster behavior within the external-baseline harness. |
| `scripts/d1_harness/collect_d1c.py` | Aggregates external-baseline outputs for the paper tables and figures. |
| `scripts/d1_harness/rescore_macro.py` | Recomputes macro-level external-baseline scores used by the comparison analyses. |
| `scripts/d1_harness/run_d1c.py` | Runs the paper external-baseline training/evaluation harness. |
| `scripts/d1_harness/score_run.py` | Scores a local external-baseline run for run_d1c.py and collection utilities. |
| `scripts/d1_harness/setup_harness.sh` | Clones PyOmniTS at the recorded commit and installs the project-owned patch and adapters. |
| `scripts/figures/fig_capacity.py` | Figure generator imported by scripts/figures/make_all.py for the paper asset build (fig_capacity). |
| `scripts/figures/fig_clinical.py` | Figure generator imported by scripts/figures/make_all.py for the paper asset build (fig_clinical). |
| `scripts/figures/fig_external.py` | Figure generator imported by scripts/figures/make_all.py for the paper asset build (fig_external). |
| `scripts/figures/fig_overview.py` | Figure generator imported by scripts/figures/make_all.py for the paper asset build (fig_overview). |
| `scripts/figures/fig_per_variable.py` | Figure generator imported by scripts/figures/make_all.py for the paper asset build (fig_per_variable). |
| `scripts/figures/fig_protocol.py` | Figure generator imported by scripts/figures/make_all.py for the paper asset build (fig_protocol). |
| `scripts/figures/make_all.py` | Top-level generator for the paper figure set, using the included figure modules. |
| `scripts/figures/style.py` | Shared plotting style and model labels imported by all paper figure generators. |
| `scripts/jbi/00_build_patient_splits.py` | Builds deterministic patient-grouped cohort splits used before model training. |
| `scripts/jbi/04_analyze_patient_sofa_components.py` | Computes patient-level SOFA component summaries used in the paper analyses. |
| `scripts/jbi/05_analyze_grade_transitions.py` | Computes grade-transition analyses and supplies local utilities used by downstream JBI scripts. |
| `scripts/jbi/07_run_patient_em1_k16.py` | Runs the patient-grouped EM1-K16 training/evaluation analysis described in README.md. |
| `scripts/jbi/14_analyze_patient_external_sofa.py` | Produces patient-level external SOFA comparisons used by the analysis and figure pipeline. |
| `scripts/jbi/18_plot_paired_aggregation.py` | Builds the paired aggregation plot from locally generated paper analysis outputs. |
| `scripts/jbi/23_adapter_vs_external_paired.py` | Computes paired adapter-versus-external baseline comparisons used by paper analyses. |
| `scripts/jbi/30_deterioration_decision_frontier.py` | Generates the deterioration decision-frontier analysis and table/figure outputs. |
| `scripts/jbi/32_make_chil_main_figures.py` | Builds the paper main figure assets from local analysis outputs. |
| `scripts/jbi/33_make_aligned_table.py` | Generates the aligned results table used in the paper asset workflow. |
| `scripts/jbi/35_gcs_correction_table.py` | Builds the GCS correction table; historical aggregate inputs can be supplied via ICU_KAN_GCS_HISTORY_ROOT. |
| `scripts/jbi/36_d1a_reproduction_table.py` | Generates the external-baseline reproduction table from local run aggregates. |
| `scripts/jbi/39_extract_landmark_events.py` | Builds landmark event caches consumed by P1 training, evaluation and sensitivity analyses. |
| `scripts/jbi/40_check_landmark_model_queries.py` | Validates the landmark dataset/model query contract before P1 training. |
| `scripts/jbi/41_landmark_use_case.py` | Implements P1 outcome, metric and decision-curve calculations imported by training and post hoc scripts. |
| `scripts/jbi/42_train_p1_landmark.py` | P1 patient-grouped CarryGate training entry point and provenance gate. |
| `scripts/jbi/43_prepare_p2_loho.py` | Constructs P2 leave-hospital-group-out folds used by P2 training and evaluation. |
| `scripts/jbi/44_stratify_existing_trackb.py` | Prepares grouped Track B strata consumed by patient-level and external-comparison analyses. |
| `scripts/jbi/47_verify_p1_microbatch_gradients.py` | Verifies microbatch gradient equivalence required by the P1 training gate. |
| `scripts/jbi/48_profile_p1_microbatch_memory.py` | Profiles the P1 microbatch path used by the training memory plan. |
| `scripts/jbi/50_p1_trigger_diagnostics.py` | Computes P1 trigger diagnostics used by the paper post hoc analysis. |
| `scripts/jbi/51_run_p1_test_once.py` | Runs the frozen one-pass P1 test evaluation after the training evidence checks. |
| `scripts/jbi/52_train_p2_loho.py` | P2 grouped hospital-fold training entry point. |
| `scripts/jbi/53_run_p2_test_once.py` | Runs the frozen one-pass P2 test evaluation after the checkpoint audit. |
| `scripts/jbi/54_grading_gap_decomposition.py` | Generates the grading-gap decomposition table/figure from local evaluation outputs. |
| `scripts/jbi/56_verify_p2_microbatch.py` | Verifies P2 microbatch behavior required by the P2 training path. |
| `scripts/jbi/57_make_p2_assets.py` | Builds P2 paper figure and table assets. |
| `scripts/jbi/58_ascertainment_tipping_point.py` | Runs the R2 post hoc sensitivity analysis; its exact historical plan is a local input. |
| `scripts/jbi/59_objective_metric_matrix.py` | Runs the R4 objective/metric sensitivity analysis. |
| `scripts/jbi/60_variable_uncertainty_threshold_sensitivity.py` | Runs the R5 variable uncertainty and threshold sensitivity analysis. |
| `scripts/jbi/61_hospital_heterogeneity.py` | Runs the R6 grouped hospital heterogeneity analysis. |
| `scripts/jbi/62_simple_trend_warning_baseline.py` | Runs the R7 simple-trend comparator on local P1 outputs and accepts local failed-attempt provenance paths through environment variables. |
| `scripts/jbi/63_redraw_s6_sensitivity_burden.py` | Redraws Supplementary Figure S6 from local sensitivity-analysis outputs. |
| `scripts/jbi/67_package_jamia_figures.py` | Packages the final figure/table outputs for the JAMIA asset workflow. |
| `scripts/jbi/output_safety.py` | Shared write-safety helpers imported by included paper-asset generators. |
| `scripts/jbi/p1_microbatch_training.py` | P1 microbatch trainer imported by the P1 training and verification entry points. |
| `scripts/jbi/p1_padded_batch_split.py` | P1 padded-batch splitter imported by training and gradient validation. |
| `scripts/jbi/p2_plan_integrity.py` | P2 internal-plan integrity checks imported by P2 training and evaluation. |
| `scripts/make_d1d2_assets.py` | Builds D1/D2 external-baseline paper assets from local outputs. |
| `scripts/make_paper_assets.py` | Builds manuscript-independent paper tables/assets from local analysis outputs. |
| `scripts/prepare_eicu_loho.py` | Prepares the patient-grouped eICU leave-hospital-group-out cohort before evaluation. |
| `scripts/run_eicu_loho.py` | Runs the eICU leave-hospital-group-out training/evaluation analysis. |
| `src/__init__.py` | Python package initializer required for imports in the traced source closure. |
| `src/data/__init__.py` | Data-pipeline implementation or local import dependency for extraction, event construction and persistence baselines. |
| `src/data/eicu_extract.py` | Data-pipeline implementation or local import dependency for extraction, event construction and persistence baselines. |
| `src/data/mimiciv_extract.py` | Data-pipeline implementation or local import dependency for extraction, event construction and persistence baselines. |
| `src/data/p12_dataset.py` | Data-pipeline implementation or local import dependency for extraction, event construction and persistence baselines. |
| `src/data/p12_download.py` | Data-pipeline implementation or local import dependency for extraction, event construction and persistence baselines. |
| `src/data/p12_events.py` | Data-pipeline implementation or local import dependency for extraction, event construction and persistence baselines. |
| `src/data/p12_forecast_dataset.py` | Data-pipeline implementation or local import dependency for extraction, event construction and persistence baselines. |
| `src/data/p12_hourly_agg.py` | Data-pipeline implementation or local import dependency for extraction, event construction and persistence baselines. |
| `src/data/p12_parse.py` | Data-pipeline implementation or local import dependency for extraction, event construction and persistence baselines. |
| `src/data/persistence_baseline.py` | Data-pipeline implementation or local import dependency for extraction, event construction and persistence baselines. |
| `src/data/variable_descriptors.py` | Data-pipeline implementation or local import dependency for extraction, event construction and persistence baselines. |
| `src/models/__init__.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/backbones/__init__.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/backbones/gru_backbone.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/backbones/strats_lite.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/baselines/__init__.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/baselines/naive.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/encoders/__init__.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/encoders/functional_event_encoder.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/encoders/shared_lowrank.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/gates/__init__.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/gates/capacity_sweep.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/gates/frozen_gate.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/gates/gate_bases.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/gates/monotone_gate.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/gates/recency_gates.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/gates/statistic_conditioned_gate.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/heads/__init__.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/heads/conditioned_residual_head.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/heads/forecast_head.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/heads/persistence_residual_head.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/icu_model.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/layers/__init__.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/layers/fourier_kan_layer.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/layers/grouped_kan.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/layers/kan_bspline_head.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/layers/mlp_head.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/layers/rbf_kan_head.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/models/layers/rbf_kan_layer.py` | Model component in the traced CarryGate/PyOmniTS forecaster and training dependency closure. |
| `src/training/__init__.py` | Training/evaluation component imported by the P1/P2 and CarryGate analysis entry points. |
| `src/training/bootstrap.py` | Training/evaluation component imported by the P1/P2 and CarryGate analysis entry points. |
| `src/training/efficiency.py` | Training/evaluation component imported by the P1/P2 and CarryGate analysis entry points. |
| `src/training/evaluate.py` | Training/evaluation component imported by the P1/P2 and CarryGate analysis entry points. |
| `src/training/forecast_loss.py` | Training/evaluation component imported by the P1/P2 and CarryGate analysis entry points. |
| `src/training/forecast_train.py` | Training/evaluation component imported by the P1/P2 and CarryGate analysis entry points. |
| `src/training/grid_calibration.py` | Training/evaluation component imported by the P1/P2 and CarryGate analysis entry points. |
| `src/training/param_matching.py` | Training/evaluation component imported by the P1/P2 and CarryGate analysis entry points. |
| `src/training/train_loop.py` | Training/evaluation component imported by the P1/P2 and CarryGate analysis entry points. |
| `src/utils/__init__.py` | Utility in the traced local-import closure for training, evaluation and paper analyses. |
| `src/utils/forecast_metrics.py` | Utility in the traced local-import closure for training, evaluation and paper analyses. |
| `src/utils/metrics.py` | Utility in the traced local-import closure for training, evaluation and paper analyses. |
| `src/utils/provenance.py` | Utility in the traced local-import closure for training, evaluation and paper analyses. |
| `src/utils/seed_utils.py` | Utility in the traced local-import closure for training, evaluation and paper analyses. |
| `src/utils/skill_metrics.py` | Utility in the traced local-import closure for training, evaluation and paper analyses. |
| `tests/test_bootstrap_ci.py` | Unit or regression coverage for the included bootstrap ci implementation and analysis code. |
| `tests/test_conditioned_residual_head.py` | Unit or regression coverage for the included conditioned residual head implementation and analysis code. |
| `tests/test_decision_curve.py` | Unit or regression coverage for the included decision curve implementation and analysis code. |
| `tests/test_determinism.py` | Unit or regression coverage for the included determinism implementation and analysis code. |
| `tests/test_eicu_extract.py` | Unit or regression coverage for the included eicu extract implementation and analysis code. |
| `tests/test_forecast_metrics.py` | Unit or regression coverage for the included forecast metrics implementation and analysis code. |
| `tests/test_fourier_kan.py` | Unit or regression coverage for the included fourier kan implementation and analysis code. |
| `tests/test_gate_bases.py` | Unit or regression coverage for the included gate bases implementation and analysis code. |
| `tests/test_grid_calibration.py` | Unit or regression coverage for the included grid calibration implementation and analysis code. |
| `tests/test_grouped_determinism.py` | Unit or regression coverage for the included grouped determinism implementation and analysis code. |
| `tests/test_kan_layer_shapes.py` | Unit or regression coverage for the included kan layer shapes implementation and analysis code. |
| `tests/test_landmark_adapter.py` | Unit or regression coverage for the included landmark adapter implementation and analysis code. |
| `tests/test_mimiciv_extract.py` | Unit or regression coverage for the included mimiciv extract implementation and analysis code. |
| `tests/test_monotone_gate.py` | Unit or regression coverage for the included monotone gate implementation and analysis code. |
| `tests/test_naive_baselines.py` | Unit or regression coverage for the included naive baselines implementation and analysis code. |
| `tests/test_p12_events.py` | Unit or regression coverage for the included p12 events implementation and analysis code. |
| `tests/test_p12_preprocessing.py` | Unit or regression coverage for the included p12 preprocessing implementation and analysis code. |
| `tests/test_p1_microbatch_planning.py` | Unit or regression coverage for the included p1 microbatch planning implementation and analysis code. |
| `tests/test_persistence_baseline.py` | Unit or regression coverage for the included persistence baseline implementation and analysis code. |
| `tests/test_residual_head_init.py` | Unit or regression coverage for the included residual head init implementation and analysis code. |
| `tests/test_shared_lowrank.py` | Unit or regression coverage for the included shared lowrank implementation and analysis code. |
| `tests/test_skill_metrics.py` | Unit or regression coverage for the included skill metrics implementation and analysis code. |
| `tests/test_sofa_truth_projection.py` | Unit or regression coverage for the included sofa truth projection implementation and analysis code. |
| `tests/test_statistic_gate_init.py` | Unit or regression coverage for the included statistic gate init implementation and analysis code. |
| `tests/test_strats_lite_shapes.py` | Unit or regression coverage for the included strats lite shapes implementation and analysis code. |
| `tests/test_variable_descriptors.py` | Unit or regression coverage for the included variable descriptors implementation and analysis code. |

## Excluded source categories

Counts below are tracked file paths in the source ICU-KAN Git tree (`git ls-files`), grouped by the category that took precedence in this table. Untracked local files are not counted. None of these files were copied.

| Excluded category | Tracked files |
|---|---:|
| Protected data/ extraction outputs | 2 |
| results/ files | 6577 |
| external/ third-party source | 2 |
| Slides and slide projects | 115 |
| Manuscript and submission files | 65 |
| Plans, docs, logs, work/archive files | 86 |
| Phase/stage/report scripts | 30 |
| Other scripts outside the reproduction closure | 73 |

The optional manuscript-linked claim audit and its registry builder are also omitted because the former requires the withheld manuscript and the latter is not needed by the released reproduction workflow. No `LICENSE` file is included; the authors can choose the project license.

## Source workspace file counts

The following counts include tracked, ignored and untracked files currently present in the source workspace. Each row counts files under the named directory; all of these directories were omitted from the export. `results/` includes its slide assets and `paper/` includes slide projects.

| Excluded directory | Files present |
|---|---:|
| `data/` | 12030 |
| `results/` | 8149 |
| `external/` | 7237 |
| `paper/` | 221 |
| `docs/` | 11 |
| `logs/` | 49 |
| `scripts/slides/` | 12 |
