# Third-party code and licenses

| Upstream project | Repository | Exact commit | License at pinned commit | Use |
|---|---|---|---|---|
| PyOmniTS | <https://github.com/Ladbaby/PyOmniTS> | `47d8a2ff774ee12ccae451b3f11c5e778dca3902` | MIT | External-baseline harness, rebuilt locally by `scripts/d1_harness/setup_harness.sh`; no upstream source is vendored here. |

The setup script applies the project-owned patch in `scripts/d1_harness/apply_task_aligned_patch.py` and copies the project-owned adapters from `scripts/d1_harness/`. The cloned harness is placed under the ignored `external/` directory.

Python package versions are pinned in `environment.yml` and `requirements.txt`. No third-party source code is copied into this repository.
