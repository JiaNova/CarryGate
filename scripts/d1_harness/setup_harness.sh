#!/usr/bin/env bash
# Rebuild the D1 external-baseline harness from scratch.
#
# Clones PyOmniTS at the pinned commit, applies the project-owned Track B patch,
# and copies the project-owned adapters into the upstream tree.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$ROOT/external/PyOmniTS"
HARNESS_PYTHON="${HARNESS_PYTHON:-python}"
PIN="${PYOMNITS_COMMIT:-47d8a2ff774ee12ccae451b3f11c5e778dca3902}"

rm -rf "$DEST"
git clone -q https://github.com/Ladbaby/PyOmniTS.git "$DEST"
git -C "$DEST" checkout -q "$PIN"
test "$(git -C "$DEST" rev-parse HEAD)" = "$PIN"
rm -rf "$DEST/.git"

cp "$ROOT/scripts/d1_harness/ICUKAN.py"       "$DEST/data/data_provider/datasets/ICUKAN.py"
cp "$ROOT/scripts/d1_harness/ICUKAN_Naive.py" "$DEST/models/ICUKAN_Naive.py"
cp "$ROOT/scripts/d1_harness/ICUKANForecaster.py" "$DEST/models/ICUKANForecaster.py"
"$HARNESS_PYTHON" "$ROOT/scripts/d1_harness/apply_task_aligned_patch.py" "$DEST"
echo "harness ready at $DEST (pinned $PIN); task-aligned validation patch applied"
