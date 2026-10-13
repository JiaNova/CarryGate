"""Planner behaviour for P1 padded-batch splitting (ISSUE-023, ISSUE-024).

The default must stay exactly as frozen: a single sample above the budget is
refused. With ``allow_oversized_singletons`` that sample becomes its own piece
(the trainer then offloads it), and nothing else about the partition changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "jbi"))

from p1_padded_batch_split import (  # noqa: E402
    SingleSamplePaddedSizeExceeded,
    plan_padded_batch,
    slice_padded_batch,
)


def _batch(rows: list[int], sample_ids: list[int] | None = None) -> dict[str, torch.Tensor]:
    """One variable per sample; sample i has rows[i] valid input and outcome rows."""
    t = max(rows)
    b = len(rows)
    x_mask = torch.zeros(b, t, 1)
    y_mask = torch.zeros(b, t, 1)
    for i, n in enumerate(rows):
        x_mask[i, :n] = 1
        y_mask[i, :n] = 1
    marks = torch.arange(t, dtype=torch.float32).repeat(b, 1).unsqueeze(-1)
    ids = sample_ids if sample_ids is not None else list(range(100, 100 + b))
    return {
        "x": torch.arange(b * t, dtype=torch.float32).reshape(b, t, 1),
        "y": torch.arange(b * t, dtype=torch.float32).reshape(b, t, 1),
        "x_mark": marks, "y_mark": marks.clone(),
        "x_mask": x_mask, "y_mask": y_mask,
        "sample_ID": torch.tensor(ids),
    }


# Per-sample padded size is (rows + rows) * (2 * rows): 2 rows -> 16, 20 rows -> 1600.
SMALL, DENSE = 2, 20


def test_default_refuses_an_oversized_single_sample():
    batch = _batch([SMALL, SMALL, DENSE, SMALL])
    with pytest.raises(SingleSamplePaddedSizeExceeded):
        plan_padded_batch(batch, budget=200)


def test_oversized_single_sample_is_isolated_when_allowed():
    batch = _batch([SMALL, SMALL, DENSE, SMALL])
    plan = plan_padded_batch(batch, budget=200, allow_oversized_singletons=True)
    assert plan.was_split
    assert (2,) in plan.piece_positions
    flattened = [p for piece in plan.piece_positions for p in piece]
    assert flattened == [0, 1, 2, 3]
    for piece, size in zip(plan.piece_positions, plan.piece_padded_sizes):
        assert size <= 200 or piece == (2,)


def test_option_does_not_change_plans_that_already_fit():
    batch = _batch([SMALL, 6, SMALL, 8, SMALL, SMALL])
    for budget in (10_000, 600, 300):
        a = plan_padded_batch(batch, budget=budget)
        b = plan_padded_batch(batch, budget=budget, allow_oversized_singletons=True)
        assert a == b


def test_rescue_budget_partitions_the_same_samples_in_order():
    """The OOM rescue re-plans at budget//4; it must cover every sample once, in order."""
    batch = _batch([SMALL, 6, DENSE, 8, SMALL, 6, SMALL])
    plan = plan_padded_batch(batch, budget=1600 // 4, allow_oversized_singletons=True)
    flattened = [p for piece in plan.piece_positions for p in piece]
    assert flattened == list(range(7))
    assert plan.original_sample_ids == tuple(range(100, 107))


def test_piece_selection_preserves_both_collated_time_axes():
    batch = _batch([2, 3, 5])
    positions = (0, 2)
    piece = slice_padded_batch(batch, positions)
    index = torch.tensor(positions)
    for key in ("x", "x_mark", "x_mask", "y", "y_mark", "y_mask"):
        assert piece[key].shape[1:] == batch[key].shape[1:]
        assert torch.equal(piece[key], batch[key].index_select(0, index))

    plan = plan_padded_batch(batch, budget=100)
    assert plan.original_padded_size == 3 * (5 + 5) * 10
    assert plan.piece_padded_sizes == (40, 60, 100)
