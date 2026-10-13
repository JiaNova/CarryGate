"""Order-preserving P1 microbatch planning for irregular ICU windows.

The DataLoader still creates the frozen batches. This module only partitions
one already-created batch when its padded footprint exceeds a caller-supplied
budget; it never samples, drops, truncates, or reorders examples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor


TIMESTAMP_DENSE_SAMPLE_ID = 53010


class SingleSamplePaddedSizeExceeded(RuntimeError):
    """Raised when one complete sample exceeds the frozen padded-size budget."""

    def __init__(self, sample_id: int, sample_padded_size: int, budget: int):
        self.sample_id = int(sample_id)
        self.sample_padded_size = int(sample_padded_size)
        self.budget = int(budget)
        super().__init__(
            "single complete sample exceeds P1 padded-size budget: "
            f"sample_ID={sample_id}, padded_size={sample_padded_size}, budget={budget}"
        )


@dataclass(frozen=True)
class PaddedBatchPlan:
    original_sample_ids: tuple[int, ...]
    input_time_row_counts: tuple[int, ...]
    outcome_time_row_counts: tuple[int, ...]
    observation_counts: tuple[int, ...]
    original_padded_size: int
    padded_size_budget: int
    dense_sample_position: int
    dense_sample_id: int
    piece_positions: tuple[tuple[int, ...], ...]
    piece_padded_sizes: tuple[int, ...]
    was_split: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_time_channel(mark: Tensor) -> Tensor:
    if mark.ndim == 2:
        return mark
    if mark.ndim == 3 and mark.shape[-1] >= 1:
        return mark[..., 0]
    raise ValueError(f"expected time marks shaped [B,T] or [B,T,C], got {tuple(mark.shape)}")


def _per_sample_window_sizes(
    batch: Mapping[str, Tensor],
) -> tuple[list[int], list[int], list[int]]:
    required = ("x_mark", "x_mask", "y_mark", "y_mask")
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"P1 padded-size calculation is missing fields: {missing}")

    x_mark = _as_time_channel(batch["x_mark"])
    y_mark = _as_time_channel(batch["y_mark"])
    x_mask = batch["x_mask"] > 0
    y_mask = batch["y_mask"] > 0
    batch_size = int(x_mark.shape[0])
    if any(int(batch[key].shape[0]) != batch_size for key in required):
        raise ValueError("P1 batch fields do not share a common sample dimension")

    input_time_row_counts: list[int] = []
    outcome_time_row_counts: list[int] = []
    observation_counts: list[int] = []
    for i in range(batch_size):
        input_rows = x_mask[i].reshape(x_mask.shape[1], -1).any(dim=1)
        outcome_rows = y_mask[i].reshape(y_mask.shape[1], -1).any(dim=1)
        # The adapter pads x and y independently before the model concatenates
        # them. Use actual valid time-axis rows rather than unique float32
        # marks, so repeated same-time measurements occupying separate rows
        # are represented in the padded-size estimate.
        input_time_row_counts.append(max(1, int(input_rows.sum().item())))
        outcome_time_row_counts.append(max(1, int(outcome_rows.sum().item())))
        observation_counts.append(int(x_mask[i].sum().item() + y_mask[i].sum().item()))
    return input_time_row_counts, outcome_time_row_counts, observation_counts


def _size_for_positions(
    positions: list[int] | tuple[int, ...],
    input_time_row_counts: list[int],
    outcome_time_row_counts: list[int],
    observation_counts: list[int],
    *,
    observation_padding_width: int | None = None,
) -> int:
    if not positions:
        return 0
    max_input_rows = max(input_time_row_counts[i] for i in positions)
    max_outcome_rows = max(outcome_time_row_counts[i] for i in positions)
    # HyperIMTS pads every sample's flattened observation sequence to the
    # largest observation count in its original 128-sample batch. Preserve
    # that width across pieces so padding does not change its attention path.
    max_observations = max(
        1,
        int(observation_padding_width)
        if observation_padding_width is not None
        else max(observation_counts[i] for i in positions),
    )
    return len(positions) * (max_input_rows + max_outcome_rows) * max_observations


def _sample_ids(batch: Mapping[str, Tensor]) -> tuple[int, ...]:
    if "sample_ID" not in batch:
        raise KeyError("P1 padded batch splitting requires stable sample_ID values")
    ids = batch["sample_ID"].detach().cpu().reshape(-1).tolist()
    return tuple(int(value) for value in ids)


def plan_padded_batch(
    batch: Mapping[str, Tensor],
    budget: int,
    allow_oversized_singletons: bool = False,
) -> PaddedBatchPlan:
    """Partition a batch into contiguous pieces under a padded-size budget.

    The padded-size proxy mirrors the model's batch tensors:
    ``B_piece * (max valid input rows + max valid outcome rows) *
    max observed measurements in the original full batch``. Input and outcome
    are padded independently by the adapter, so their maxima are summed.
    HyperIMTS receives the original batch's observation-padding width for each
    piece to preserve its attention semantics. Actual event rows are counted,
    including repeated measurements at one timestamp. If
    splitting is needed, sample ID 53010 (the audited 1,394-timestamp window)
    is isolated when present; otherwise the sample with the largest per-window
    (input rows + outcome rows) × observation-count is isolated. The preceding
    and following runs are greedily packed under the same budget.

    With ``allow_oversized_singletons`` a sample that alone exceeds the budget
    becomes a piece of its own instead of raising; the trainer then runs that
    piece with its autograd tensors offloaded to host memory. Order and the
    one-sample-per-position partition are unchanged either way.
    """

    budget = int(budget)
    if budget < 1:
        raise ValueError("padded-size budget must be a positive integer")
    sample_ids = _sample_ids(batch)
    input_time_row_counts, outcome_time_row_counts, observation_counts = _per_sample_window_sizes(batch)
    if not sample_ids or len(sample_ids) != len(input_time_row_counts):
        raise ValueError("empty or inconsistent P1 batch")

    positions = list(range(len(sample_ids)))
    observation_padding_width = max(observation_counts)
    original_size = _size_for_positions(
        positions, input_time_row_counts, outcome_time_row_counts, observation_counts,
        observation_padding_width=observation_padding_width,
    )
    density = [
        (input_time_row_counts[i] + outcome_time_row_counts[i]) * observation_padding_width
        for i in positions
    ]
    # Keep the known 1,394-timestamp MIMIC-IV window on its own whenever its
    # original batch must be split. Other oversized batches isolate the sample
    # with the largest per-window padded footprint.
    dense_position = (
        sample_ids.index(TIMESTAMP_DENSE_SAMPLE_ID)
        if TIMESTAMP_DENSE_SAMPLE_ID in sample_ids
        else max(
            positions,
            key=lambda i: (
                density[i],
                input_time_row_counts[i] + outcome_time_row_counts[i],
                observation_counts[i],
                -i,
            ),
        )
    )
    dense_single_size = _size_for_positions(
        [dense_position], input_time_row_counts, outcome_time_row_counts, observation_counts,
        observation_padding_width=observation_padding_width,
    )
    if dense_single_size > budget and not allow_oversized_singletons:
        raise SingleSamplePaddedSizeExceeded(
            sample_ids[dense_position], dense_single_size, budget
        )

    if original_size <= budget:
        pieces = [tuple(positions)]
    else:
        def greedy_contiguous(run: list[int]) -> list[tuple[int, ...]]:
            packed: list[tuple[int, ...]] = []
            current: list[int] = []
            for position in run:
                candidate = current + [position]
                candidate_size = _size_for_positions(
                    candidate,
                    input_time_row_counts,
                    outcome_time_row_counts,
                    observation_counts,
                    observation_padding_width=observation_padding_width,
                )
                if candidate_size <= budget:
                    current = candidate
                    continue
                if current:
                    packed.append(tuple(current))
                current = [position]
                if _size_for_positions(
                    current,
                    input_time_row_counts,
                    outcome_time_row_counts,
                    observation_counts,
                    observation_padding_width=observation_padding_width,
                ) > budget:
                    if not allow_oversized_singletons:
                        raise SingleSamplePaddedSizeExceeded(
                            sample_ids[position],
                            _size_for_positions(
                                [position], input_time_row_counts, outcome_time_row_counts,
                                observation_counts, observation_padding_width=observation_padding_width,
                            ),
                            budget,
                        )
                    packed.append(tuple(current))
                    current = []
            if current:
                packed.append(tuple(current))
            return packed

        prefix = greedy_contiguous(positions[:dense_position])
        suffix = greedy_contiguous(positions[dense_position + 1 :])
        pieces = prefix + [(dense_position,)] + suffix

    piece_sizes = tuple(
        _size_for_positions(
            piece, input_time_row_counts, outcome_time_row_counts, observation_counts,
            observation_padding_width=observation_padding_width,
        )
        for piece in pieces
    )
    if any(
        size > budget and not (allow_oversized_singletons and len(piece) == 1)
        for piece, size in zip(pieces, piece_sizes)
    ):
        raise AssertionError("P1 batch partition contains a multi-sample piece above its budget")
    flattened = tuple(position for piece in pieces for position in piece)
    if flattened != tuple(positions):
        raise AssertionError("P1 batch partition changed the original sample order")
    if original_size > budget and pieces.count((dense_position,)) != 1:
        raise AssertionError("the densest P1 window was not isolated exactly once")

    return PaddedBatchPlan(
        original_sample_ids=sample_ids,
        input_time_row_counts=tuple(input_time_row_counts),
        outcome_time_row_counts=tuple(outcome_time_row_counts),
        observation_counts=tuple(observation_counts),
        original_padded_size=original_size,
        padded_size_budget=budget,
        dense_sample_position=dense_position,
        dense_sample_id=sample_ids[dense_position],
        piece_positions=tuple(pieces),
        piece_padded_sizes=piece_sizes,
        was_split=original_size > budget,
    )


def slice_padded_batch(
    batch: Mapping[str, Tensor], positions: tuple[int, ...]
) -> dict[str, Tensor]:
    """Return one order-preserving piece, repadded to its own true row lengths.

    ``collate_fn`` pads each full batch to its longest input and outcome
    sequences.  Merely selecting samples keeps those large trailing pads in
    every piece, so it does not lower the model's padded time-axis footprint.
    The adapter stores all real rows before padding; crop each time-bearing
    tensor after selection to the longest nonempty row count in this piece.
    At least one row is retained because empty windows use a masked dummy row.
    """

    if not positions:
        raise ValueError("cannot materialize an empty P1 batch piece")
    batch_size = int(batch["sample_ID"].shape[0])
    if any(position < 0 or position >= batch_size for position in positions):
        raise IndexError("P1 piece contains a position outside the original batch")
    device = batch["sample_ID"].device
    index = torch.as_tensor(positions, dtype=torch.long, device=device)
    result: dict[str, Tensor] = {}
    for key, value in batch.items():
        if not isinstance(value, Tensor) or value.ndim == 0 or int(value.shape[0]) != batch_size:
            raise ValueError(f"P1 batch field {key!r} does not have the batch dimension first")
        result[key] = value.index_select(0, index.to(value.device))

    def piece_time_length(mask_key: str) -> int:
        mask = result[mask_key] > 0
        if mask.ndim < 2:
            raise ValueError(f"P1 measurement mask {mask_key!r} must have a time dimension")
        row_has_measurement = mask.reshape(mask.shape[0], mask.shape[1], -1).any(dim=2)
        rows_per_sample = row_has_measurement.sum(dim=1)
        return max(1, int(rows_per_sample.max().item()))

    for prefix, mask_key in (("x", "x_mask"), ("y", "y_mask")):
        length = piece_time_length(mask_key)
        for key in (prefix, f"{prefix}_mark", mask_key):
            value = result[key]
            if value.ndim < 2:
                raise ValueError(f"P1 time-bearing tensor {key!r} must have a time dimension")
            if int(value.shape[1]) < length:
                raise ValueError(
                    f"P1 tensor {key!r} is shorter than its measured rows: "
                    f"{value.shape[1]} < {length}"
                )
            result[key] = value[:, :length]
    return result
