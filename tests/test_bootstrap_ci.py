import numpy as np

from src.training.bootstrap import paired_bootstrap_ci
from src.utils.metrics import auprc_score


def test_paired_bootstrap_is_reproducible_and_positive_for_better_predictions():
    y = np.asarray([0, 0, 0, 1, 1, 1])
    p_good = np.asarray([0.05, 0.1, 0.2, 0.8, 0.9, 0.95])
    p_bad = 1.0 - p_good
    first = paired_bootstrap_ci(y, p_good, p_bad, auprc_score, n_boot=300, seed=7)
    second = paired_bootstrap_ci(y, p_good, p_bad, auprc_score, n_boot=300, seed=7)
    assert first == second
    assert first["mean_delta"] > 0
    assert first["ci95_high"] > first["ci95_low"]
