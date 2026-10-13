import numpy as np

from src.utils.skill_metrics import persistence_skill


def test_naive_one_skill_is_exactly_zero():
    x_last = np.repeat([1.0, 3.0], 50)
    target = np.repeat([2.0, 1.0], 50)
    variable = np.repeat([0, 1], 50)

    skill, frame, excluded = persistence_skill(x_last, target, variable, x_last)

    assert abs(skill) < 1e-9
    assert not excluded
    assert np.allclose(frame["skill"], 0.0)


def test_degenerate_variable_is_excluded():
    prediction = np.zeros(50)
    target = np.zeros(50)
    variable = np.zeros(50, dtype=np.int64)
    x_last = np.zeros(50)

    skill, frame, excluded = persistence_skill(prediction, target, variable, x_last)

    assert np.isnan(skill)
    assert frame.empty
    assert excluded == [0]
