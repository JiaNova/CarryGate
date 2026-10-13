import numpy as np

from src.utils.forecast_metrics import normalized_forecast_metrics


def test_normalized_mae_has_analytic_value():
    prediction = np.asarray([1.0, 3.0, 2.0, 2.0])
    target = np.asarray([0.0, 2.0, 2.0, 4.0])
    variable = np.asarray([0, 0, 1, 1])
    result = normalized_forecast_metrics(prediction, target, variable, [1.0, 3.0])
    # v0: errors 1+1 over baseline 1+1 -> 1; v1: errors 0+2 over 1+1 -> 1.
    assert np.isclose(result["nmae"], 1.0)


def test_constant_variable_is_reported_but_not_epsilon_amplified():
    result = normalized_forecast_metrics([1.0, 1.0, 0.0], [0.0, 0.0, 2.0], [0, 0, 1], [0.0, 1.0])
    assert result["n_variables_undefined"] == 1
    assert 0 in result["undefined_variables"]
    assert np.isfinite(result["nmae"])
