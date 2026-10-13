"""Exact head parameter counting and budget matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


def _canonical(layer_type: str) -> str:
    value = layer_type.lower().replace("-", "_")
    aliases = {
        "bspline": "kan_bspline",
        "kan": "kan_bspline",
        "rbf": "rbf_kan",
        "linear_head": "linear",
    }
    return aliases.get(value, value)


def count_head_params(
    layer_type: str,
    d_in: int,
    d_out: int = 1,
    hidden_dim: int | None = None,
    **layer_kwargs: Any,
) -> int:
    """Count trainable parameters for a one- or two-layer head analytically."""

    layer_type = _canonical(layer_type)
    d_in, d_out = int(d_in), int(d_out)
    if d_in < 1 or d_out < 1:
        raise ValueError("d_in and d_out must be positive")
    if layer_type == "linear":
        return (d_in + 1) * d_out
    hidden = int(hidden_dim if hidden_dim is not None else layer_kwargs.get("hidden", 64))
    if hidden < 1:
        raise ValueError("hidden_dim must be positive")

    if layer_type == "mlp":
        return (d_in + 1) * hidden + (hidden + 1) * d_out
    if layer_type == "kan_bspline":
        grid_size = int(layer_kwargs.get("grid_size", 5))
        spline_order = int(layer_kwargs.get("spline_order", 3))
        edge_params = grid_size + spline_order + 2
        return (d_in * hidden + hidden * d_out) * edge_params
    if layer_type == "rbf_kan":
        num_centers = int(layer_kwargs.get("num_centers", 8))
        edge_params = num_centers + 1
        return (
            (d_in * hidden + hidden * d_out) * edge_params
            + 2 * num_centers * (d_in + hidden)
        )
    raise ValueError(f"unknown head type: {layer_type}")


def _nearest_hidden(
    layer_type: str,
    d_in: int,
    d_out: int,
    target_params: int,
    *,
    max_hidden: int,
    **layer_kwargs: Any,
) -> int:
    """Binary-search the monotone hidden-width budget and return the nearest."""

    low, high = 1, max_hidden
    while low <= high:
        middle = (low + high) // 2
        count = count_head_params(
            layer_type, d_in, d_out, hidden_dim=middle, **layer_kwargs
        )
        if count < target_params:
            low = middle + 1
        elif count > target_params:
            high = middle - 1
        else:
            return middle
    candidates = [max(1, min(max_hidden, low)), max(1, min(max_hidden, high))]
    return min(
        candidates,
        key=lambda hidden: abs(
            count_head_params(layer_type, d_in, d_out, hidden_dim=hidden, **layer_kwargs)
            - target_params
        ),
    )


def solve_hidden_dim_for_budget(
    layer_type: str,
    d_in: int,
    d_out: int,
    target_params: int,
    **layer_kwargs: Any,
) -> int:
    """Return a hidden width whose head count is nearest to ``target_params``.

    For KAN layers, passing ``search_grid=True`` searches the integer grid size
    as well as the hidden width.  The function's public return value remains
    the hidden width as required by the Phase 0 interface; callers needing the
    matched grid should use :func:`match_head_spec`.
    """

    if target_params < 1:
        raise ValueError("target_params must be positive")
    max_hidden = int(layer_kwargs.pop("max_hidden", 1024))
    search_grid = bool(layer_kwargs.pop("search_grid", False))
    if not search_grid or _canonical(layer_type) not in {"kan_bspline", "rbf_kan"}:
        return _nearest_hidden(
            layer_type, d_in, d_out, target_params, max_hidden=max_hidden, **layer_kwargs
        )
    spec = match_head_spec(
        layer_type,
        d_in,
        d_out,
        target_params,
        max_hidden=max_hidden,
        **layer_kwargs,
    )
    return int(spec["hidden_dim"])


def match_head_spec(
    layer_type: str,
    d_in: int,
    d_out: int,
    target_params: int,
    *,
    reference_hidden: int = 64,
    max_hidden: int = 1024,
    grid_size: int = 5,
    spline_order: int = 3,
    num_centers: int = 8,
    search_grid: bool = True,
    max_grid_size: int = 32,
    x_range: tuple[float, float] | None = None,
) -> Dict[str, Any]:
    """Find a concrete head specification and report the exact mismatch."""

    canonical = _canonical(layer_type)
    if canonical == "linear":
        hidden = None
        kwargs: Dict[str, Any] = {}
    elif canonical == "mlp":
        hidden = _nearest_hidden(
            canonical,
            d_in,
            d_out,
            target_params,
            max_hidden=max_hidden,
        )
        kwargs = {}
    elif canonical == "kan_bspline":
        candidates = []
        grid_values = range(2, max_grid_size + 1) if search_grid else [grid_size]
        for candidate_grid in grid_values:
            hidden_candidate = _nearest_hidden(
                canonical,
                d_in,
                d_out,
                target_params,
                max_hidden=max_hidden,
                grid_size=candidate_grid,
                spline_order=spline_order,
            )
            count = count_head_params(
                canonical,
                d_in,
                d_out,
                hidden_candidate,
                grid_size=candidate_grid,
                spline_order=spline_order,
            )
            candidates.append((abs(count - target_params), hidden_candidate, candidate_grid))
        _, hidden, chosen_grid = min(candidates, key=lambda item: item[0])
        kwargs = {"grid_size": chosen_grid, "spline_order": spline_order}
    elif canonical == "rbf_kan":
        hidden = _nearest_hidden(
            canonical,
            d_in,
            d_out,
            target_params,
            max_hidden=max_hidden,
            num_centers=num_centers,
        )
        kwargs = {"num_centers": num_centers}
    else:
        raise ValueError(f"unknown head type: {layer_type}")

    count = count_head_params(canonical, d_in, d_out, hidden, **kwargs)
    result: Dict[str, Any] = {
        "layer_type": canonical,
        "hidden_dim": hidden,
        "target_params": int(target_params),
        "n_params": int(count),
        "param_diff": int(count - target_params),
        "param_diff_pct": 100.0 * (count - target_params) / target_params,
        **kwargs,
    }
    if x_range is not None:
        result["x_range"] = tuple(x_range)
    return result


def mlp_target_params(d_in: int, hidden_dim: int = 64, d_out: int = 1) -> int:
    """The target budget used for the B1/B2 comparison."""

    return count_head_params("mlp", d_in, d_out, hidden_dim=hidden_dim)


def assert_within_budget(actual: int, target: int, tolerance: float = 0.05) -> None:
    relative = abs(int(actual) - int(target)) / max(int(target), 1)
    if relative > tolerance:
        raise AssertionError(
            f"parameter budget mismatch: actual={actual}, target={target}, "
            f"relative_error={relative:.3%} > {tolerance:.1%}"
        )


def count_gate_basis_params(
    basis: str,
    n_groups: int,
    *,
    num_centers: int = 8,
    parameterization: str = "fixed",
    grid_size: int | None = None,
    spline_order: int = 3,
    n_freq: int | None = None,
    num_num: int | None = None,
    num_den: int | None = None,
) -> int:
    """Count the compact Phase 5 scalar-edge bank exactly."""

    groups = int(n_groups)
    if groups < 1:
        raise ValueError("n_groups must be positive")
    name = str(basis).lower().replace("-", "_")
    mode = str(parameterization).lower().replace("-", "_")
    if mode not in {"fixed", "adaptive"}:
        raise ValueError("parameterization must be fixed or adaptive")
    if mode == "adaptive":
        if name not in {"rbf", "gaussian_rbf"}:
            raise ValueError("adaptive parameterization is only supported for RBF")
        centers = int(num_centers)
        if centers < 1:
            raise ValueError("num_centers must be positive")
        return groups * (3 * centers + 1)
    if name in {"rbf", "gaussian_rbf"}:
        per_edge = int(num_centers) + 1
    elif name in {"bspline", "b_spline", "spline"}:
        chosen_grid = max(int(num_centers) - 3, 1) if grid_size is None else int(grid_size)
        per_edge = chosen_grid + int(spline_order) + 2
    elif name in {"fourier", "fou", "truncated_fourier"}:
        chosen_freq = max((int(num_centers) - 1) // 2, 1) if n_freq is None else int(n_freq)
        per_edge = 2 * chosen_freq + 1
    elif name in {"rational", "kat"}:
        chosen_num = (int(num_centers) + 1) // 2 if num_num is None else int(num_num)
        chosen_den = int(num_centers) // 2 if num_den is None else int(num_den)
        per_edge = chosen_num + chosen_den + 1
    else:
        raise ValueError(f"unknown gate basis: {basis}")
    if per_edge < 1:
        raise ValueError("gate basis must have a positive parameter count")
    return groups * per_edge


def solve_gate_basis_match(
    basis: str,
    n_groups: int,
    target_params_per_edge: int,
    *,
    spline_order: int = 3,
) -> Dict[str, Any]:
    """Choose native basis sizes so a scalar-edge bank is exactly matched.

    The Phase 5 primary pair uses a per-edge MLP budget of 25 parameters
    (hidden width 8).  Unlike a percentage tolerance, this routine rejects
    budgets that a basis cannot represent exactly, making the track-P claim
    auditable from the serialized configuration.
    """

    target = int(target_params_per_edge)
    if target < 2:
        raise ValueError("target_params_per_edge must be at least two")
    name = str(basis).lower().replace("-", "_")
    if name in {"rbf", "gaussian_rbf"}:
        params = {"num_centers": target - 1}
    elif name in {"bspline", "b_spline", "spline"}:
        grid = target - int(spline_order) - 2
        if grid < 1:
            raise ValueError("target budget is too small for the requested spline order")
        params = {"grid_size": grid, "spline_order": int(spline_order)}
    elif name in {"fourier", "fou", "truncated_fourier"}:
        if target % 2 == 0:
            raise ValueError("Fourier scalar edges require an odd parameter budget")
        params = {"n_freq": (target - 1) // 2}
    elif name in {"rational", "kat"}:
        remaining = target - 1
        params = {"num_num": remaining // 2, "num_den": remaining - remaining // 2}
        if min(params.values()) < 1:
            raise ValueError("target budget is too small for a rational edge")
    else:
        raise ValueError(f"unknown gate basis: {basis}")
    total = count_gate_basis_params(name, n_groups, **params)
    return {
        "basis": name,
        "n_groups": int(n_groups),
        "params_per_edge": int(target),
        "n_params": int(total),
        "parameters": params,
    }


def count_grouped_function_params(
    function_type: str,
    n_groups: int,
    d_in: int,
    d_out: int,
    *,
    hidden_dim: int = 16,
    grid_size: int = 8,
    spline_order: int = 3,
    basis: str = "bspline",
    num_centers: int = 8,
    n_freq: int = 6,
) -> int:
    """Analytically count one grouped event-function module."""

    n_groups, d_in, d_out = int(n_groups), int(d_in), int(d_out)
    if min(n_groups, d_in, d_out) < 1:
        raise ValueError("grouped dimensions must be positive")
    kind = str(function_type).lower().replace("-", "_")
    if kind in {"linear", "identity"}:
        return n_groups * ((d_in + 1) * d_out)
    if kind == "mlp":
        hidden_dim = int(hidden_dim)
        return n_groups * ((d_in + 1) * hidden_dim + (hidden_dim + 1) * d_out)
    if kind in {"kan", "kan_bspline", "bspline"}:
        edge_params = int(grid_size) + int(spline_order) + 2
        return n_groups * (d_in * d_out) * edge_params
    if kind in {"fourier", "kan_fourier"}:
        edge_params = 2 * int(n_freq) + 1
        return n_groups * (d_in * d_out) * edge_params
    if kind in {"rbf", "rbf_kan"}:
        centers = int(num_centers)
        return n_groups * ((d_in * d_out) * (centers + 1) + 2 * d_in * centers)
    raise ValueError(f"unknown grouped function type: {function_type}")


def solve_matched_pair(
    n_groups: int,
    d_in: int = 1,
    d_out: int = 64,
    *,
    n_functions: int = 1,
    reference_hidden: int = 16,
    grid_size: int = 8,
    spline_order: int = 3,
    max_hidden: int = 4096,
    basis: str = "bspline",
    num_centers: int = 8,
    n_freq: int = 6,
    target_params_per_edge: int | None = None,
    gate_spline_order: int = 3,
) -> Dict[str, Any]:
    """Return width- and parameter-matched grouped MLP specifications.

    ``n_functions`` is two for the additive value+gap configurations and one
    for a joint function.  The result is metadata only; the concrete modules
    remain the simple classes in ``src.models.layers.grouped_kan``.
    """

    if int(n_functions) < 1:
        raise ValueError("n_functions must be positive")
    if target_params_per_edge is not None:
        # Explicit Phase 5 mode: the function is a bank of scalar recency
        # edges rather than a d_out-dimensional event encoder.  Keeping this
        # route on the public matcher prevents an approximate percentage match
        # from being mistaken for the track-P comparison.
        matched = solve_gate_basis_match(
            basis,
            n_groups,
            int(target_params_per_edge),
            spline_order=int(gate_spline_order),
        )
        mlp_hidden_numerator = int(target_params_per_edge) - 1
        if mlp_hidden_numerator % 3:
            raise ValueError("the scalar MLP cannot represent this exact budget")
        mlp_hidden = mlp_hidden_numerator // 3
        return {
            "kan": matched,
            "mlp_param": {
                "hidden_dim": mlp_hidden,
                "params_per_edge": int(target_params_per_edge),
                "n_params": int(n_groups) * int(target_params_per_edge),
                "param_diff": 0,
            },
        }
    basis_name = str(basis).lower().replace("-", "_")
    kan_one = count_grouped_function_params(
        basis_name,
        n_groups,
        d_in,
        d_out,
        grid_size=grid_size,
        spline_order=spline_order,
        num_centers=num_centers,
        n_freq=n_freq,
    )
    kan_params = int(n_functions) * kan_one
    width_one = count_grouped_function_params(
        "mlp", n_groups, d_in, d_out, hidden_dim=reference_hidden
    )
    width_params = int(n_functions) * width_one
    # Parameter-matched MLP widths are monotone for fixed input/output sizes.
    best_hidden = 1
    best_diff = None
    for hidden in range(1, int(max_hidden) + 1):
        candidate = int(n_functions) * count_grouped_function_params(
            "mlp", n_groups, d_in, d_out, hidden_dim=hidden
        )
        diff = abs(candidate - kan_params)
        if best_diff is None or diff < best_diff:
            best_diff, best_hidden = diff, hidden
        if candidate > kan_params and diff > best_diff:
            break
    param_count = int(n_functions) * count_grouped_function_params(
        "mlp", n_groups, d_in, d_out, hidden_dim=best_hidden
    )
    return {
        "kan": {
            "n_params": kan_params,
            "n_groups": int(n_groups),
            "d_in": int(d_in),
            "d_out": int(d_out),
            "n_functions": int(n_functions),
            "basis": basis_name,
            "grid_size": int(grid_size),
            "spline_order": int(spline_order),
            "num_centers": int(num_centers),
            "n_freq": int(n_freq),
        },
        "mlp_width": {
            "hidden_dim": int(reference_hidden),
            "n_params": width_params,
            "param_diff": int(width_params - kan_params),
        },
        "mlp_param": {
            "hidden_dim": int(best_hidden),
            "n_params": param_count,
            "param_diff": int(param_count - kan_params),
        },
    }
