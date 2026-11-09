"""Numerical benchmark checks must reject failures and retain useful evidence."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))
spec = importlib.util.spec_from_file_location(
    "benchmark_checks", Path(__file__).parents[1] / "benchmarks/checks.py"
)
assert spec is not None and spec.loader is not None
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)
check_variance = checks.check_variance


def test_roundoff_allowance_does_not_hide_negative_variance():
    evidence = {}
    case = {"dtype": "float32"}
    check_variance([-np.finfo(np.float32).eps], case, evidence)
    assert evidence["minimum_variance"] < 0
    with pytest.raises(ValueError, match="negative"):
        check_variance([-1e-4], case, evidence)
    with pytest.raises(ValueError, match="non-finite"):
        check_variance([np.nan], case, evidence)


@pytest.mark.parametrize("values", [[np.inf], [np.nan], [-1.0], [2.0]])
def test_invalid_variances_are_rejected(values):
    with pytest.raises(ValueError):
        check_variance(values, {"dtype": "float64"}, {})


def test_variance_bounds_allow_roundoff_on_the_prior_scale():
    eps = np.finfo(np.float32).eps
    check_variance([-eps, 1 + eps], {"dtype": "float32"}, {})
    check_variance([3.125], {"dtype": "float64"}, {}, prior_variance=3.125)


def test_warning_categories_separate_numerical_failures_from_advice():
    from types import SimpleNamespace

    comparison_spec = importlib.util.spec_from_file_location(
        "benchmark_compare", Path(__file__).parents[1] / "benchmarks/compare.py"
    )
    assert comparison_spec is not None and comparison_spec.loader is not None
    comparison = importlib.util.module_from_spec(comparison_spec)
    comparison_spec.loader.exec_module(comparison)

    class PerformanceWarning(RuntimeWarning):
        pass

    caught = [
        SimpleNamespace(category=UserWarning, message="prefer float64"),
        SimpleNamespace(category=PerformanceWarning, message="slow operation"),
        SimpleNamespace(category=RuntimeWarning, message="CG did not converge"),
    ]
    evidence, errors = [], []
    comparison.record_warnings(caught, evidence, errors, (PerformanceWarning,))
    assert len(evidence) == 3
    assert errors == ["CG did not converge"]
