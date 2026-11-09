from __future__ import annotations

import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

ROOT = Path(__file__).parents[1]
EXAMPLES = sorted((ROOT / "examples").glob("*.py"))


@pytest.mark.parametrize("source", EXAMPLES, ids=lambda path: path.stem)
def test_example_runs_in_isolation(source: Path, tmp_path: Path) -> None:
    example_directory = tmp_path / "examples"
    example_directory.mkdir()
    example = shutil.copy(source, example_directory / source.name)
    # Reuse the normal font cache; rebuilding it can dominate this smoke test.
    environment = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "PYTHONPATH": str(ROOT / "src"),
    }

    command = [sys.executable, example]
    if source.stem == "map_multistart":
        # Run the example and its independent optimum check in the same fresh process.
        command = [sys.executable, str(Path(__file__).resolve()), example]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def _check_map_multistart_optimum(example):
    result = example["result"]
    points = np.asarray(example["points"], dtype=np.float64)
    readings = np.asarray(example["observations"], dtype=np.float64)
    distance_squared = (points - points.T) ** 2
    identity = np.eye(len(points))

    def reference(log_parameters):
        variance, lengthscale, observation_variance = np.exp(log_parameters)
        scaled_distance = distance_squared / lengthscale**2
        signal = variance * np.exp(-0.5 * scaled_distance)
        matrix = signal + (observation_variance + example["JITTER"]) * identity
        lower = np.linalg.cholesky(matrix)
        whitened = np.linalg.solve(lower, readings)
        alpha = np.linalg.solve(lower.T, whitened)
        nll = (
            0.5 * whitened @ whitened
            + np.log(np.diag(lower)).sum()
            + 0.5 * len(points) * np.log(2 * np.pi)
        )
        # Gamma(2, 1), LogNormal(0, 0.7), and Gamma(2, 20) densities on
        # constrained parameters; no optimization-coordinate Jacobian.
        log_variance, log_lengthscale, log_observation_variance = log_parameters
        log_prior = (
            log_variance
            - variance
            - log_lengthscale
            - np.log(0.7 * np.sqrt(2 * np.pi))
            - 0.5 * (log_lengthscale / 0.7) ** 2
            + 2 * np.log(20)
            + log_observation_variance
            - 20 * observation_variance
        )
        weight = np.linalg.solve(matrix, identity) - np.outer(alpha, alpha)
        gradient = 0.5 * np.array(
            [
                np.sum(weight * signal),
                np.sum(weight * signal * scaled_distance),
                observation_variance * np.trace(weight),
            ]
        ) - np.array(
            [1 - variance, -1 - log_lengthscale / 0.7**2, 1 - 20 * observation_variance]
        )
        return nll - log_prior, gradient

    optimum = minimize(
        reference,
        np.log([1.0, 0.8, 0.05]),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-6},
    )
    assert optimum.success, optimum.message
    assert np.linalg.norm(optimum.jac) < 1e-6
    assert np.all(result.converged)
    assert np.all(result.gradient_norms <= example["GRADIENT_TOLERANCE"])
    # The optimizer's explicit stopping target determines the parameter budget.
    np.testing.assert_allclose(result.final_losses, optimum.fun, atol=2e-5, rtol=0)
    for model, data in result.models:
        fitted = [
            model.kernel.variance,
            model.kernel.lengthscales.lengthscale,
            data.variance_flat[0],
        ]
        np.testing.assert_allclose(fitted, np.exp(optimum.x), rtol=2e-3, atol=0)


if __name__ == "__main__":
    _check_map_multistart_optimum(runpy.run_path(sys.argv[1], run_name="__main__"))
