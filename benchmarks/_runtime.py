"""Initialize numerical libraries without compiling a benchmark workload."""

from __future__ import annotations

import time
from functools import cache

TIMING_PROTOCOL = "initialized-runtime-v1"


def initialize(case):
    """Return separately recorded process initialization for this backend.

    The fixed seven-row problem exercises factorization, triangular solves and
    multiplication. It shares neither a callable nor inputs with a workload.
    Sequence preparation may call this before its timing helper, so initialize
    each framework/backend/precision only once per worker.
    """
    framework = "torch" if "gpytorch" in case["workload"] else "jax"
    elapsed = _initialize(framework, case["backend"], case.get("dtype"))
    return dict(
        timing_protocol=TIMING_PROTOCOL,
        runtime_initialization_ms=elapsed,
    )


@cache
def _initialize(framework, backend, dtype):
    start = time.perf_counter()
    if framework == "jax":
        import jax
        import jax.numpy as jnp
        import jax.scipy.linalg as linalg

        jax.config.update("jax_enable_compilation_cache", False)
        if jax.default_backend() != backend:
            raise ValueError(f"requested {backend}, got {jax.default_backend()}")

        @jax.jit
        def exercise(matrix, rhs):
            factor = jnp.linalg.cholesky(matrix)
            projected = linalg.solve_triangular(factor, rhs, lower=True)
            solved = linalg.solve_triangular(factor, projected, lower=True, trans="T")
            return solved, projected.T @ projected

        matrix = jnp.eye(7) + jnp.full((7, 7), 0.05)
        rhs = jnp.ones((7, 3))
        for _ in range(2):
            jax.block_until_ready(exercise(matrix, rhs))
    else:
        import torch

        precision = torch.float64 if dtype == "float64" else torch.float32
        device = torch.device("cuda" if backend == "gpu" else "cpu")
        if backend == "gpu" and not torch.cuda.is_available():
            raise ValueError("CUDA-enabled PyTorch is required")
        with torch.inference_mode():
            matrix = torch.eye(7, dtype=precision, device=device) + 0.05
            rhs = torch.ones((7, 3), dtype=precision, device=device)
            for _ in range(2):
                factor = torch.linalg.cholesky(matrix)
                projected = torch.linalg.solve_triangular(factor, rhs, upper=False)
                torch.linalg.solve_triangular(factor.T, projected, upper=True)
                _ = projected.T @ projected
                if backend == "gpu":
                    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - start)
