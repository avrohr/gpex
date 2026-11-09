"""Run in the precision selected by JAX_ENABLE_X64 before collection."""

import jax
import jax.numpy as jnp
import pytest


def pytest_report_header():
    precision = "float64" if jax.config.read("jax_enable_x64") else "float32"
    return f"JAX {jax.__version__}: {precision}"


@pytest.fixture
def float_info():
    return jnp.finfo(jnp.asarray(0.0).dtype)


@pytest.fixture
def extreme_scales(float_info):
    """Representable entries whose squares underflow or overflow."""
    limit = float(float_info.max)
    return limit**-0.65, limit**0.65
