"""Small numerical sanity checks for profiling, outside measured intervals."""

from __future__ import annotations

from typing import Any

import numpy as np


def serializable(value: Any) -> Any:
    """Preserve nonfinite evidence without writing nonstandard JSON numbers."""
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if hasattr(value, "__array__"):
        return serializable(np.asarray(value))
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def check_variance(values, case, evidence, *, prior_variance=1.0):
    """Check the physical range, allowing subtraction roundoff without clipping."""
    array = np.asarray(values)
    tolerance = (
        8 * np.finfo(np.dtype(case.get("dtype", "float64"))).eps * prior_variance
    )
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    evidence["minimum_variance"] = serializable(minimum)
    evidence["maximum_variance"] = serializable(maximum)
    evidence["prior_variance"] = prior_variance
    evidence["variance_roundoff_allowance"] = tolerance
    if not np.isfinite(array).all() or minimum < -tolerance:
        raise ValueError("negative or non-finite posterior variance")
    if maximum > prior_variance + tolerance:
        raise ValueError("posterior variance exceeds prior variance")
