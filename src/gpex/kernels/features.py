"""Explicit kernel capability for finite feature construction."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable

from jaxtyping import Array, PRNGKeyArray

from .base import Kernel, Point


class FeatureKernel(Kernel):
    """A covariance kernel that can construct a finite feature map.

    Feature support is independent of radiality and coordinate separability.
    Subclasses implement both covariance evaluation and ``random_features``.
    Ordinary GP inference and finite-query sampling require only ``Kernel``.
    """

    @abstractmethod
    def random_features(
        self,
        key: PRNGKeyArray,
        input_dimension: int,
        *,
        feature_count: int = 1024,
    ) -> Callable[[Point], Array]:
        """Construct a fixed random feature map approximating this kernel.

        The returned JAX-compatible callable maps a point of shape ``(d,)`` to
        ``(feature_count,)``. Its feature inner product approximates the kernel,
        including its variance. Reusing the callable reuses the same features.

        Features may be random Fourier features or another finite basis.
        They need not be Fourier or radial. An exact finite feature map is
        also supported. The callable must store numerical state in dynamic
        PyTree leaves and support JIT, vmap, and differentiation up to the
        kernel's supported derivative order.

        Args:
            key: JAX random key for feature construction.
            input_dimension: Number of coordinates in each input point.
            feature_count: Positive number of features. Defaults to 1024.
        """
        ...


__all__ = ["FeatureKernel"]
