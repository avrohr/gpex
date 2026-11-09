"""Reuse compiled conditioning and compare it with incremental posterior updates."""

# Run with JAX_ENABLE_X64=1 for the recommended precision.

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from gpex import (
    FunctionalBlock,
    FunctionalDataset,
    FunctionalQuery,
    GaussianProcess,
    ObservationBlock,
)
from gpex.blocks import Locations
from gpex.kernels import RBFKernel
from gpex.operators import Identity


@eqx.filter_jit
def condition(gp, dataset):
    return gp.condition(dataset)


@eqx.filter_jit
def append_and_predict(posterior, points, values, query):
    posterior = posterior.append(Locations(points), values, block="values")
    return posterior, posterior.predict(query)


x = jnp.linspace(-1.0, 1.0, 12)[:, None]
y = jnp.sin(2.0 * x[:, 0])
block = ObservationBlock.from_arrays(
    points=x[:3], y=y[:3], variance=0.02, name="values", max_size=16
)
gp = GaussianProcess(RBFKernel(lengthscale=0.5))
query = FunctionalQuery(
    [
        FunctionalBlock(
            locations=jnp.linspace(-1.2, 1.2, 41)[:, None],
            operator=Identity(),
            name="prediction",
        )
    ]
)
initial = condition(gp, FunctionalDataset(blocks=[block]))
posterior = initial
for i in range(3, 12):
    points, values = x[i : i + 1], y[i : i + 1]

    # Updating the dataset then conditioning reuses the compiled program, but
    # computes a new factorization. This also permits changed GP parameters.
    block = block.append(locations=Locations(points), y=values)
    rebuilt = condition(gp, FunctionalDataset(blocks=[block]))

    # Posterior append updates the existing factorization. Its GP parameters,
    # operators, and measurement variance must stay fixed between updates.
    posterior, prediction = append_and_predict(posterior, points, values, query)

# Both routes use all twelve readings and should give the same posterior.
reference = rebuilt.predict(query)
# Float32 accumulates rounding across nine updates of this well-conditioned GP.
tolerance = 2e-11 if jax.config.x64_enabled else 2e-5
np.testing.assert_allclose(
    prediction.mean.flat, reference.mean.flat, atol=tolerance, rtol=tolerance
)
np.testing.assert_allclose(
    prediction.covariance.full,
    reference.covariance.full,
    atol=tolerance,
    rtol=tolerance,
)
print("Nine posterior updates agree with repeated conditioning.")

# Replacing the dataset also reuses conditioning's reserved shapes. Recondition
# from these readings rather than appending to the old posterior.
replacement = block.replace(points=x[:2], y=y[:2])
replaced = condition(gp, FunctionalDataset(blocks=[replacement]))
assert block.capacity == replacement.capacity == 16
assert int(block.count) == 12 and int(replacement.count) == 2
assert jnp.all(jnp.isfinite(replaced.predict_mean(query).flat))
print("Reconditioned on two replacement readings with the same capacity.")
