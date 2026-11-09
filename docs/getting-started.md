# Getting started

## Install

gpex requires **Python 3.12 or newer** and **JAX 0.11.2**.

Install the released package from PyPI:

```bash
pip install gpex
```

Or install the latest development version from GitHub:

```bash
pip install "git+https://github.com/avrohr/gpex.git@main"
```

To run the tutorials locally, install
[`uv`](https://docs.astral.sh/uv/getting-started/installation/), then clone the
repository and install the optional notebook dependencies:

```bash
git clone https://github.com/avrohr/gpex.git
cd gpex
uv sync --extra examples
uv run jupyter lab docs/tutorials/first-posterior.ipynb
```

JAX selects the CPU backend by default. Follow the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html) when
you need a platform-specific accelerator build.

## Floating precision

Float64 is recommended for numerical robustness. Enable it before importing
gpex or constructing arrays:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

Use JAX's default floating dtype throughout model parameters, locations,
readings, and queries. Mixed dtypes are unsupported. Float32 is supported when
x64 is disabled, with less accuracy available for nearly noiseless measurements,
small posterior variances, and tightly converged fits.

## Build a dataset

A `FunctionalBlock` describes what to measure: locations, a linear operator,
and a name. Here, `Identity` means function values. An `ObservationDesignBlock`
adds the measurement noise variance, and an `ObservationBlock` attaches the
observed readings. A `FunctionalDataset` collects one or more observation blocks.

Locations use shape `(n, d)`: `n` rows of measurement locations, each containing
`d` input coordinates. Here there are three locations in one dimension, so the
shape is `(3, 1)`. Coordinates must have a floating-point dtype.

```python
import jax.numpy as jnp

from gpex import (
    FunctionalBlock,
    FunctionalDataset,
    ObservationBlock,
    ObservationDesignBlock,
    operators,
)

points = jnp.array([[-1.0], [0.0], [1.0]])
readings = jnp.array([-0.8, 0.0, 0.8])

functional = FunctionalBlock(
    locations=points,
    operator=operators.Identity(),
    name="observed_values",
)
design = ObservationDesignBlock(
    functional=functional, variance=0.01, max_size=points.shape[0],
)
observations = ObservationBlock(design=design, y=readings)
dataset = FunctionalDataset(blocks=[observations])
```

For convenience, `ObservationBlock.from_arrays` constructs the functional and
design blocks for you. It defaults to the identity operator, so the following
creates the same dataset:

```python
dataset = FunctionalDataset(blocks=[
    ObservationBlock.from_arrays(
        points=points,
        y=readings,
        variance=0.01,
        name="observed_values",
        max_size=points.shape[0],
    )
])
```

`variance` specifies the measurement-error variance shared by the block.
Here `max_size=points.shape[0]` reserves exactly the three locations in this
fixed dataset. [Observation capacity](concepts/jax-and-capacity.md) explains
reserving space for a growing experiment.

## Construct a query

Queries have the same location/operator structure but no observations or noise.
Block names become keys in the result.

`jnp.linspace` returns a one-dimensional array of shape `(101,)`. The
`[:, None]` below adds a column axis, giving the required `(101, 1)` locations.

```python
from gpex import FunctionalBlock, FunctionalQuery, operators

x = jnp.linspace(-1.5, 1.5, 101)[:, None]
query = FunctionalQuery([
    FunctionalBlock(x, operators.Identity(), "value"),
    FunctionalBlock(x, operators.Gradient(), "gradient"),
])
```

## Condition and predict

```python
from gpex import GaussianProcess, kernels

gp = GaussianProcess(kernels.RBFKernel(variance=1.0, lengthscale=0.7))
posterior = gp.condition(dataset)
prediction = posterior.predict_marginals(query)

prediction.mean["value"]          # shape (101,)
prediction.mean["gradient"]       # shape (101, 1)
prediction.variance["value"]      # shape (101,)
prediction.variance["gradient"]   # shape (101, 1)
```

`predict_marginals` computes the mean and variance at each query output without
building a full query covariance. When you need correlations between outputs,
request a joint prediction explicitly:

```python
joint = posterior.predict(query)
joint.covariance.full                    # shape (202, 202)
joint.covariance.block("value", "gradient")  # shape (101, 101, 1)
```

To use a Matérn kernel or a lengthscale per coordinate, see
[Kernels and means](api/kernels.md).

Continue with [First posterior](tutorials/first-posterior.ipynb), or read
[Observations, designs, and queries](concepts/observations-designs-queries.md)
for the underlying block model. To train kernel or observation-noise
parameters, see [Fitting hyperparameters](tutorials/mle-map.ipynb).
