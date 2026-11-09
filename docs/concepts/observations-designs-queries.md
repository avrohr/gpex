# Observations, designs, and queries

gpex separates the quantity being measured, the uncertainty of the measurement,
and the reading obtained. The same function value can be described before an
experiment is performed and then paired with its reading afterwards.

| Object | Contains | Role |
| --- | --- | --- |
| `FunctionalBlock` | locations, operator, name | quantities to measure or predict |
| `ObservationDesignBlock` | functional block and measurement-error variance | planned measurements |
| `ObservationBlock` | design and readings `y` | realized measurements |

The following sections build these objects in order, using the function-value
measurements from [Getting started](../getting-started.md#build-a-dataset).

## Describe the quantity to measure

Suppose we want to measure an unknown function $f$ at three locations in one
dimension. A `FunctionalBlock` describes those quantities through their
locations and operator. `Identity()` means evaluating $f$ itself at each
location. The name `"observed_values"` identifies this group of quantities.

```python
import jax.numpy as jnp

from gpex import FunctionalBlock, operators

points = jnp.array([[-1.0], [0.0], [1.0]])
functional = FunctionalBlock(
    locations=points,
    operator=operators.Identity(),
    name="observed_values",
)
```

Each row of `points` specifies one location. This block describes the three
latent values $f(-1)$, $f(0)$, and $f(1)$. It does not yet specify measurement
error or contain readings.

## Designs specify measurements before readings

A reading may differ from the latent function value because of measurement
error. An `ObservationDesignBlock` adds the variance of that error to the
functional block. For this example, choose a variance of `0.01` for every
reading. At each location $x_i$, the reading $y_i$ is modelled as

$$
y_i = f(x_i) + \varepsilon_i, \qquad
\varepsilon_i \sim \mathcal{N}(0, 0.01).
$$

The errors are independent of one another and of the latent function.

`ObservationDesign` collects one or more design blocks:

```python
from gpex import ObservationDesign, ObservationDesignBlock

design_block = ObservationDesignBlock(functional=functional, variance=0.01)
design = ObservationDesign(blocks=[design_block])
```

Here the design contains just the planned value measurements at our three
locations. Each block shares one error variance across its measurements.
Separate blocks can represent measurements with different variances.

The design is complete before any readings are available: it states what
will be measured, where, and with what measurement uncertainty.

## Datasets contain realized measurements

Once the measurements have been taken, an `ObservationBlock` attaches the
readings to their design. A `FunctionalDataset` collects these observation
blocks for conditioning a GP. Suppose we obtained `-0.8`, `0.0`, and `0.8` at
the three locations, in that order:

```python
from gpex import FunctionalDataset, ObservationBlock

readings = jnp.array([-0.8, 0.0, 0.8])
observation = ObservationBlock(design=design_block, y=readings)
dataset = FunctionalDataset(blocks=[observation])
```

Each entry in `readings` corresponds to one row of `points`. These are noisy
function-value observations, with the error variance already specified by
`design_block`.

This dataset contains one block. Larger datasets can combine several blocks,
each with its own locations, operator, and variance. Block names must be
unique within the collection, and all blocks describe measurements of the
same latent function on the same input domain.

See [Conditioning and prediction](../api/inference.md) for constructing
a posterior from the dataset, and [Streaming inference](../tutorials/sequential-streaming.ipynb)
for incorporating later readings.

## Queries describe latent quantities

After observing function values, we may want to predict how quickly the
function changes between the measurement locations. This is a different
quantity of the same latent function: its derivative. In one dimension,
`Gradient()` returns that derivative at each location.

A `FunctionalQuery` collects the functional blocks we want to predict.
Here we ask for derivatives at `-0.5` and `0.5`:

```python
from gpex import FunctionalQuery

target_points = jnp.array([[-0.5], [0.5]])
target_gradient = FunctionalBlock(
    locations=target_points,
    operator=operators.Gradient(),
    name="target_gradient",
)
query = FunctionalQuery(blocks=[target_gradient])
```

The query contains no readings or measurement-error variance. It describes
the latent derivatives whose distribution we want to predict. The name
`"target_gradient"` identifies their means, variances, or samples in a
prediction result. The observed function values inform these derivatives
through their shared GP, as explained in
[Gaussian processes with linear functionals](linear-functionals.md).

Designs and queries also work together when planning a future experiment:
the design specifies the proposed measurements, and the query specifies
the quantities whose uncertainty we want to reduce. For example, we could
compare possible value measurements by how much they would reduce uncertainty
about these target derivatives. With the GP parameters fixed, the conditional
covariance can be calculated before the future readings are known. See
[Experimental design](../tutorials/design-of-experiments.ipynb) for this workflow.

## Access the measurement description

The observation retains the design that gives its readings meaning. Its
functional block describes the underlying latent values, while its design
also specifies the measurement-error variance:

```python
observation.functional             # descriptions of f(-1), f(0), and f(1)
observation.functional.operator    # Identity()
observation.functional.name        # "observed_values"
observation.design                 # the design_block used above
observation.design.variance        # 0.01
observation.y                      # stored readings
```

A design also exposes a query through `design.query`. For the design above,
this describes the underlying function values at the measurement locations.
A prediction for this query excludes measurement error. To obtain the
covariance of future noisy readings at those locations, add the
measurement-error variance to the covariance diagonal.

## A convenience constructor for observed data

When locations and readings are already available, `ObservationBlock.from_arrays`
constructs the functional and design blocks for you. Its default operator is
`Identity()`, so the following is a shorter way to create the same value
observations and dataset as above:

```python
observation = ObservationBlock.from_arrays(
    points=points,
    y=readings,
    variance=0.01,
    name="observed_values",
)
dataset = FunctionalDataset(blocks=[observation])
```

For array shapes and numerical setup, see [Getting started](../getting-started.md).
Storage capacity and compiled updates are covered in
[Reuse compiled conditioning](jax-and-capacity.md).
