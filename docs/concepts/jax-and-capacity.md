# JAX and observation capacity

A sequential experiment adds observations over time. JAX specializes compiled
functions to array shapes, so growing an array can require another compilation
at every step. gpex observation blocks can reserve space in advance, keeping
their shapes fixed as observations arrive.

## Observed locations and reserved locations

Suppose an experiment starts with three observations and will collect at most
twenty. Reserve twenty locations when constructing the observation block:

```python
import jax.numpy as jnp

from gpex import ObservationBlock

observations = ObservationBlock.from_arrays(
    points=jnp.array([[-1.0], [0.0], [1.0]]),
    y=jnp.array([-0.8, 0.0, 0.8]),
    variance=0.01,
    name="measurements",
    max_size=20,
)
```

This block has `observations.count == 3` active locations and
`observations.capacity == 20` reserved locations. Only the three observations
inform the posterior. Appending two observations increases the count to five
and leaves the capacity at twenty. An append beyond capacity raises an error.

`max_size` counts locations. Each location can contribute several scalar
outputs, as specified by the operator. A gradient block in two dimensions
with capacity twenty reserves forty scalar outputs. Queries use their supplied
locations without reserving additional space.

The default capacity is `max(1024, number_of_supplied_locations)`. For a fixed
dataset, pass `max_size=points.shape[0]` to use compact storage. For an experiment,
choose the capacity from its planned number of measurements. Dense conditioning
uses quadratic memory and cubic factorization work in the total reserved scalar
capacity, including unused space. Each observation block contributes its own
reserve.

## Reusing compiled code

A compiled conditioning function can accept blocks whose active counts change
while their reserved shapes stay fixed. Locations, readings, and noise values
may also change. Keep the compiled function and pass the current data to it.
The [sequential observations tutorial](../tutorials/sequential-streaming.ipynb)
shows how to append readings to an existing posterior.

Block names, operator structure, capacities, array shapes, and dtypes determine
which compiled program can be reused. Changing those can trigger compilation.
Prediction also specializes to query structure and shapes. A different append
batch size can compile another update function even when it fits within the
same capacity.

When model parameters change, pass the model as an argument to the compiled
function. Compiling `gp.condition` directly captures that particular immutable
GP. Constructing and compiling a new closure on every iteration loses the
opportunity to reuse the existing compiled function.

## Updating a dataset and updating a posterior

An observation block stores the measurements. A conditioned posterior stores
the result of inference. Their update methods have different purposes:

| Operation | Result |
| --- | --- |
| `observations.append(locations, readings)` | A new observation block containing the additional data |
| `observations.replace(points, readings)` | A new block with replacement data and the same capacity |
| `gp.condition(dataset)` | A posterior calculated from the dataset and current model parameters |
| `posterior.append(locations, readings, block="measurements")` | A posterior updated using its existing factorization |

All four return new objects. Keep the dataset if you will later fit
hyperparameters or replace observations.

Posterior append assumes the kernel, mean, operator parameters, noise, and
numerical settings stay fixed. Under those conditions, it gives the same
posterior as conditioning on all observations, up to floating-point error.
A single scalar append costs quadratic work in reserved scalar capacity.
Reusing a compiled `condition` call still performs the full factorization.

After fitting hyperparameters, replacing readings, or introducing another
named block, condition on the complete dataset again. Rebuild any
`PreparedQuery` when its conditioning state or target changes.

## Blocks for future observations

An experiment can declare a block before any of its readings are available.
Use empty arrays with their intended shapes, choose a capacity, and provide
a `padding_point` valid for the kernel and operator:

```python
future = ObservationBlock.from_arrays(
    points=jnp.empty((0, 1)),
    y=jnp.empty((0,)),
    variance=0.01,
    name="future_measurements",
    max_size=10,
    padding_point=jnp.array([0.0]),
)
```

The block contributes no observations until it receives readings. Its name,
operator, and noise variance are already available to the compiled update.
See the [block reference](../api/blocks.md) for constructor arguments and shapes.
