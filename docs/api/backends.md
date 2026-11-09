# Numerical backends

`GaussianProcess` accepts a numerical backend through `solver`.
`linalg.DenseBackend()` is the stateless default backend. Its `prepare` method
returns a `CholeskySystem` containing the factorization and activity state for
exact dense inference and fitting.

## Backend contract

`SolverConfiguration.prepare(system, ...)` prepares an immutable `LinearSystem`.
The input `SymmetricLinearOperator` supplies `capacity`, `matmul`, `diagonal`,
and tiled `materialize`.
Prepared systems provide solves, quadratic forms, log determinants, covariance
and variance reductions, reusable projections, and append operations. These
interfaces use numerical arrays and activity information rather than GP
measurement types.

For a prepared system, `size` is the active scalar count as a dynamic JAX
integer, and `capacity` is the reserved scalar dimension as a static Python
integer. Solves and projections accept full-capacity inputs. Activation increases
`size` while preserving `capacity`. Input operators also expose `capacity`, and
updates expose `previous_capacity` for the reserved dimension before activation.

Custom backends inherit from `eqx.Module` and `SolverConfiguration`. Their
prepared systems inherit from `eqx.Module` and `LinearSystem`. Both interfaces
are abstract base classes, so subclasses must implement all abstract methods
before they can be instantiated. Numerical inputs and update descriptions
explicitly inherit the `SymmetricLinearOperator` and `LinearSystemUpdate`
abstract base classes. Updates store their numerical system and newly active
indices as fields.

The small input interface keeps backends independent of kernels and observation
types. Dense preparation requests materialization. The operator also supports
products without storing the full matrix, ready for future iterative backends.

`CovarianceProjection` is an abstract interface for reusable covariance
reductions. The dense `CholeskyProjection` stores a whitened cross-covariance
whose Gram matrix is the reduction. Diagonal and cross reductions let consumers
reuse that state without accessing a backend's factorization.

Likelihood objectives use the configured backend's `quadratic_form` and
`logdet`. The dense quadratic form needs one triangular solve. Invalid
factorizations propagate non-finite statistics so fitting can reject invalid
trials. Backend configuration, prepared state, and returned results must support
JAX transformations, with numerical arrays stored as dynamic PyTree leaves.

Use `configuration.prepare(system)` to construct prepared state. For an already
materialized matrix, use `CholeskySystem.from_matrix(matrix, jitter=...)`.
Factorization uses the symmetric part of the matrix with diagonal jitter.
cholops owns this input symmetrization and the symmetry-preserving reverse rule.

Dense systems store a cholops `PaddedCholesky` in `factor`. cholops owns
factorization, numerical consumers, reserved-slot activation, and coefficient
reuse. `factor.order` maps physical factor rows to canonical slots, and
`factor.inverse_order` maps canonical slots back to physical rows. `factor.factor`
exposes the lower-triangular array in physical order. `solve` and
`project` accept canonical-order inputs. Preparing a system with `active=mask`
reserves the inactive identity slots for append. Solves, quadratic forms, and
projections ignore inactive input entries, including non-finite values. Solves
return zero inactive coefficients. Generic systems without a mask are fully
active. `CholeskySystem(factor, tile_size)` wraps an existing `PaddedCholesky`.


::: gpex.linalg.DenseBackend

::: gpex.linalg.CholeskySystem

::: gpex.linalg.SolverConfiguration

::: gpex.linalg.LinearSystem

::: gpex.linalg.SymmetricLinearOperator

::: gpex.linalg.LinearSystemUpdate

::: gpex.linalg.CovarianceProjection

::: gpex.linalg.CholeskyProjection

## Moment assembly

`gpex.moments` assembles prior means and covariances of linear functionals in
canonical block order. These functions accept public blocks or reusable
`PreparedFunctionals` from `gpex.moments.measurements`. Preparation groups
compatible operators and retains activity and ordering information.

`assemble_covariance` supports tiled evaluation while still returning the full
dense matrix. `assemble_covariance_diagonal` evaluates only the diagonal.
Inactive reserved entries are zero. These functions evaluate arrays and can be
wrapped in an outer JIT with their consumers.

::: gpex.moments
    options:
      members:
        - apply_covariance
        - assemble_mean
        - assemble_covariance
        - assemble_covariance_diagonal

## Covariance operator

`CovarianceSystem` describes an observation covariance including block noise,
jitter, and reserved inactive slots. Backends can consume it through the
`SymmetricLinearOperator` interface:

```python
product = system.matmul(rhs, tile_size=128)
diagonal = system.diagonal()
matrix = system.materialize(tile_size=128)
```

`rhs` has shape `(capacity, *columns)` in canonical order. Multiplication
accumulates tiles directly and passes inactive inputs through as identity rows.
The diagonal contains ones in inactive slots. This differs from prepared
`solve`, which ignores inactive inputs. Tile widths count scalar outputs, but
one operator output is indivisible. Tile size is static under JIT.

Products support JIT, vmap, and numerical differentiation. Pairwise intermediates
are recomputed in reverse mode so differentiation also avoids full covariance
storage. General kernels still require quadratic pairwise work. Dense Cholesky
remains the only backend, and likelihood evaluation remains exact.

::: gpex.covariance_systems.CovarianceSystem
