# Operators

See [Linear operators](../concepts/linear-operators.md) for formulas and
interpretations, and [Implementing a custom operator](../tutorials/custom-operators.ipynb)
for a complete extension example.

## Custom operators

A `LinearOperator` implements `output_shape(input_dimension)` and
`apply(function, point)`. The point has shape `(d,)`. For a scalar function,
`apply` must return the declared output shape. It must also preserve any
output axes already present on the supplied function, because covariance
construction applies operators to both kernel arguments.

The operation must be linear in `function`. Its locations, weights, and other
parameters must be independent of the function's values. Use JAX-compatible
operations for `jit`, `vmap`, and differentiation. Store differentiable numerical
parameters as array fields. Structural settings such as quadrature order can
be static Equinox fields.

The default `derivative_order` is zero. Override it when the operator takes
derivatives of the function. gpex checks it against the kernel's supported
GP derivative order. A finite difference made from function evaluations
has order zero under this contract.

## Quadrature interfaces

`QuadratureOperator.quadrature(point)` returns absolute evaluation points with
shape `(q, d)` and weights with shape `(q,)`. Its inherited `apply` evaluates
and weights the function while preserving existing output axes. The default
scalar `output_shape` may be overridden for input-dimension restrictions.

Weights may be negative and need not sum to one. Include normalization and
change-of-variable factors in the rule. Points and weights may depend on the
query location and operator parameters. Keep the number of nodes fixed under
tracing.

`TensorQuadratureOperator.coordinate_quadrature(point)` returns one
`(nodes, weights)` pair per coordinate in input order. Each array is
one-dimensional. Node counts can differ between coordinates and must remain
fixed under tracing. The full weight is the product of the coordinate weights.
Any override of `quadrature` or `apply` must represent that same product rule.
This interface enables efficient covariance evaluation with `SeparableKernel`.

## Built-in constructors and methods

`Identity()`, `Gradient()`, `Hessian()`, and `Laplacian()` are stateless.
`PartialDerivative(axis)` selects one coordinate axis, which is static under
JAX transformations. Numerical directions, displacements, radii, and smoothing
scales are dynamic array parameters.

::: gpex.operators
    options:
      members:
        - LinearOperator
        - QuadratureOperator
        - TensorQuadratureOperator
        - Identity
        - Increment
        - PartialDerivative
        - DirectionalDerivative
        - Gradient
        - Hessian
        - Laplacian
        - BoxAverage
        - GaussianSmoothing
        - CumulativeIntegral1D
        - HaarWavelet1D
        - LinearCombination
