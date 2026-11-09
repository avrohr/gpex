# Linear operators

An operator describes what a measurement tells us about the unknown function,
or which quantity we want to predict. For a spatial temperature field, for
example, we might observe the temperature at a point, an average over a sensor's
footprint, or the rate of change in a particular direction.

gpex supports all of these as observations and prediction targets. They can
be mixed in one model because they are linear transformations of
[one latent Gaussian process](linear-functionals.md). The operator describes
the measured quantity, while observation noise describes uncertainty in the
reading.

## Built-in operators

Write $f$ for the latent function and $x$ for a location in $d$ dimensions.
For the averaging formulas, $B_r(x)$ is the box centered at $x$ with
half-widths $r$, and $\varphi_s$ is a centered Gaussian density with independent
coordinates and standard deviations $s$. A scalar radius or scale uses the
same width in every coordinate.

The integral formulas describe the continuous quantities. gpex approximates
these integrals with numerical quadrature.

| Operator | Quantity at each location | Output |
| --- | --- | --- |
| `Identity()` | $f(x)$ | scalar |
| `Increment(h)` | $f(x+h)-f(x)$ | scalar |
| `PartialDerivative(j)` | $\dfrac{\partial f(x)}{\partial x_j}$ | scalar |
| `DirectionalDerivative(v)` | $v^\top \nabla f(x)$ | scalar |
| `Gradient()` | $\nabla f(x)$ | $d$-vector |
| `Hessian()` | $\nabla^2 f(x)$ | $d\times d$ matrix |
| `Laplacian()` | $\operatorname{tr}\nabla^2 f(x)=\sum_j \dfrac{\partial^2 f(x)}{\partial x_j^2}$ | scalar |
| `BoxAverage(r)` | $\dfrac{1}{\operatorname{vol}(B_r(x))}\int_{B_r(x)} f(u)\,\mathrm{d}u$ | scalar |
| `GaussianSmoothing(s)` | $\int_{\mathbb{R}^d} f(x+u)\,\varphi_s(u)\,\mathrm{d}u$ | scalar |
| `CumulativeIntegral1D(b)` | $\int_b^x f(t)\,\mathrm{d}t$ | scalar |
| `HaarWavelet1D(s)` | $\dfrac{1}{\sqrt{s}}\left[\int_{x-s/2}^{x} f(t)\,\mathrm{d}t-\int_x^{x+s/2} f(t)\,\mathrm{d}t\right]$ | scalar |
| `LinearCombination(...)` | $\sum_j c_j(L_jf)(x)$ | matches components |

The output is the quantity at one location. A vector or matrix output still
comes from the same scalar latent function.

## Values, changes, and curvature

`Identity` represents an ordinary point measurement. `Increment` measures
the total change between two points separated by a displacement $h$. It does
not divide by the distance between them, and it can be used even when the
latent function is not differentiable.

Derivatives describe local rates of change. `PartialDerivative` selects one
coordinate, `DirectionalDerivative` selects a direction $v$, and `Gradient`
collects all coordinate derivatives. A unit vector $v$ gives the rate per unit
distance in that direction. Scaling $v$ scales the directional derivative.

`Hessian` describes local curvature, including mixed derivatives that relate
changes along different coordinates. `Laplacian` sums the coordinate second
derivatives into one scalar, as used in diffusion and Poisson equations.
These operators let gpex incorporate slope or curvature measurements and
predict them from other observations, such as function values.

Derivative observations and predictions require a sufficiently smooth latent
GP. First derivatives need less smoothness than Hessians or Laplacians. See
[Choosing a kernel](../api/kernels.md#built-in-kernel-families) for the
supported derivative orders.

## Measurements over a region

`BoxAverage` models a measurement with a finite footprint, such as a pixel or
a sensor that averages over a rectangular area. Every part of the box has
equal weight. The radius $r$ is its half-width, so a one-dimensional box spans
from $x-r$ to $x+r$.

`GaussianSmoothing` gives more weight to nearby values, with the weight
decreasing smoothly with distance. This can describe a blurred measurement
without a sharp boundary. Its scale $s$ is the Gaussian standard deviation.
Both averaging operators preserve constant functions and have the same units
as $f$. Their widths set the spatial scale over which variation is averaged.

`CumulativeIntegral1D` measures an accumulated quantity from a fixed base $b$
to the query location $x$, such as a total amount obtained by integrating a
rate. Its value is zero at $b$, and reversing the integration limits reverses
the sign.

`HaarWavelet1D` measures a contrast between the left and right halves of an
interval centered at $x$. Its scale $s$ is the full interval width. The
coefficient is positive when the left half has the larger average, negative
when the right half does, and zero for a constant function. The factor
$1/\sqrt{s}$ gives the analyzing wavelet unit $L^2$ norm.
Both the cumulative integral and Haar wavelet are restricted to
one-dimensional inputs.

gpex evaluates these regional measurements using weighted function
evaluations, called quadrature. Increasing the quadrature order uses more
evaluation points. Accuracy should be checked at the spatial scales
relevant to the model. [Learning from averages](../tutorials/nonlocal-measurements.ipynb)
shows why accurately measured averages can leave individual function values
uncertain.

## Factored tensor quadrature

A `SeparableKernel` expresses a covariance as a product of one-dimensional
kernels. A `TensorQuadratureOperator` describes a quadrature rule through nodes
and weights for each coordinate. When these interfaces meet, gpex evaluates
one-dimensional quadrature sums and multiplies the results.

For two rules with `q` nodes along each of `d` coordinates, this needs
`d * q²` coordinate-kernel evaluations rather than `q^(2d)` full-kernel
evaluations. `RBFKernel` provides the separable representation for scalar and
ARD lengthscales. `BoxAverage` and `GaussianSmoothing` provide tensor rules.
Custom implementations of these interfaces receive the same optimization,
as do separable components of kernel sums.

This rearranges the same quadrature approximation. It does not remove its
integration error. Generic kernels and kernel products use the general
calculation. Mean and feature evaluation still expand the full tensor grid.
The optimization also applies when a tensor rule occurs on only one side of
a covariance. See the [kernel interfaces](../api/kernels.md#extension-interfaces)
and [quadrature interfaces](../api/operators.md#quadrature-interfaces).

## Combining operators

A single reading may combine several effects. `LinearCombination` forms one
operator from component operators $L_j$ and fixed coefficients $c_j$. For
example, a sensor that responds to both the function value and a slope can be
described by

$$
(Lf)(x) = a f(x)+b\frac{\partial f(x)}{\partial x_j}.
$$

The component operators must have matching outputs: scalars can be added to
scalars, or vectors of the same size to each other. To model separate readings
of different quantities, use separate
[observation blocks](observations-designs-queries.md). Those readings can have
different operators, locations, and noise levels while informing one posterior.

## What the measurements can reveal

Derivative measurements do not tell us the function's vertical offset. We can
shift a function up or down without changing any of its derivatives. Increments
and Haar coefficients have the same limitation. A function value or an average
measurement helps pin down that offset.

An average can hide small fluctuations. A flat function and one that rises
and falls around the same value can have the same average over a region.
Measuring that average does not tell us which function we have. The kernel
describes how much fluctuation the GP expects, while measurements at individual
points or over smaller regions can help reveal it.

See [Learning a function from derivatives](../tutorials/mixed-measurements.ipynb)
for the effect of adding a single value to derivative measurements, and
[Learning from averages](../tutorials/nonlocal-measurements.ipynb) for the
variation left uncertain by window measurements.
Other linear measurements can be expressed with
[custom operators](../tutorials/custom-operators.ipynb). Constructor arguments
and method details are listed in the [operator API reference](../api/operators.md).
