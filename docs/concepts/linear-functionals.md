# Gaussian processes with linear functionals

gpex models one unknown scalar-valued function $f(x)$, where
$x \in \mathbb{R}^d$ is an input location. This is the **latent function**:
the underlying function whose values, derivatives, or averages we may observe
and want to predict. A Gaussian process (GP) describes uncertainty about it:

$$
f \sim \mathcal{GP}(m, k).
$$

Here $m(x)$ is the mean of $f(x)$, and the kernel $k(x, x')$ is the covariance
between $f(x)$ and $f(x')$ at two input locations. The GP assumption means that
the function values at any finite collection of locations are jointly Gaussian.

## Linear transformations of a GP

A linear operator $L$ maps a function $f$ to another function $Lf$. For
example, the identity gives $(Lf)(x)=f(x)$, a partial derivative gives
$(Lf)(x)=\partial f(x)/\partial x_r$, and a local averaging operator gives
the average of $f$ over a region around $x$.

Linearity refers to the operator's action on functions. For functions $u,v$
and scalar coefficients $a,b$, it means

$$
L(au+bv) = aLu+bLv.
$$

Neither $f$ nor $Lf$ needs to be a linear function of the input $x$.
The operator and its parameters are fixed independently of the realization
of $f$.

A GP remains Gaussian under a linear transformation, provided the transformed
quantities are well defined in mean square. In particular, derivatives require
sufficient kernel smoothness, and integrals require finite second moments.
The transformed process has distribution

$$
Lf \sim \mathcal{GP}(Lm,\, L_x L_{x'}k).
$$

The transformed mean is obtained by applying $L$ to the original mean
function. Its covariance kernel is obtained by applying $L$ to **both**
arguments of the original kernel:

$$
m_L(x) = (Lm)(x), \qquad
k_L(x,x') = L_x L_{x'} k(x,x').
$$

The subscripts indicate which argument the operator acts on. Here $x'$ is a
second input location, not a derivative. This is the process analogue of
transforming a Gaussian vector $z$ by a matrix $A$: its mean becomes
$A\mathbb{E}[z]$ and its covariance becomes $A\operatorname{cov}(z)A^\top$.

For example, for a sufficiently smooth one-dimensional GP, the derivative is
itself a GP:

$$
f' \sim \mathcal{GP}\!\left(m',\,
\frac{\partial^2 k(x,x')}{\partial x\,\partial x'}\right).
$$

An operator can also produce several components at each location. The gradient
$\nabla f$ is a vector-valued GP derived from the scalar latent function. Its
covariance is matrix-valued, with one entry for each pair of gradient
components.

## Different quantities share one latent function

Two linear transformations $Lf$ and $Mf$ are jointly Gaussian, with
cross-covariance

$$
\operatorname{cov}[(Lf)(x),(Mf)(x')] = L_x M_{x'}k(x,x').
$$

For example, the covariance between a function value and a derivative in one
dimension is

$$
\operatorname{cov}[f(x),f'(x')] = \frac{\partial k(x,x')}{\partial x'}.
$$

These cross-covariances allow a value observation to inform a derivative
prediction and conversely. Values, derivatives, and averages are coupled
through the same latent function. gpex constructs their joint Gaussian
distribution from the operators and the shared kernel, so mixed measurements
can all contribute to one posterior over $f$.

Fixing a location in $(Lf)(x)$, and selecting one component if needed, produces
a scalar **linear functional** of $f$. If $L_i$ and $L_j$ denote two such
functionals, then

$$
\mathbb{E}[L_i f] = L_i m, \qquad
\operatorname{cov}[L_i f,L_j f] = L_i L_j' k.
$$

In this notation the prime on $L_j'$ marks action on the second kernel
argument. For point evaluations $L_i f=f(x_i)$ and $L_j f=f(x_j)$, the
covariance reduces to $k(x_i,x_j)$.

## What an operator looks like in gpex

`GaussianProcess` specifies the mean and kernel of the latent function $f$.
A `LinearOperator` describes the transformation $L$ through two methods:

- `output_shape(input_dimension)` declares the output shape at one location.
- `apply(function, point)` computes $(L\,\text{function})(\text{point})$.

For example, the built-in `Gradient` operator differentiates the supplied
callable with JAX:

```python
from collections.abc import Callable

import jax

from gpex.operators import LinearOperator


class Gradient(LinearOperator):
    @property
    def derivative_order(self) -> int:
        return 1

    def output_shape(self, input_dimension: int) -> tuple[int, ...]:
        return (input_dimension,)

    def apply(
        self, function: Callable[[jax.Array], jax.Array], point: jax.Array
    ) -> jax.Array:
        return jax.jacfwd(function)(point)
```

For a scalar callable and a point of shape `(d,)`, `apply` returns a gradient
of shape `(d,)`. The `derivative_order` property declares the smoothness
required of the latent GP. The identity operator instead returns
`function(point)` and declares scalar output shape `()`.

During inference, gpex applies these operators to the mean function and to
the kernel arguments to construct the transformed means and covariances.
Differential operators use automatic differentiation, while built-in integral
and averaging operators use explicit quadrature rules.

A `FunctionalBlock` combines an operator with locations and a name. At $n$
locations in $d$ dimensions, `Identity()` gives a block of shape `(n,)`, while
`Gradient()` gives `(n, d)`. These blocks describe the finite collections of
functionals used for observations and queries. See
[Getting started](../getting-started.md) and
[Learning a function from derivatives](../tutorials/mixed-measurements.ipynb)
for their use in inference, or
[Implementing custom operators](../tutorials/custom-operators.ipynb) for
a complete extension example.

## Consequences for modelling

The operator affects what the data identify. Gradients and increments, for
example, do not identify a constant offset on their own. Averaging operators
attenuate short-scale structure. Derivative observations also require the
kernel to possess the corresponding mean-square derivatives.

See [Linear operators](linear-operators.md) for the
built-in operators and their conventions.
