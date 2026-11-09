# Design

gpex provides inference for one scalar Gaussian process through heterogeneous
linear measurements. Experimental design and Bayesian optimization consume this
inference; acquisition functions, constraints, stopping rules, and experiment
loops remain application code. Dense inference is the default; numerical and
statistical approximations are explicit.

This document owns architectural contracts and their rationale. Coding and test
rules are in [AGENTS.md](AGENTS.md); setup and checks are in
[CONTRIBUTING.md](CONTRIBUTING.md). User-facing details belong in the
[concepts](docs/concepts/linear-functionals.md) and [tutorials](docs/tutorials/index.md).

## Ownership

- `blocks/functionals.py` owns locations, individual functionals, and shared
  collection validation. `blocks/queries.py` owns prediction queries.
  `blocks/designs.py` owns measurement designs and reserved capacity.
  `blocks/datasets.py` attaches observed outcomes to designs.
  `blocks/indexing.py` owns static index maps and named array views.
  Each numerical value has one owner in the parameter tree.
- `kernels/base.py` owns the kernel interface and composition.
  `kernels/features.py` owns the explicit `FeatureKernel` capability.
  `kernels/radial.py` owns radial evaluation and the `RadialFeatureKernel`
  implementation shared by built-in Fourier-feature kernels.
  `kernels/separable.py` owns coordinate-product representations.
  `kernels/lengthscales.py` owns ordinary and generalized input scaling.
  `kernels/rbf.py` and `kernels/matern.py` own concrete families, their correlation
  functions, and frequency distributions. Matérn derivative rules stay with the
  Matérn family. `means.py` owns mean functions.
- `parameters/parameters.py` owns trainable values and prior traversal.
  `parameters/transforms.py` owns optimization transforms, including packed
  Cholesky coordinates. `parameters/priors.py` owns distributions independently
  of parameter-tree mechanics.
- `validation/checks.py` owns shared runtime checks and jitter validation.
  `validation/cholesky.py` owns matrix-factor structure and support checks shared
  by parameters, priors, and kernel lengthscales.
- `operators/base.py` owns the linear operator interface and weighted
  combinations. `operators/local.py` owns point evaluation and derivatives.
  `operators/differences.py` owns displaced function differences.
  `operators/quadrature.py` owns quadrature contracts and concrete rules.
- `moments/measurements.py` owns numerical functional grouping, activity, and
  mappings to public order. `moments/operator_covariance.py` owns kernel/operator
  dispatch. `moments/assembly.py` assembles means, full covariances, and covariance
  diagonals. Neither inference nor linear algebra inspects concrete kernel or
  operator classes.
- `gp.py` owns GP, conditioning state, and posterior state. `prepared_query.py` and
  `feature_paths.py` consume them through `PreparedQuery(conditioning, query)` and
  `FeaturePosteriorPath(posterior, key)`. Dependencies point toward `gp.py`.
  `covariance_systems.py` describes observation covariance systems and their
  updates for numerical backends. `results.py` owns structured prediction and
  covariance results.
- `linalg/base.py` owns numerical operator, prepared-system, update, and
  covariance-projection interfaces, array aliases, and symmetry handling.
  `linalg/dense.py` owns the dense Cholesky adapter and its concrete projections.
  Solver configurations prepare numerical systems. Prepared backends own
  factorization, solves, covariance reductions, likelihood statistics, and
  numerical updates. GP consumers depend on these operations, not factor layout.

Constructors establish invariants; consumers trust them. Checks protect against
silent wrong results or difficult failures, especially accidental broadcasting.
Constrained parameters own support validation. Numerical failures remain visible:
no input repair, solver substitution, altered-tolerance retry, jitter escalation,
or replacement by saved results is allowed.

## Measurement and result model

A `LinearOperator` maps a scalar function and a reference point to a fixed
per-point output shape. Each anchored scalar output component is a linear
functional. Local derivatives and nonlocal measurements share this contract.

| Type | Owned state and contract |
| --- | --- |
| `Locations` | Nonempty real floating array `(n, d)`, in JAX's default precision. |
| `FunctionalBlock` | Locations, operator, name; structured shape `(n, *operator_output_shape)`. |
| `ObservationDesignBlock` | Reserved functional locations, dynamic active prefix count and one finite nonnegative scalar noise variance, fixed or trainable, shared across independent errors in the block. |
| `ObservationBlock` | Design and outcomes padded to the reserved functional shape. `from_arrays` builds the same nested representation. |
| `FunctionalQuery`, `ObservationDesign`, `FunctionalDataset` | Ordered collections with unique block names. |

Block order and row-major flattening define the numerical vector. A static
`IndexMap` records reserved shapes and slices. Observation capacity counts
locations and defaults to `max(1024, supplied locations)`. Operator output sizes
determine the scalar reserve. Explicit capacity must accommodate all supplied
locations; exact-size capacity gives compact storage. Active counts are dynamic
integer leaves and may be zero. A new empty block requires a valid padding point.
Other unused locations repeat the first supplied point, avoiding invalid kernel
or operator evaluation and derivatives at an invented coordinate.

Observation blocks replace or append their contents within capacity, retaining
operator, name and shared noise. Overflow raises rather than changing shapes or
dropping data. Changing active counts preserves the index map. Unreserved
functional blocks and queries retain their supplied size. Posterior append
fills a declared named observation block. A new named block changes structure
and requires reconditioning.

Activity belongs to observation design, because conditional covariance does not
depend on outcomes. Prepared functionals retain its canonical scalar mask alongside
the grouped operator data. Moment assembly boundaries mask means, covariance rows and
columns and diagonals consistently. Kernel and operator evaluators do
not inspect capacity. Padded outcomes are zero and residuals exclude unused slots.
The numerical system supplies a unit diagonal shift for those slots, with zero
kernel covariance. Solvers receive numerical activity and update descriptions without inspecting
measurement types. The cholops factor owns numerical activity and factor
ordering, and ignores inactive RHS and cross-covariance entries. Posterior paths
also mask the simulated-noise residual. No large noise variance or extra jitter
is introduced.

Prepared numerical systems expose dynamic `size` for the active scalar count
and static `capacity` for the reserved array dimension. Input operators expose
the same `capacity`, and updates expose `previous_capacity`. Numerical inputs retain
the full capacity. Activation changes size without changing array shapes.

`StructuredValues`, `CovarianceResult`, `Prediction`, and `MarginalPrediction`
are PyTrees with dynamic numerical arrays and static index maps. Flat arrays
are the source of truth: named values and covariance blocks are views, and
variances derive from covariance diagonals. Complete results support outer
`jit` and `vmap`, including views on batched outputs.

## Operators and moment assembly

`LinearOperator` subclasses implement `output_shape(input_dimension)` and
`apply(function, point)`. The latter is JAX-compatible and linear in the
function. Applying operators to means and both kernel arguments gives every
measurement the same semantics throughout inference and design.

Moment preparation groups compatible operator PyTrees by structure, static settings,
and array-leaf shapes, never by numerical equality. Numerical operator parameters
are batched over locations and stay differentiable. Prepared functionals contain
grouped arrays and integer permutations; a separate static index map preserves
public order. Conditioning state retains a static map from public blocks to
their reserved rows in numerical groups. Appending fills those rows without
changing group structure. This map stays outside moment assembly inputs so equivalent
grouped measurements share covariance compilation despite different public block
boundaries. Fitting derives packing from current parameter values.

`QuadratureOperator.quadrature(point)` supplies absolute nodes `(q, d)` and
weights `(q,)`. Rules own domains, normalization, signs, and validation; node
counts are static. `TensorQuadratureOperator.coordinate_quadrature(point)`
supplies node and weight vectors per coordinate, defining their Cartesian
product. Nodes may depend on the entire query location.

`Kernel` and `Mean` have abstract evaluation contracts and sum/product
composition. `SeparableKernel.factorization(input_dimension)` supplies a scalar
coefficient and one two-argument factor per coordinate. Factors derive from
current parameters and need not be stationary. Scalar/ARD RBF supports this
contract; full-metric and general radial Matérn kernels do not claim it.

Type-based dispatch contracts separable kernels with tensor quadrature one
coordinate at a time. Sums recurse; generic combinations use nested operator
application. Every covariance entry point shares this evaluator. These are
exact contractions of the same discrete rule, up to summation order, not an
additional approximation. Two tensor rules need O(d q²) factor evaluations
instead of q^(2d) full-kernel evaluations. Generic kernels, means, and feature
maps still expand tensor grids. Kernels know factors, operators know rules,
and moment assembly depends only on their covariance interface.

`LinearCombination` requires matching per-location output shapes. Unrelated
locations need different geometry; heterogeneous outputs use separate blocks.
Hessians retain their full matrix shape for clarity, including duplicate mixed
partials. Redundant noiseless measurements require explicit noise or jitter.
One-dimensional integral and wavelet operators keep their dimensional limits.

## Conditioning, design, and sampling

`GaussianProcess.prepare_conditioning(design)` returns a `ConditioningState`
that owns the design and prepared noisy covariance system.
`GaussianProcess.condition(dataset)` returns a `ConditionedPosterior` that adds
outcomes and solved residual coefficients to its conditioning state. Gaussian
conditional covariance depends on the kernel, measurement design, and noise,
not observed values. Hypothetical design updates therefore change uncertainty
without inventing a future posterior mean.

Prediction offers full, marginal, mean-only, and variance-only operations.
Ordinary `predict_variance` assembles only the prior diagonal and backend
reduction, avoiding quadratic query storage. Full prediction necessarily stores
full query covariance.

`PreparedQuery(conditioning, query)` caches a full target covariance and an
observation-query projection for repeated independent proposals. Candidate
systems and cross-covariances reduce this cache. Rebuild it when the GP,
existing design, or target changes. Even variance-only prepared evaluations
retain the full target cache; they have a different memory contract from
ordinary variance prediction. Numerical locations and operator parameters
remain differentiable for design optimization.

`posterior.sample(key, query)` samples the finite Gaussian at a fixed query.
Its stabilized query covariance is factorized directly with cholops, while
gpex owns random draws and the transformation to the requested mean and
covariance.
`FeaturePosteriorPath(posterior, key)` constructs a persistent function draw
using a kernel-owned finite feature map and a conditioning correction. General
infinite-rank priors are approximated, while an exact finite feature map can
represent its prior exactly. The constructor requires `FeatureKernel`, which
declares `random_features` as an abstract operation. Ordinary `Kernel` has no
feature-construction method. `RadialFeatureKernel` shares Fourier construction
and requires a spectral sampler, leaving ordinary `RadialKernel` usable without
one. The conditioning correction accepts a zero-mean prior callable and is
independent of how its features are constructed.
Repeated calls share the draw and support differentiation and operators.
RBF uses Gaussian Fourier frequencies; Matérn 1/2, 3/2, and 5/2 use multivariate
t frequencies with 1, 3, and 5 degrees of freedom. Kernels apply their own
lengthscale transform. Paths depend on the feature capability rather than
concrete kernel classes. Kernels without that capability raise `TypeError`
before feature draws or conditioning work.
`GaussianProcess[K]`, `ConditioningState[K]`, and `ConditionedPosterior[K]` preserve
the concrete kernel type through preparation, conditioning, and append operations.
`FeaturePosteriorPath[K]` bounds that type by `FeatureKernel`. Kernel sums and
products currently supply only covariance evaluation, even when their operands
provide features. Finite-query sampling remains available for these kernels.

## Streaming

All state is immutable. `posterior.append(locations, values, block="name")`
fills unused locations in a declared block. `ConditioningState.append` takes
locations and the block name without outcomes. Both retain capacities, public index maps and
PyTree structure. The block fixes its operator and scalar noise variance.
Changing the kernel, mean, operator parameters, noise, jitter or block structure
requires reconditioning. Overflow raises under JIT instead of truncating or
resizing. Dataset edits remain separate from numerical posterior updates.

Dense preparation factorizes in canonical order without packing active rows.
A dynamic permutation owned by the cholops factor maps factor rows to
canonical scalar slots. Unused rows are independent identity entries. General
append moves the newly occupied slots to the end, preserving all other rows'
relative order, and fills those rows using a Schur complement. Moving independent
identity entries preserves triangularity without refactorization. Initial
conditioning therefore performs no rearrangement for a possible future append.

Solves and projections accept canonical arrays, and solves return canonical
arrays regardless of factor ordering. Both single-block trailing padding and
multiblock interspersed padding use cholops activation. Numerical operator
grouping is independent of factor ordering.

The numerical update describes canonical newly active indices, the updated
fixed-size system, old-to-new covariance and the small new covariance block.
Dense append uses a Schur complement and triangular solves over reserved size
M, filling existing factor storage and updating coefficients directly. For r
new scalar outputs its cost is O(M²r + Mr² + r³), quadratic for scalar append.
It does not materialize or factor the full covariance on each update. Inactive
RHS entries are ignored and returned inactive coefficients are zero.
`append_solve` requires unchanged old active RHS entries and delegates factor
and coefficient reuse to `PaddedCholesky.activate_solve`.

General replacement, removal and factor downdates are unsupported.
Prepared queries remain bound to their immutable conditioning state and must be
rebuilt to evaluate designs against an updated posterior.

## Parameters and fitting

`fitting/objectives.py` owns the exact likelihood and MAP objectives, exported
alongside the optimization routines from `gpex.fitting`. Objectives consume GP
state and numerical backends. Optimizers accept arbitrary scalar objectives.

Only explicit `Parameter` leaves are trainable; ordinary arrays stay fixed.
The base constructor accepts unconstrained coordinates. `PositiveParameter`
and `CholeskyParameter` validate constrained values and convert them to
optimization coordinates. Parameters live directly in model PyTrees, without
a separate registry.

Marginal likelihood excludes priors. Its Gaussian normalization uses the active
scalar observation count. The unused identity system contributes zero quadratic
form and zero log determinant. `negative_log_posterior` requires a prior
for every trainable parameter and subtracts their log densities on constrained
model values. Optimization transforms add no MAP Jacobian. A prior on an
unconstrained coordinate uses an identity-transformed parameter.

`fit` runs a supplied Optax optimizer in a fixed-step scan; `fit_lbfgs` owns
line search and gradient convergence. Both return the final iterate and reject
non-finite accepted states. The scan accumulates finiteness and checks once
after the loop to avoid per-step GPU error branches. L-BFGS rejects failed line
searches; trial objectives may be non-finite for the search to reject.
`fit_multistart` accepts a fitting callable and selects the lowest final objective.

Random-subset fitting is a separate Optax workflow. Subsets are drawn without
replacement from active locations within each block and use compact storage.
Each block must contain at least the requested subset count. Selection uses an
explicit full-data objective. A scaled subset
likelihood is not unbiased because GP likelihood does not decompose over
observations. L-BFGS requires a fixed objective for line search and curvature
history and does not use changing subsets.

## Numerical backends

[cholops](https://github.com/avrohr/cholops) provides the numerical implementation for the only backend shipped with
gpex: exact dense Cholesky inference. `DenseBackend` is the default solver
configuration. It prepares a `CholeskySystem`, the gpex adapter around cholops's
`PaddedCholesky`. gpex owns moment assembly and GP semantics. cholops
owns factorization, solves, whitening, quadratic forms, log determinants,
reserved-slot activation, and coefficient reuse.

The interfaces in `gpex.linalg` separate GP operations from this implementation
so future backends can provide a different numerical representation without
changing observation or prediction APIs. `GaussianProcess` accepts a backend
through `solver`. Its consumers use the prepared system's operations rather
than inspecting cholops state. Alternative backends must satisfy the full
interface, including activity, updates, differentiation, and JAX transformation
contracts. This boundary is an extension point, not a claim that additional
solvers are currently provided.

`SolverConfiguration.prepare` accepts a numerical symmetric operator exposing
`matmul`, `diagonal`, and tiled `materialize`, plus numerical activity information. It returns an immutable
`LinearSystem` supporting `solve`, `quadratic_form`, `logdet`, `project`,
`covariance_reduction`, `variance_reduction`, `append`, and `append_solve`.
Update descriptions supply canonical indices and trailing covariance blocks.
These interfaces do not depend on observation types.

`SolverConfiguration`, `LinearSystem`, `SymmetricLinearOperator`, and
`LinearSystemUpdate`, together with `CovarianceProjection`, are abstract base
classes. Implementations inherit their
interface explicitly and implement all abstract operations. GP numerical
descriptions and prepared backend state also inherit `eqx.Module`.
The input interface lets backends apply the operator or request tiled
materialization without depending on GP measurement types. Update descriptions own their numerical
system and newly active indices as ordinary fields.

Likelihood objectives use the configured backend's quadratic form and log
determinant. They retain the active observation count and prior semantics in
the GP layer. Backends preserve non-finite statistics for invalid fitting trials
so an optimizer can reject them. The dense quadratic form uses one triangular
solve, without computing the full inverse action.

The `CovarianceProjection` interface exposes Gram, diagonal, and cross
reductions for targets of the same prepared system. The dense implementation,
`CholeskyProjection`, caches a whitened cross-covariance. Consumers need no
access to a backend's factor ordering or representation.

`CovarianceSystem.matmul(rhs, tile_size=128)` accumulates covariance tiles
against canonical vector or multi-column inputs without allocating a full
observation matrix. `diagonal()` evaluates only the covariance diagonal. Both
include block noise, jitter, and independent unit entries for inactive slots.
Operator multiplication passes inactive RHS entries through, while prepared
solves ignore inactive inputs and return zero inactive coefficients.
`moments.apply_covariance` implements the latent part, masking inactive entries
and translating between canonical and operator-group order.

A tile never splits one operator output, so its scalar width may exceed the
requested tile size for a vector-valued operator. The multiplication checkpoints
both row traversal and individual tiles to recompute pairwise intermediates
in reverse mode. Pairwise storage stays bounded in forward and backward
computations. General kernels still require quadratic pairwise work per
application. Dense preparation continues to materialize and factorize the
matrix. No iterative solver is provided. An operator product alone does not
supply an efficient exact log determinant, and approximate determinants must
not silently change the exact likelihood contract.

The dense backend supports exact inference, MLE/MAP fitting, design, and
streaming. cholops's symmetry-preserving derivative rule is used for the initial
factorization and append Schur complements. Supported numerical differentiation
is first order in forward and reverse mode.

Prepared state stores one `PaddedCholesky` and the materialization tile size.
cholops owns its activity mask, both directions of the permutation, and
transposed factor storage across JIT boundaries. The cholops factor exposes its
lower-triangular array and ordering maps. Padded consumers ignore inactive
inputs, including non-finite unused entries, and solves return zero inactive
coefficients.
Fully active matrices use the same factor with an all-true activity mask.
gpex supplies covariance blocks and invokes `activate` or `activate_solve`
under the highest matrix multiplication precision, without imposing a dtype.
Covariance tiling bounds pairwise intermediates but does not remove quadratic
dense storage.
Failed Cholesky factorizations propagate NaNs. Explicit factorization exceptions
incur substantial tracing overhead; callers check outputs, while fitting
validates accepted states. A finite factor is not a rank certificate: rounded
singular systems can have small positive pivots.

The configurable backend prepares the observation system. Finite-query sampling
in `posterior.sample` separately factorizes the query covariance directly with
cholops. Replacing the observation backend does not replace that sampling path.

## Numerical conventions

- Built-in transforms and priors explicitly inherit the `Transform` and `Prior`
  protocols. Their abstract methods require complete explicit subclasses, while
  custom objects implementing the same contracts remain structurally accepted.
- Observation designs own one scalar measurement-error `variance`, fixed or
  trainable. `ObservationBlock.from_arrays` accepts the same `variance` parameter.
  A block's `variances` repeats its numerical value over the output shape.
  Collections expose `variance_flat` in canonical scalar-output order.
- Statistical observation noise is separate from numerical factorization
  jitter.
- Recommend enabling `jax_enable_x64` before importing gpex and constructing
  arrays. The library never changes or enforces the global setting. Floating
  arrays, random draws, and work buffers use JAX's default dtype; callers must
  use it consistently across model arrays, observations, queries, solve inputs,
  and streaming updates. gpex does not cast or reconcile floating dtypes and has
  no per-object precision options. cholops validates supported factor dtypes and
  compatible solve inputs. Mixed dtypes are unsupported.
  Both float32 and float64 are tested in separate processes, including example
  subprocesses; float64 remains recommended for ill-conditioned systems, small
  posterior variances, and tight optimization tolerances. Accuracy budgets and
  representable-range stress cases may depend on precision in tests only.
  Locations and means require real floating values by contract;
  custom means must return a scalar. Integer indices and boolean state retain
  their semantic types.
- Fitting checks finiteness of parameters and gradients entry by entry.
  Gradient norms use scaled reductions to avoid unnecessary overflow and
  underflow. A finite vector can still have a norm outside its dtype's range;
  that diagnostic alone does not make its entries invalid.
- Convergence tolerances depend on the objective and numerical conditioning.
  Solver defaults and failure criteria are shared by both precisions. There are
  no automatic tolerance changes, retries, dtype promotion, or jitter escalation.
  Explicit regularization changes the numerical system and must be kept
  consistent between fitting and inference. Tests use justified tolerances and
  verify convergence. Fitting failure checks stay active.
- Critical dot products in kernel input scaling, quadrature, prediction, sampling,
  objectives, gradient norms, paths, and dense updates request the highest
  multiplication precision without changing stored dtypes. This avoids
  reduced-precision accelerator multiplication in cancellation-sensitive reductions.
- Numerical system descriptions and GP covariance results are symmetrized to
  preserve their symmetric-matrix contracts. Cholesky input symmetrization is
  owned by cholops, including for sampling and append Schur complements. Its
  symmetry-preserving reverse rule handles gradient intermediates separately.
- Standard RBF and Matérn kernels use scalar or axis-wise ordinary lengthscales.
  Each generalized variant takes a complete lower-triangular precision factor
  `L` with positive diagonal: `M = L @ L.T` and distance `||L.T @ (x-y)||`.
  Ordinary lengthscales divide coordinates. Generalized lengthscales multiply
  them by the transposed precision factor.
  Built-in radial kernels do not accept arbitrary input metrics.
  Each concrete radial kernel selects a correlation and spectral law on
  `RadialKernel`. Its `lengthscales` field is a `Lengthscales` or
  `GeneralizedLengthscales` object owning the lengthscale or precision parameter
  once. Point evaluation and random features use the same transform convention.
  Radial and separable are overlapping contracts: scalar/ARD RBF implements both,
  while full-metric RBF and radial Matérn provide only the radial contract.
- `CholeskyParameter` packs only free lower-triangle entries, with softplus on
  the diagonal. It participates in ordinary parameter partitioning and prior
  traversal. Fixed factors are validated by kernel construction; constrained
  parameters own and preserve their invariants.
- Matrix priors are densities with respect to the free constrained entries of
  `L`. `IndependentCholeskyPrior` combines log-normal diagonals and normal
  strict-lower entries. `ScaledLKJCholeskyPrior` separates positive row scales
  from a correlation factor and includes the conversion Jacobian. Neither adds
  a Jacobian for the optimization transform; MAP targets the density on `L`.
- Matérn kernels use analytic radial derivative rules to preserve their valid
  coincident-point spatial derivatives without distance clamping. Matérn 1/2
  supports no latent derivatives, 3/2 supports first derivatives, and 5/2
  supports second derivatives. Covariance entry points check each operator's
  `derivative_order` against the kernel's `max_derivative_order`. Custom
  differential operators and kernels with finite smoothness declare these
  properties; purely value-based operators need no declaration.
- Covariance blocks must share an input dimension, including across training
  and query collections. An ARD lengthscale vector must have exactly that size;
  a scalar lengthscale is isotropic.
- Public array-facing signatures use jaxtyping axis names as documentation and
  static-analysis aids, without runtime shape conversion.

## Test value and scope

The test suite goal is to protect desired behavior with minimal runtime and maintenance burden.
Every test should add meaningful protection and expensive tests require proportionately
greater value. Remove redundancy and constraints on harmless implementation
choices while preserving numerical guarantees.

## JAX execution model

Inference operations are pure and composable rather than imposing a JIT
boundary on every method. Performance-sensitive callers should JIT the outer
workload so covariance construction, factorization, prediction, and design
criteria compile together. Programs specialize to array shapes and static block
structure. Observation blocks keep reserved shapes while their active counts
change. `condition = eqx.filter_jit(gp.condition)` accepts different datasets
with matching capacities, block structure, numerical shapes and dtypes without
recompiling. The compiled method captures the immutable GP.
Passing the GP explicitly to an outer compiled function allows its numerical
parameters to vary too. Keep the compiled callable rather than recreating it.

Reserved conditioning recomputes the numerical system for the current contents.
Dense factorization costs cubic work and quadratic storage in total reserved
scalar size, not active size. Fixed-capacity posterior append reuses numerical
state and compiled code for a fixed target block and batch shape. Changing
batch shapes or static structure can
compile again. Queries remain unreserved. Changed query shapes, block structure
or capacities still specialize compiled functions. Likewise, rebuilding blocks
from varying-size input arrays can specialize their construction even though
conditioning reuses the resulting reserved shapes.

Valid numerical model values can be constructed inside JAX transformations,
including kernel parameters, operator widths, noise, locations, and outcomes.
Names, shapes, quadrature orders, solver limits, and prior hyperparameters are
static. Constructors retain necessary support checks; fitting evaluates trials
by updating existing parameter trees rather than constructing invalid models.

Conditioned geometries, posteriors, paths, and prepared systems are Equinox
PyTrees. Their arrays cross an outer JIT boundary as dynamic state rather than
being embedded as static constants. Fitting partitions explicit parameters
from fixed leaves, carries optimizer state through JAX control flow, and
vectorizes independent starts with `jax.vmap`. Fixed-step fitting supports
reverse-mode differentiation through the scan. L-BFGS uses a dynamic while
loop, which stops objective work once all starts have finished but does not
support reverse-mode differentiation through the complete fit. Its per-start
evaluation counts describe active trajectory work, not all vectorized device
computation.

Numerical failure checks must survive `jit` and `vmap` when a caller consumes
computed numerical results, including a single fitted parameter or loss.
Selecting only constant counters, metadata, or unchanged input arrays may let
JAX eliminate unused numerical work and its checks. Those fields do not force
an otherwise unused fit or inference computation to execute.

Performance measurements must synchronize asynchronous JAX work and separate
compilation from warm execution. Benchmark methodology
belong in [`docs/benchmarks.md`](docs/benchmarks.md), not in this design record.

## Dependencies

The runtime uses JAX, Equinox, Optax, and cholops with NumPy, jaxtyping, and
Plum for operator-covariance dispatch. GPJax and
GPyTorch are optional benchmark comparisons; their APIs and dependencies stay
outside the inference package.

Dependency declarations and `uv.lock` change together. Test the locked versions,
including comparison workloads when those dependencies change. Setup and CI
commands are maintained in [CONTRIBUTING.md](CONTRIBUTING.md) and the workflows.
