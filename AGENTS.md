# Working on gpex

gpex provides JAX/Equinox inference for a scalar Gaussian process observed
through heterogeneous linear measurements. Acquisition strategies and experiment
loops belong in application code.

[DESIGN.md](DESIGN.md) owns architectural and numerical contracts.
[CONTRIBUTING.md](CONTRIBUTING.md) owns setup, checks, and release procedures.
Read the relevant contracts before changing behavior.

## Implementation

- Prioritize readability and measured performance, then brevity. Remove
  unnecessary work, state, branches, and wrappers before adding abstractions.
  Classes own coherent state; helpers name meaningful operations or remove
  real duplication. Keep one implementation per contract.
- Constructors own invariants and derived state. Consumers trust them and
  validate new relationships or computed state where needed. Keep checks that
  prevent silent wrong results or explain difficult failures; accept adequate
  Python/JAX exceptions. Never assert traced numerical values.
- Numerical failure stays visible. Do not repair inputs, switch solvers,
  increase jitter, retry with altered settings, or substitute saved results.
  A line search may reject non-finite trial objectives.
- Keep dependencies one-way: design and paths consume GP state; linear algebra
  consumes numerical systems; kernels own input geometry and random features.
  Do not conceal cycles with local imports or add forwarding factories.
- Keep public signatures typed, with useful jaxtyping shapes. Docstrings state
  units, shapes, invariants, and supported JAX transformations. Comments explain
  reasoning rather than restating code.

## Numerical discipline

- Preserve immutable state, scalar noise variance per block, unique block names,
  canonical flattening, and static structural metadata. Numerical arrays are
  dynamic PyTree leaves; complete public results must cross an outer `jit`.
- Train only explicit `Parameter` leaves. Priors describe constrained values;
  optimization transforms add no MAP Jacobian. Marginal likelihood excludes priors.
- Preserve dedicated variance-only prediction and backend reductions. Ordinary
  `predict_variance` must not construct full query covariance. Prepared queries
  explicitly cache it and must be rebuilt when their inputs change.
- Recommend x64 before importing gpex and constructing arrays. Never change or
  enforce that setting in the library. Use JAX's default floating dtype without
  casts, promotion, dtype checks, or per-object precision settings. Mixed dtypes
  are unsupported. Integer indices and boolean state retain their semantic types.
- Check finiteness entry by entry; norms can overflow for finite vectors. Use
  scaled reductions where needed. Numerical failures must survive `jit` and
  `vmap` when computed results are consumed, including one fitted parameter or
  loss. Constant metadata and unchanged inputs need not force unused work.
- Keep solver accuracy and kernel derivative limits explicit. Preserve bounded
  memory in backward as well as forward computations.

## Tests and performance

- Keep a clean, focused, efficient, concise suite. Every test must protect
  desired behavior or detect a failure that matters. Ask what unacceptable
  change it would catch and what protection it adds beyond existing tests.
  Test counts and coverage percentages do not establish value or justify deletion.
- Remove redundant and uninformative tests even when they are cheap: they add
  maintenance burden and obstruct harmless changes. Do not test internal behavior
  that could change without violating a numerical, behavioral, or resource
  contract. A test of the current implementation is not itself evidence of value.
- Prefer independent references, gradients, convergence, failure visibility,
  and meaningful resource bounds. Loss improvement alone does not establish
  convergence. Shape or PSD checks do not establish accuracy.
- Expensive tests must provide proportionately greater, distinct value. Identify
  the important failure they catch, which part of the test is expensive, and why
  a cheaper check cannot establish the same claim. Use the smallest convincing
  problem; justify costly sizes or parameter matrices near the fixture. Preserve
  conditioning, statistical, and memory stress cases that cheaper checks miss.
- Test shared validation once. Parameterize independently failing paths such as
  eager/JIT and full/variance-only prediction. Remove overlapping tests only after
  verifying the retained test protects the complete claim, including full covariances.
- Keep tests direct and readable. Do not pin incidental representations or exact
  helper-call sequences. Avoid timing-only harness complexity; compiling
  expensive independent references is acceptable. Improve the suite rather than
  tuning library code to a CI runner.
- Reduce CI cost by removing tests without value and replacing unnecessarily
  expensive checks with cheaper convincing ones. Running a valuable, necessarily
  expensive test only locally or before releases is a last resort. Moving a
  low-value test out of CI does not justify keeping it.
- Run float32 and float64 in separate processes, selecting `JAX_ENABLE_X64`
  before collection. Examples inherit it. Keep strong float64 references and
  justify float32 tolerances independently; precision-dependent stress scales
  and accuracy budgets belong in tests, never library algorithms.
- Measure representative performance with synchronized JAX results and separate
  compilation and warm execution.

## Changes and documentation

- Use the locked environment and start with checks relevant to the change.
  Follow the CI workflows for full validation. Documentation-only changes need
  documentation checks, not a new numerical test run.
- Update dependency declarations and `uv.lock` together; verify `uv lock --check`
  and installed versions. Dependency upgrades must pass the CI benchmark smoke
  command. Comparison libraries stay in the optional comparison group and
  benchmark workload scripts. Keep compared libraries' timed operations together
  in the same workload file.
- Update the relevant guide with behavior changes and DESIGN.md with ownership
  or contract changes. Describe the supported system directly; retain rationale
  without migration narratives or references to superseded implementations.
- Introduce the problem and unfamiliar quantities before a small useful example.
  Keep the homepage an overview; put detail in focused guides and executable
  notebooks. Check notebooks from fresh kernels with `scripts/verify_tutorials.py`.
- Update CHANGELOG.md during release preparation or when explicitly requested.
  Keep agent guidance focused on durable project rules.
