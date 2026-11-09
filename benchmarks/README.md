# Performance measurements

Each workload has a short executable script. Compared libraries' operations
are in the same file. Shared code handles process isolation, timing, model
settings, numerical checks, and plotting.

| Script | Measured operation |
| --- | --- |
| [dense.py](dense.py) | Compact conditioning and prediction in gpex, GPJax and GPyTorch |
| [datasets.py](datasets.py) | Recondition 100 different datasets, comparing reserved gpex, compact gpex, GPJax and GPyTorch |
| [sequential.py](sequential.py) | Grow a posterior from 512 to 1,024 observations with gpex append/reconditioning, GPJax reconditioning and GPyTorch incremental updates |
| [mixed.py](mixed.py) | Condition on values, slopes and averages |
| [fitting.py](fitting.py) | Complete 100-step MLE and MAP fits |
| [design.py](design.py) | Prepared design criterion and coordinate gradient |
| [prediction.py](prediction.py) | Full covariance or marginal prediction from an existing posterior |

Each script accepts the runner's options, including `--plan`, `--dtype`,
`--backend`, `--process-samples`, and `--output`. Optional diagnostics remain
in `single_update.py`, `construction.py`, `features.py`, `gradients.py`, and
`sampling.py`. The single-update diagnostic and unprepared design remain
available explicitly, but are not part of the default run.

`run.py` owns workers, provenance and resume. `workloads.py` times single JAX
operations. `compare.py` supplies optional comparison models and eager timing.
`_sequence.py` times complete sequences. These helpers contain no alternative
workload implementations.

## Compilation reuse and sequential updates

```bash
# Retain the compact baseline with first-call and warm timings.
uv run --frozen python benchmarks/dense.py --sizes 16 32 64 128 256 512 \
  --process-samples 3 --output results/dense.json

# Capacity 512. Both size schedules contain 100 conditioning calls in total.
uv run --frozen python benchmarks/datasets.py \
  --process-samples 3 --output results/datasets.json

# Capacity 1,024. Initial conditioning at 512, followed by 512 scalar updates.
uv run --frozen python benchmarks/sequential.py \
  --process-samples 3 --output results/sequential.json

uv run --frozen python benchmarks/plot.py \
  --results results/dense.json results/datasets.json results/sequential.json \
  --output-dir results/figures
```

Install the comparison group as described below. These commands default to
float64 on CPU. Run float32 and GPU in separate invocations with distinct output
paths. GPU execution requires an allocated GPU and never falls back to CPU.
No script submits cluster jobs.

**Different datasets.** `datasets.py` uses the same fixed-seed input schedule
in every library. It visits 100 distinct sizes spanning 16–512 in shuffled
order. Every dataset has new locations and
values, so numerical posterior state cannot be reused. Data construction,
padding and transfer are outside timing for every method. The gpex reserve is
exactly the largest dataset size. There is no extra-capacity sweep.

**Sequential observations.** `sequential.py` conditions at 512 observations,
then adds one reading and predicts at each size through 1,024. Initial
conditioning, prediction-cache preparation, buffer insertion and subsequent
updates are timed. Hyperparameters and noise stay fixed. GPyTorch uses
`get_fantasy_model` with dense settings and retains its caches. GPJax
reconditions the growing dataset. gpex compares append with reconditioning
inside the same capacity. Both sequences predict one point by default.

The optional `sequential-multiblock` and `sequential-rebuild-multiblock` variants split
the same readings alternately between two declared blocks, each with half the
total capacity. They exercise interspersed padding and alternate append targets
while keeping the GP model and observation sequence identical. Each static
target block compiles its own update executable once.
These are gpex-specific diagnostics, not comparisons with other libraries.
They are excluded from the default run and profiling campaign. Select them
explicitly with `--workloads sequential-multiblock sequential-rebuild-multiblock`.

Each **entire sequence** runs in one fresh process. Compilation and posterior
caches survive between its steps. Three process samples repeat the whole
sequence three times, not each step three times. Timings include the first
compilation and synchronize returned state and predictions. Validation runs
outside each measured interval. Reports retain per-step and cumulative times,
initial conditioning time, total time, JAX trace counts, predictions, and peak
memory. Trace counts count outer function traces, not individual XLA kernels.
A tiny unrelated numerical problem initializes the runtime before sequence
preparation. There is no workload warm-up pass or cache reset inside a sequence. `--repeats` applies
only to single-operation workloads.

The cumulative plots and sequence summary distinguish these totals from warm
per-call timings. A sequence with any failed step is rejected, while its
partial timings and outputs remain available. The existing finite-output and
variance checks apply without cross-library comparisons or new tolerances.

For small local checks, use `--sizes 6 --min-size 2 --sequence-length 4` with `datasets.py`, or `--sizes 6 --start-size 2` with
`sequential.py`. `--plan` prints the complete cases without running them.

## Run locally

Use the locked environment. Comparison libraries are optional.

The comparison group pins GPJax to upstream commit `964bad7`, which removes
the JAX version cap in its 1.0.0 release. Use `uv sync` to install this pinned
source alongside JAX 0.11.2.

```bash
uv sync --locked --all-groups

JAX_DEFAULT_MATMUL_PRECISION=highest uv run --frozen python benchmarks/run.py \
  --dtype float32 --process-samples 3 --label "Local CPU" \
  --output results/performance/cpu-float32.json
JAX_DEFAULT_MATMUL_PRECISION=highest uv run --frozen python benchmarks/run.py \
  --dtype float64 --process-samples 3 --label "Local CPU" \
  --output results/performance/cpu-float64.json

uv run --frozen python benchmarks/plot.py \
  --results results/performance/cpu-float32.json results/performance/cpu-float64.json \
  --output-dir results/performance/figures
```

Open `results/performance/figures/report.html` for figures and case coverage.
Float32 and float64 use separate panels with shared axes. CPU and GPU curves
can be included in each panel. Failed and unmeasured sizes break curves and
remain visible in the coverage table, including their diagnostic messages.
Historical reports retain their original acceptance criteria.

The generic runner defaults to 32 and 128 observations, one query point, five warm repetitions,
one independent process, and 100 fitting steps. Use three processes for routine
measurements. Twenty processes are useful only when investigating cold-start
variability. Each case has a 600-second timeout. Failures are saved, subsequent
cases continue, and the command exits nonzero if any case failed.

Increase sizes deliberately. Dense inference has quadratic memory use and
cubic factorization cost. Use different size limits for different workloads.
`--backend gpu` selects an allocated GPU and rejects CPU fallback.

To measure prediction from an existing posterior, sweep query counts at a fixed
observation count:

```bash
uv run --frozen python benchmarks/run.py \
  --dtype float32 --workloads predict predict-marginals \
  --sizes 4096 --queries 1 16 64 256 1024 --process-samples 3 --repeats 20 \
  --output results/performance/prediction-float32.json --guard-cpu
```

Add `--plan` to print the cases without executing them. These workloads prepare
and synchronize a dense posterior before timing. The timed calls receive that
posterior and alternating query locations, so conditioning is excluded without
making the prediction outputs constant. `predict` returns the mean and full
query covariance. `predict-marginals` returns the mean and marginal variances.
The prediction figures use query count on the x-axis and separate figures for
each observation count. Other workloads retain observation count on the x-axis,
with separate figures when their query counts differ.

```bash
uv run --frozen --group comparison python benchmarks/run.py \
  --dtype float64 --workloads dense gpjax gpytorch \
  --sizes 256 1024 4096 --queries 1 --process-samples 3 --repeats 20 \
  --output results/performance/inference.json --resume --guard-cpu
```

`--resume` reuses accepted cases with matching measured sources, dependencies,
hardware and numerical settings. Scheduler IDs and output paths do not affect
reuse. GPU identity records CUDA-visible devices in their selected order,
including model, memory, compute resources, and driver release. Equivalent
allocations remain reusable; different models or partition sizes do not.
Each attempt retains its output and errors. Sources are checked before
and after every worker. `--guard-cpu` rejects interference during measurement,
waits for a quiet interval after competing work, and retries at most twice.
Already-idle machines start the next case immediately. Persistent interference
leaves a resumable report marked `paused`.

## What is timed

| Workload | Operation |
| --- | --- |
| `dense` | Condition on values and predict the full query covariance |
| `predict`, `predict-marginals` | Predict from an existing dense posterior, returning full covariance or marginal variances |
| `mixed` | Condition on values, slopes and local averages, then predict values |
| `append`, `rebuild` | Add one observation to a prepared posterior or rebuild, then predict |
| `mle`, `map` | A complete 100-step Adam fit, with explicit Gamma priors for MAP |
| `design`, `design-prepared` | Gradient-variance trace after four future observations and its coordinate gradient |
| `gpjax`, `gpytorch` | The same value-only inference problem in comparison libraries |
| `marginals`, `sample` | Condition, then return marginal prediction or eight joint posterior draws |
| `kernel-gradient` | Differentiate predicted means plus variance trace with respect to lengthscale |
| `feature-gradient` | Evaluate and differentiate 256 fixed-key Fourier features |
| `construction` | Construct a dataset and read its flat and named values |

`n` counts scalar observations. Mixed measurements split them between values,
one-dimensional derivatives and box averages. Design uses two-dimensional
locations and one gradient query. Other workloads use a one-dimensional RBF
GP with variance 1, lengthscale 0.8 and noise variance 0.05.

Each fresh worker selects its dtype before importing JAX or PyTorch. NumPy
seeds define the same design geometry in both precisions, rounded to the
selected dtype. A one-point value query is at 0.137. Larger query batches span
[-3.8, 3.8]. Two input variants, shifted by 0.037, alternate. Each fit starts
from the same initial parameters.

Before preparing a workload, each worker initializes its numerical runtime
with a fixed seven-row positive-definite problem: Cholesky factorization,
forward/backward triangular solves and matrix multiplication. JAX and PyTorch
use the same problem on the requested device and precision. Two synchronized
calls load the numerical libraries and exercise the runtime. This does not
condition a GP, compile a target workload or populate its posterior caches.

**First call means a new workload shape in an initialized runtime.** It includes
JAX tracing/lowering, compilation and first execution, recorded separately from
synchronized warm calls. Persistent compilation caching is disabled. The
separate `runtime_initialization_ms` records the numerical initialization pass,
including imports performed by that pass. It is not total process startup time.
It is excluded from first-call, warm and sequence timings.

Provenance and measurements record `timing_protocol=initialized-runtime-v1`.
The plotting reader will not pool this protocol with earlier process-first-call
measurements under the same label. Old results remain readable and unchanged.
Rerun cold and sequence measurements for the new protocol rather than subtracting
an estimated initialization cost from historical data.

Inherited XLA flags are recorded in provenance and included in resume identity.
Append/rebuild posterior preparation and conditioning-state preparation are
outside timing. Prepared design also excludes fixed-target preparation.
The two prediction-only workloads exclude conditioning from timing. Their
memory figures still include posterior preparation and storage, so they do not
represent incremental prediction memory alone.
Fits are reported as complete fits, never divided by their step counts.
Update figures use first-call time and show compilation separately. Other
figures use the median warm time from each process. Bands show the IQR across
process summaries, not a confidence interval.

CPU memory is peak process RSS. GPU memory is peak live allocation. They are
different metrics, labelled in the figure. No reference computations or extra
factorizations run after measurement. Compiler memory estimates are retained
when available.
`--trace PATH` captures a separate untimed warm JAX execution for Perfetto.
Tracing supports gpex workloads only; this is checked after selecting cases
from the command line or a manifest.

## Numerical acceptance

A successful run completes the requested repetitions on the requested backend,
returns finite numerical outputs, and produces valid timing samples. Every
repetition checks means, covariances, gradients, fitting losses and parameters
as applicable. Exceptions, allocation failures, timeouts and numerical warnings
remain failures, with their available outputs and timings retained.
All adapters store numerical results once in `outputs`, grouped by input variant.

Failed dense Cholesky factorizations can return NaNs, which the profiler rejects
through its finite-output checks.
Comparison-library numerical warnings also fail the run.
Recommendations about precision, performance warnings and deprecations are
recorded without rejecting a measurement.

Posterior variances and design variance traces must lie between zero and their
prior values, allowing eight machine epsilons of the prior variance scale for
subtraction roundoff. GPJax's predictive jitter is included in its upper bound.
This is a physical sanity check, not a comparison to a computed reference.
Values are retained without clipping. Finite gradients may be zero.
A fixed-step fit records its initial and final losses, but is not required
to improve or converge within the chosen step budget.

There are no cross-library or cross-version output comparisons, dense reference
runs, finite-difference checks, reference caches, or relative-accuracy thresholds.
These belong in correctness tests and dedicated accuracy studies. Numerical
acceptance does not claim a bound on prediction error or equivalent accuracy
between libraries. Plotting reports recorded results without changing success
or failure after execution.

Schema-5 and schema-6 reports use this policy. Schema 6 also records complete
sequences without treating their changing-size steps as warm repetitions.
Older reports retain their recorded outcomes.
Source and environment hashes prevent mixing incompatible measurements or
reusing results from a different implementation. They do not compare numerical
answers between versions.

## Choosing useful cases

Use a fixed, nonzero observation noise and explicit solver tolerances. The
standard workloads use noise variance 0.05 and jitter 1e-6. Keep those settings
fixed within a curve and report dtype and solver settings. Do not increase
jitter, switch solvers or loosen tolerances automatically to finish a run.

Choose observation limits by workload using established memory and runtime
measurements. Libraries may stop at different sizes. The campaign uses three
independent processes, 20 warm calls for inference and design, five complete
fits and two warm calls for updates. These measure specific operations rather
than convergence of an optimizer or statistical quality of a design.

## Solver and library settings

`--covariance-tile` controls covariance materialization, including fitting.
Use separate result labels for different numerical settings.

All libraries receive noise variance 0.05 plus fixed training jitter 1e-6.
GPJax additionally includes its 1e-6 predictive diagonal.
GPyTorch clears its prediction cache on every call and disables fast predictive
variance and forces Cholesky. Numerical warnings fail a case. Precision recommendations are recorded as advisories.

These measurements describe the selected kernels, operators, query sizes and
hardware. They do not establish performance for every GP problem.
The `profiling` branch contains the compact CPU/GPU campaign grid and thin
HELMA launcher. Its README covers preparation, submission and collection.
