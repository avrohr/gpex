# CPU and H100 profiling

This branch adds launch scripts and compact release measurements to main.
Scientific workloads and numerical checks live in `benchmarks/`.

- `plan.json` defines the workload sizes and solver settings.
- `campaign.py` freezes committed source and writes workload manifests. It does
  not install environments, launch processes, connect to hosts or monitor jobs.
- `local/` owns CPU installation, launching and cleanup. `cpu.sh` runs the two
  precisions sequentially, with the benchmark runner's contention guard.
- `../cluster/` owns installation, Slurm submission, execution and cleanup on HELMA.
- `summarize.py` retains timing distributions, memory and provenance in
  `results/<revision>.json`. `plot.py` renders figures from that summary alone.

## Run on HELMA

Log in to HELMA and run these commands from the repository root:

```bash
git checkout profiling
git pull --ff-only
bash cluster/install.sh
bash cluster/submit.sh
```

`uv` must be on `PATH` or in `~/.local/bin`. Installation uses Python 3.12 and the
current committed checkout's locked CUDA13 and comparison dependencies. The
current lock uses JAX 0.11.2. Installation creates no campaign or jobs.

The account-local root defaults to `~/gpex-profiling`, configured by `helma.root`
in the plan. Environments live in `runtime/<short-commit>`, the dedicated package
cache in `cache/`, and campaigns in `runs/<run-name>` under that root. No account
name or connection details are stored in these scripts.

Submission checks the installed lockfiles and source receipt, freezes committed
`HEAD`, then submits eight jobs. Each precision has one job for fitting,
conditioning, design and prediction, and sequences. The plan lists the
workloads and Slurm time limit for each group. Workloads run serially on the
assigned H100. Each measured case still gets a fresh process.

The worker copies and checksums the frozen source, locked environment, and
Python interpreter once into Slurm's node-local scratch. `logs/<job-id>/staging.tsv`
records archive sizes, hashes and staging times. Staging happens before any
measurement. Each process uses fresh local scratch and a private CUDA cache.
The fixed runtime initializer runs before workload measurement. Target lowering,
compilation, and first execution retain their existing timers.

The worker publishes one report and its case evidence per workload. It attempts
to save the active workload's partial results when Slurm warns that time is
running out or requests termination. Independent workloads continue after an
ordinary failure, and the job reports a nonzero exit status. `launch.tsv`
records the selected workloads and job IDs.

Use `--dry-run` to list the jobs without creating a campaign. Give a new
campaign a run name, or use `--resume` with the existing name after all its jobs
have stopped. Resume skips validated complete workload reports and reuses valid
case components with matching provenance. To limit the selection:

```bash
bash cluster/submit.sh my-run --dry-run --dtype float64 --workload gpjax
bash cluster/submit.sh my-run --resume --dtype float64 --workload gpjax
```

## Run CPU workloads locally

From the local checkout root:

```bash
bash profiling/local/install.sh
bash profiling/local/run.sh
```

Installation creates the locked comparison environment in
`profiling/.local/runtime/<short-commit>` and a cache in `profiling/.local/cache`.
Run prepares `profiling/raw/<short-commit>` and starts all 525 CPU cases
sequentially, recording the controller in `cpu.pid` and its output in `cpu.log`.
It refuses to overlap another CPU profiling controller or worker in this checkout.
On macOS, a one-shot launchd job uses explicit `Interactive` resource policy.
It survives terminal closure and never automatically restarts on success or
failure. `cpu-launch.json` records the policy, inherited QoS, process ID and
launcher/worker hashes. Resume through the same entrypoint to preserve that
policy, rather than using `launchctl submit` or another ad hoc launcher.
Other platforms use the detached shell controller.


Use `bash profiling/local/run.sh my-run` for a fresh named campaign. After resolving
contention, use `bash profiling/local/run.sh my-run --resume` from the same committed
checkout to reuse validated successful cases. The runner retains rejected attempts.

Both workflows require committed source, dependencies, plan and workers. Later
checkout edits do not affect frozen runs. A new commit needs its matching environment
installed before launch. An existing runtime can be reused without reinstalling.
Do not edit source snapshots or environments while workers use them.

## Summarize and plot

Copy completed GPU results to the local CPU campaign directory using your usual
file-transfer tool. Keep the `gpu/` directory and the `gpu-*.json` manifests from
the HELMA run. Both runs must use the same committed source and plan. Verify the
transferred files against checksums from the source machine.

```bash
uv run --frozen --group dev python profiling/summarize.py profiling/raw/REVISION --output profiling/results/REVISION.json --profiling-commit "$(git rev-parse HEAD)"
uv run --frozen --group dev python profiling/plot.py profiling/results/REVISION.json --output-dir /tmp/profiling-figures
```

Summarization requires complete validated coverage and retains report/source
hashes and package/hardware provenance. Fitting is labelled as complete 100-step
fit time divided by 100. Prediction query sweeps are shown separately.
Schema 2 summaries keep complete-experiment workloads in `sequences`, separate
from ordinary warm-runtime `points`. Each sequence retains its settings, observation
counts, initial/total/per-step/cumulative timing medians, quartiles and extrema across
processes, individual process totals and trace counts, peak memory and validation.
Compilation is included in these sequence timings, not estimated separately.
Ordinary points retain `process_ranges` for first-call time, lowering, compilation,
combined setup, warm runtime and memory. Warm ranges use one median per process,
not pooled execution samples. Setup ranges sum lowering and compilation within
each process before computing the extrema. Documentation plots show light min–max
bands across independent processes. Timing lines remain medians and memory lines
remain maximum process peaks. These bands are not confidence intervals. Older
summaries without extrema are plotted without bands.
Different dataset sizes, lengths and starting sizes are never pooled.
`plot.py` renders ordinary points and complete dataset/sequential workflows
from the compact summary. The documentation measurements are in
`results/4e5cada.json`: CPU from profiling revision `683bade` and staged GPU
from `4e5cada`, both measuring library source `5ffd0bf`. The summary records
source hashes and profiling revisions per backend. Scientific workloads, locked
dependencies and timing protocol match. This composite was audited separately
because the generic summarizer requires identical runner source across devices.
`results/6a14d63.json` and `results/2f95894.json` retain comparison baselines.
Monitoring and result transfer are deliberately outside the campaign scripts.

## Cleanup

After validating the compact summary, remove a selected CPU campaign and all
owned CPU environments/cache:

```bash
bash profiling/local/cleanup.sh
# Or select a named campaign:
bash profiling/local/cleanup.sh my-run
```

On HELMA, remove all owned campaigns, environments and the dedicated cache:

```bash
bash cluster/cleanup.sh
```

Cleanup refuses active profiling processes/jobs and requires the environment's
ownership marker. It preserves tracked summaries and does not touch shared user
caches or unrelated projects. Raw arrays, compiler traces and attempt logs need
not be retained after the summary is validated.

## Current matrix

The plan contains 1,161 fresh-process cases: CPU 267 float32 and 258 float64,
H100 318 per precision. Single-operation settings have three processes, with 20 synchronized
warm calls, five 100-step fits, or two posterior updates. Prediction sweeps use
4,096 observations and 1, 16, 64, 256 or 1,024 queries.

The dataset sequences condition on 100 independent datasets spanning 16–512
observations, each with a distinct size. The sequential workloads
grow from 512 to 1,024 observations. Each complete sequence has three process
samples per method and precision, including its first compilation. Sequence
settings match the workload scripts in `benchmarks/`.

Settings, numerical checks and workload limits are in the plan and benchmark
source. Every worker disables the experimental fusion autotuner
while preserving other autotuning. Compilation, warm execution and memory are
recorded separately. Each H100 job requests one GPU and 32 CPU cores.
