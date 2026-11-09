# Contributing

gpex is an early-stage research library. Small changes with an explicit
mathematical meaning, a focused test, and a readable public API are preferred.

## Local setup

The locked environment installs the published cholops 0.2.1 release. Its
`PaddedCholesky` provides masked solves and whitening, reserved-slot activation,
and coefficient reuse for streaming updates.

The project uses [uv](https://docs.astral.sh/uv/) for Python, dependencies, and
commands:

```bash
uv sync --locked --all-groups
uv run --frozen ruff check .
uv run --frozen pyright
JAX_ENABLE_X64=0 uv run --frozen pytest --cov=gpex --cov-report=term-missing
JAX_ENABLE_X64=1 uv run --frozen pytest --cov=gpex --cov-report=term-missing
uv build
uv run --frozen twine check dist/*
```

Numerical tests and CI cover float32 and float64 in separate processes. Select
precision before test collection with `JAX_ENABLE_X64`; the tests and example
subprocesses respect that setting. Float64 remains recommended. Mixed dtypes
are unsupported. Test tolerances and range stress cases may depend on precision;
library algorithms must remain dtype-agnostic. Preserve tight float64 checks
and justify float32 accuracy against independent numerical references.

Routine CI runs the full suite on Python 3.13 in both precisions and on Python
3.12 in float64. This covers the oldest and newest advertised Python versions
without repeating the full precision matrix on every interpreter.

Test and documentation jobs install the default development group; only the
benchmark smoke job needs the comparison group and its separate dependency cache.
Superseded pull-request runs are cancelled. Pytest prints the slowest 30 test
phases in each CI job; collect all per-case timings locally with
`--junitxml=/tmp/gpex-tests.xml`. Measure a
complete suite in a fresh process, since JAX compilation reuse makes isolated
test timings differ from timings in suite order.

Keep integration fixtures focused on the behavior under test. Cover operator
combinations, tiling, and right-hand-side layouts in their dedicated tests;
integration tests should retain representative combinations and distinct solver
paths rather than repeat every combination at each layer. Use small eager
fixtures for transformation and dispatch behavior, and compiled tests for broad
mixed-operator comparisons. Compile expensive independent numerical references
as well as the operation under test.

Test observable behavior, numerical accuracy, and meaningful resource guarantees.
Do not pin private field names, object identity, PyTree leaf counts, or exact
helper call sequences when equivalent implementations would be acceptable.
Keep checks that prevent dense allocation, unnecessary solver work, or silently
lost numerical failures. Remove weaker overlapping tests when a stronger test
already covers their behavior; test count and coverage percentage are not goals
in themselves.

Verify the tutorial and example notebooks and their links, or open one notebook:

```bash
uv run --frozen python scripts/verify_tutorials.py --write
uv run --frozen mkdocs build --strict
uv run --frozen python scripts/check_docs_links.py
uv run --frozen python scripts/check_tutorial_outputs.py
uv run --frozen jupyter lab docs/tutorials/first-posterior.ipynb
```

The verifier discovers notebooks in both `docs/tutorials/` and `examples/`.
Pass notebook names without their `.ipynb` extension to run selected files,
for example `python scripts/verify_tutorials.py --write bayesian-optimization`.

Commit the executed notebooks when changing their content. `--write` saves
plots, text output, and execution counts from fresh kernels. For tutorial
notebooks, the output checker compares every saved output with the built page
and checks the downloadable notebook. Plotting setup may be collapsed, but its
outputs stay visible.

Regenerate the overview figure with
`uv run python scripts/generate_intro_poster.py`.

When changing dependencies, regenerate `uv.lock` with `uv lock` and include it
with the `pyproject.toml` changes. Run `uv lock --check` before installing with
`--locked`; `--frozen` alone does not check whether the declarations changed.
For comparison-library upgrades, run the benchmark smoke command in
[.github/workflows/ci.yml](.github/workflows/ci.yml) with the new installed
versions. [benchmarks/README.md](benchmarks/README.md) describes the protocol.

## Documentation deployment

The Docs workflow executes and saves the tutorials, builds MkDocs in strict
mode, and checks links and rendered notebook outputs. It passes those same
executed notebooks to the deployment job. Public deployments use mike to
retain versions:

- <https://avrohr.github.io/gpex/> opens the latest stable release. Before the
  first release, it opens the development documentation.
- `/gpex/dev/` follows successful pushes to `main` and displays an unreleased
  development banner.
- `/gpex/0.1.0/`, for example, is built from the exact `v0.1.0` release tag.
  Each published, non-prerelease `v*` GitHub release gets its own version.
- `/gpex/stable/` follows the highest published stable version. Publishing an
  older maintenance release preserves the newer default.

The header selector switches versions. Built pages and version metadata are
preserved on `gh-pages`. Deployment jobs serialize updates, check the combined
site, push that branch, and deploy its contents through GitHub Actions.
Pull requests and private repositories only run the documentation checks.

After making the repository public, enable **Settings → Pages → Build and
deployment → Source: GitHub Actions**. Then run the **Docs** workflow manually
with **Run workflow** on `main`, or push a commit to `main`. Deployment uses
the `github-pages` environment and the built-in `GITHUB_TOKEN`, including
`contents: write` permission to preserve documentation history. Keep the Pages
source set to GitHub Actions. If the environment restricts deployment branches
and tags, allow the `main` branch and `v*` tags. Re-run a release's Docs workflow
run to rebuild that version from its original tag checkout.

## Design expectations

The architectural rationale and current modeling boundaries are recorded in
[`DESIGN.md`](DESIGN.md). [AGENTS.md](AGENTS.md) records coding preferences,
module ownership, and guidance for keeping tests and documentation useful.

- Keep the general block API as the semantic core.
- Make parameter constraints and priors explicit at their definitions.
- Preserve JAX transformations: avoid hidden mutation and Python loops inside
  hot compiled paths.
- State whether an approximation is exact, quadrature-based, or statistical.
- Add a numerical covariance or inference test for every new operator.

Substantial API changes should update `DESIGN.md` and the relevant guide in
the same pull request.

## Releases

1. Summarize user-visible changes since the previous release in `CHANGELOG.md`,
   consolidating any `Unreleased` draft into a dated version section.
2. Update the version in `pyproject.toml` and `CITATION.cff`.
3. Run the full local checks above.
4. Push a tag such as `v0.1.0` and publish the matching GitHub release.

The release workflow builds the wheel and source archive and publishes them
through PyPI Trusted Publishing. Configure a PyPI publisher for the GitHub
repository, workflow `release.yml`, and environment `pypi` before the first
release.
