# Examples

Start with the [tutorials](../docs/tutorials/index.md) for the core GP workflows.
These examples cover additional fitting strategies, compiled inference, and
an application loop:

| Example | Purpose |
| --- | --- |
| [d_scaled_lengthscale_prior.py](d_scaled_lengthscale_prior.py) | Fit an ARD model with a dimension-scaled lengthscale prior. |
| [map_multistart.py](map_multistart.py) | Fit MAP from multiple starting points and inspect convergence. |
| [map_random_subsets.py](map_random_subsets.py) | Approximate MAP with fresh observation subsets and a full-data reference objective. |
| [reusable_conditioning.py](reusable_conditioning.py) | Compare repeated compiled conditioning with incremental posterior updates, then replace the dataset. |
| [bayesian-optimization.ipynb](bayesian-optimization.ipynb) | Run a continuous Bayesian optimization loop with expected improvement. |

The dimension-scaled prior follows Hvarfner, Hellsten, and Nardi,
[Vanilla Bayesian Optimization Performs Great in High Dimensions](https://proceedings.mlr.press/v235/hvarfner24a.html)
(ICML 2024). The script demonstrates its lengthscale prior in a small regression
problem with normalized inputs and standardized outputs.

From a checkout, install dependencies with `uv sync --extra examples`, then run
a script in the recommended precision:

```bash
JAX_ENABLE_X64=1 uv run python examples/map_multistart.py
```

Open the BO notebook with `uv run jupyter lab examples/bayesian-optimization.ipynb`.
It enables x64 before constructing arrays.

For verification, install the development dependencies with `uv sync`.
The normal test suite runs the Python scripts in isolated temporary directories
in the selected precision. Run those checks with
`uv run pytest tests/test_examples.py`. The documentation workflow also executes
all tutorial and example notebooks. To check only the BO notebook, run:

```bash
uv run python scripts/verify_tutorials.py --write bayesian-optimization
```
