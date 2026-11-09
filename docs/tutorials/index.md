# Tutorials

Each tutorial is a complete, runnable notebook. The website includes the
outputs from an executed run, including plots and printed results.

Start with **First posterior**, then choose the workflow you need:

| Tutorial | Task |
| --- | --- |
| [First posterior](first-posterior.ipynb) | Condition on a small dataset and plot a prediction. |
| [Learning a function from derivatives](mixed-measurements.ipynb) | Combine slopes and curvature, then anchor the function with one value. |
| [Learning from averages](nonlocal-measurements.ipynb) | Infer function values and accumulated integrals from window averages. |
| [Experimental design](design-of-experiments.ipynb) | Optimize a batch of observations to reduce uncertainty about a target. |
| [Sequential observations](sequential-streaming.ipynb) | Reuse compiled inference and update a posterior as observations arrive. |
| [Fitting hyperparameters](mle-map.ipynb) | Fit kernel and observation-noise parameters with MLE and MAP. |
| [Sampling from a GP posterior](posterior-paths.ipynb) | Draw values at fixed locations, or sample a function to evaluate and differentiate. |
| [Implementing a custom operator](custom-operators.ipynb) | Infer a scalar function from vector readings using a custom operator. |

The [Concepts](../concepts/linear-functionals.md) pages explain gpex's model
and terminology. The [API reference](../api/inference.md) covers individual
methods and their options. Additional configurations, including Bayesian
optimization and advanced fitting, are available as
[executable examples](https://github.com/avrohr/gpex/tree/main/examples).

To run the notebooks from a checkout:

```bash
uv sync --group dev --extra examples
uv run jupyter lab docs/tutorials/first-posterior.ipynb
```

Use **Restart Kernel and Run All Cells** to reproduce a notebook. Each runs
independently and includes its own data. Blue denotes the main posterior,
orange denotes observations or initial proposals, and purple denotes a
comparison or path. Shaded bands show pointwise 95% latent intervals.
