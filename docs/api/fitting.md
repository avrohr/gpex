# Fitting and objectives

See [Fitting hyperparameters](../tutorials/mle-map.ipynb) for an end-to-end MLE
and MAP example. Fitting functions optimize explicit `Parameter` leaves in a
model PyTree. To fit both GP and observation-noise parameters, include both
objects, for example `(gp, dataset)`, in the fitted model.

## Optimizers and results

| Function | Use |
| --- | --- |
| `fit_lbfgs(model, objective, ...)` | Deterministic objective with line search and gradient convergence |
| `fit(model, objective, optimizer, ...)` | A fixed number of updates using a supplied Optax optimizer |
| `fit_multistart(starts, fit_one)` | Independent fits from several initial parameter settings |

Single fits return the final `model`, `loss`, `gradient_norm`, and update and
evaluation counts. `fit_lbfgs` sets `converged=True` when it meets its gradient
tolerance. Reaching its iteration limit returns `converged=False`. Fixed-step
`fit` always reports `converged=False`.

`losses` includes the initial loss. L-BFGS allocates `max_steps + 1` entries
and fills unused entries with NaN. Outside JIT, use
`result.losses[:int(result.step_count) + 1]` for the completed history.
A failed line search raises, as does an accepted iterate with non-finite
parameters, objective, or gradients.

Multistart inputs may differ in trainable parameter values. Their shapes,
transforms, priors, names, and fixed data must agree. `result.model` selects
the smallest final objective, and plural diagnostic fields report each start
in input order. Any failed run raises. See the
[multistart example](https://github.com/avrohr/gpex/blob/main/examples/map_multistart.py).

The fitting loops are compiled. Changing optimizer configuration, model
structure, shapes, or iteration budgets may recompile them. `fit` supports
reverse differentiation through its updates. `fit_lbfgs` differentiates the
objective but does not support reverse differentiation through the entire fit.
Call multistart helpers outside an outer JIT because best-start selection
returns a Python index. See [JAX and observation capacity](../concepts/jax-and-capacity.md)
for data shape reuse.

::: gpex.fitting
    options:
      members:
        - FitResult
        - MultiStartFitResult
        - fit
        - fit_lbfgs
        - fit_multistart

## Subset objectives

`fit_random_subsets` draws a fresh subset without replacement from each
observation block at every Optax update. An integer `subset_size` applies to
every block, or a mapping gives counts by block name. All output components
at a selected location stay together, along with the block's shared noise.
Each selected count must be between one and the active location count.

A GP marginal likelihood couples observations through its covariance. A
subset objective is therefore an approximation, even after likelihood scaling.
For MAP, scale only the likelihood and include the prior once. Use a full-data
`reference_objective` to evaluate the final model. The
[subset fitting example](https://github.com/avrohr/gpex/blob/main/examples/map_random_subsets.py)
shows the complete objective and its scaling.

`subset_losses` records each update's proxy objective. Values from different
subsets are not directly comparable. `reference_loss` is evaluated once at
the final model. Multistart subset fits select by their final full-data
`reference_losses`. Use a first-order Optax optimizer for the changing subset
objective.

::: gpex.fitting
    options:
      members:
        - RandomSubsetFitResult
        - MultiStartRandomSubsetFitResult
        - fit_random_subsets
        - fit_multistart_random_subsets

## Exact objectives

::: gpex.fitting.objectives
    options:
      members:
        - negative_log_marginal_likelihood
        - negative_log_posterior
