"""Optional comparison models and eager timing, shared by workload scripts."""

from __future__ import annotations

import time
import traceback
import warnings
from contextlib import ExitStack
from typing import cast

import numpy as np
from _runtime import initialize
from checks import serializable


def gpjax_model():
    import gpjax as gpx
    import jax.numpy as jnp

    prior = gpx.gps.Prior(
        mean_function=gpx.mean_functions.Zero(),
        kernel=gpx.kernels.RBF(
            variance=gpx.parameters.NonNegativeReal(1.0),
            lengthscale=gpx.parameters.PositiveReal(0.8),
        ),
    )
    return prior * gpx.likelihoods.Gaussian(obs_stddev=jnp.sqrt(jnp.asarray(0.05)))


def torch_model(case, x, y):
    import gpytorch
    import torch

    dtype = torch.float64 if case["dtype"] == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device("cuda" if case["backend"] == "gpu" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA-enabled PyTorch is required")

    def array(value):
        return torch.as_tensor(value, dtype=dtype, device=device)

    class Model(gpytorch.models.ExactGP):
        def __init__(self, x, y, likelihood):
            super().__init__(x, y, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                cast(torch.Tensor, self.mean_module(x)), self.covar_module(x)
            )

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = Model(array(x), array(y), likelihood).to(device)
    # Match the fixed training jitter in the other libraries' numerical systems.
    likelihood.noise = array(0.05 + 1e-6)
    model.covar_module.outputscale = 1.0
    model.covar_module.base_kernel.lengthscale = array(0.8)
    model.eval()
    likelihood.eval()
    return model, array


def torch_settings(case):
    import gpytorch
    import torch

    context = ExitStack()
    context.enter_context(torch.inference_mode())
    context.enter_context(gpytorch.settings.fast_pred_var(False))
    context.enter_context(gpytorch.settings.max_cholesky_size(10**9))
    return context


def torch_sync(result=None, *, backend):
    import torch

    if backend == "gpu":
        torch.cuda.synchronize()


def record_warnings(caught, evidence, errors, advisory_types=()):
    for warning in caught:
        message = str(warning.message).strip()
        evidence.append(dict(category=warning.category.__name__, message=message))
        if issubclass(warning.category, RuntimeWarning) and not issubclass(
            warning.category, advisory_types
        ):
            errors.append(message)


def measure(case, trace_dir=None, *, phase=None):
    import torch
    from dense import prepare
    from linear_operator.utils.warnings import PerformanceWarning
    from run import memory

    initialization = initialize(case)
    operation, inputs, validate, evidence = prepare(case)
    torch_sync(backend=case["backend"])
    if case["backend"] == "gpu":
        torch.cuda.reset_peak_memory_stats()
    samples, outputs, errors = [], {}, []
    evidence["warnings"] = []
    if phase:
        phase("measure_start")
    try:
        with torch_settings(case):
            for i in range(case["repeats"] + 1):
                args = inputs[i % len(inputs)]
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    start = time.perf_counter()
                    try:
                        result = operation(*args)
                        torch_sync(result, backend=case["backend"])
                    except Exception as error:
                        errors.append(str(error))
                        evidence["execution_error"] = traceback.format_exc()
                        break
                    samples.append(1000 * (time.perf_counter() - start))
                record_warnings(
                    caught, evidence["warnings"], errors, (PerformanceWarning,)
                )
                values = tuple(value.detach().cpu().numpy() for value in result)
                outputs[i % len(inputs)] = values
                if not all(np.isfinite(value).all() for value in values):
                    errors.append("non-finite posterior prediction")
                try:
                    validate(values, args)
                except ValueError as error:
                    errors.append(str(error))
                if errors:
                    break
        measured_memory = memory()
    finally:
        if phase:
            phase("measure_end")
    return serializable(
        dict(
            **initialization,
            first_call_ms=samples[0] if samples else 0,
            samples_ms=samples[1:],
            memory=measured_memory,
            outputs=list(outputs.values()),
            validation=evidence,
            device=torch.cuda.get_device_name() if case["backend"] == "gpu" else "CPU",
            validated=not errors,
            **({"error": " / ".join(errors)} if errors else {}),
        )
    )
