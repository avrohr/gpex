"""Time one complete sequence without clearing compilation or posterior caches."""

from __future__ import annotations

import time
import traceback
import warnings
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
from _runtime import initialize
from checks import check_variance, serializable


@dataclass
class Sequence:
    """Prepared inputs and explicit initial/update operations for one process."""

    step: Callable
    inputs: list[tuple]
    counts: list[int]
    framework: str
    initial: Callable | None = None
    initial_args: tuple = ()
    prior_variance: float = 1.0


def measure(case, experiment: Sequence, *, phase=None):
    from run import memory

    initialization = initialize(case)
    sync: Callable[..., Any]
    traces = []
    errors, outputs, elapsed, warning_evidence = [], [], [], []
    evidence: dict[str, Any] = dict(finite_outputs=True)
    if experiment.framework == "torch":
        import torch
        from compare import record_warnings, torch_settings, torch_sync
        from linear_operator.utils.warnings import PerformanceWarning

        context = torch_settings(case)

        def torch_ready(result=None):
            torch_sync(result, backend=case["backend"])

        sync = torch_ready
        initial, step = experiment.initial, experiment.step

        def torch_arrays(result):
            return [x.detach().cpu().numpy() for x in result]

        arrays = torch_arrays
        sync()
        if case["backend"] == "gpu":
            torch.cuda.reset_peak_memory_stats()
        device = torch.cuda.get_device_name() if case["backend"] == "gpu" else "CPU"
    else:
        import equinox as eqx
        import jax

        jax.config.update("jax_enable_compilation_cache", False)
        if jax.default_backend() != case["backend"]:
            raise ValueError("JAX backend mismatch")
        jit = jax.jit if experiment.framework == "gpjax" else eqx.filter_jit

        def compile_call(function, label):
            def call(*args):
                traces.append(label)
                return function(*args)

            return jit(call)

        step = compile_call(experiment.step, "step")
        initial = (
            compile_call(experiment.initial, "initial") if experiment.initial else None
        )
        sync = jax.block_until_ready

        def jax_arrays(result):
            return [np.asarray(x) for x in result]

        arrays = jax_arrays

        context = nullcontext()
        sync((experiment.initial_args, experiment.inputs))
        device = str(jax.devices()[0])

    def check(result):
        values = arrays(result)
        outputs.append(values)
        if not all(np.isfinite(x).all() for x in values):
            evidence["finite_outputs"] = False
            raise ValueError("non-finite posterior prediction")
        check_variance(
            np.diag(values[1]), case, evidence, prior_variance=experiment.prior_variance
        )

    initial_ms = 0.0
    state = None
    if phase:
        phase("measure_start")
    caught = []
    try:
        with context, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if initial:
                start = time.perf_counter()
                state, result = initial(*experiment.initial_args)
                sync((state, result))
                initial_ms = 1000 * (time.perf_counter() - start)
                check(result)
            for args in experiment.inputs:
                start = time.perf_counter()
                state, result = step(state, *args)
                sync((state, result))
                elapsed.append(1000 * (time.perf_counter() - start))
                check(result)
    except Exception as error:
        errors.append(str(error))
        evidence["execution_error"] = traceback.format_exc()
    finally:
        if experiment.framework == "torch":
            record_warnings(caught, warning_evidence, errors, (PerformanceWarning,))
        else:
            for warning in caught:
                warning_evidence.append(
                    dict(
                        category=warning.category.__name__,
                        message=str(warning.message),
                    )
                )
                if issubclass(warning.category, RuntimeWarning):
                    errors.append(str(warning.message))
        measured_memory = memory()
        if phase:
            phase("measure_end")
    evidence["warnings"] = warning_evidence
    evidence["checked_predictions"] = len(outputs)
    return serializable(
        dict(
            **initialization,
            measurement="sequence",
            initial_ms=initial_ms,
            step_ms=elapsed,
            observation_counts=experiment.counts[: len(elapsed)],
            cumulative_ms=(initial_ms + np.cumsum(elapsed)).tolist(),
            total_ms=initial_ms + sum(elapsed),
            trace_count=len(traces) if experiment.framework != "torch" else None,
            memory=measured_memory,
            outputs=outputs,
            validation=evidence,
            device=device,
            validated=not errors and len(elapsed) == len(experiment.inputs),
            **({"error": " / ".join(errors)} if errors else {}),
        )
    )
