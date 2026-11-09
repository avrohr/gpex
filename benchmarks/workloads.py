"""Shared JAX timing. Read each workload script for the measured operation."""

from __future__ import annotations

import importlib
import time
import traceback
import warnings
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
from _runtime import initialize
from checks import serializable

MODULES = {
    "predict": "prediction",
    "predict-marginals": "prediction",
    "construction": "construction",
    "feature-gradient": "features",
    "marginals": "sampling",
    "sample": "sampling",
    "kernel-gradient": "gradients",
    "dense": "dense",
    "gpjax": "dense",
    "gpytorch": "dense",
    "mixed": "mixed",
    "append": "single_update",
    "rebuild": "single_update",
    "mle": "fitting",
    "map": "fitting",
    "design": "design",
    "design-prepared": "design",
}


def prepare(case):
    return importlib.import_module(MODULES[case["workload"]]).prepare(case)


def _finite(result):
    leaves = [np.asarray(x) for x in jax.tree.leaves(result) if eqx.is_array(x)]
    if not leaves or not all(np.isfinite(x).all() for x in leaves):
        raise ValueError("non-finite or empty numerical output")


def measure(case, trace_dir=None, *, phase=None):
    initialization = initialize(case)
    jax.config.update("jax_enable_compilation_cache", False)
    if jax.default_backend() != case["backend"]:
        raise ValueError(f"requested {case['backend']}, got {jax.default_backend()}")
    function, inputs, validate, evidence = prepare(case)
    jax.block_until_ready(inputs)
    if phase:
        phase("measure_start")
    results, samples, errors = {}, [], []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if case["workload"] == "construction":
                jax.block_until_ready(function(*inputs[0]))
                eager_samples = []
                for _ in range(case["repeats"]):
                    start = time.perf_counter()
                    jax.block_until_ready(function(*inputs[0]))
                    eager_samples.append(1000 * (time.perf_counter() - start))
                evidence["eager_samples_ms"] = eager_samples
            start = time.perf_counter()
            jit = jax.jit if case["workload"] == "gpjax" else eqx.filter_jit
            lowered = jit(function).lower(*inputs[0])
            lower_ms = 1000 * (time.perf_counter() - start)
            start = time.perf_counter()
            executable = lowered.compile()
            compile_ms = 1000 * (time.perf_counter() - start)
            for i in range(case["repeats"] + 1):
                args = inputs[i % len(inputs)]
                start = time.perf_counter()
                try:
                    result = executable(*args)
                    jax.block_until_ready(result)
                except Exception as error:
                    # A failed call is never eligible for timing, but earlier
                    # calls and compilation still provide useful diagnostic data.
                    samples.append(1000 * (time.perf_counter() - start))
                    errors.append(f"{type(error).__name__}: {error}")
                    evidence["execution_error"] = traceback.format_exc()
                    break
                samples.append(1000 * (time.perf_counter() - start))
                # Retain only two variants, rather than every covariance allocation.
                results[i % len(inputs)] = (result, args)
                try:
                    _finite(result)
                    if validate is not None:
                        validate(result, args)
                except ValueError as error:
                    errors.append(str(error))
                    break
            from run import memory

            measured_memory = memory()
    finally:
        if phase:
            phase("measure_end")
    evidence["warnings"] = [
        dict(category=w.category.__name__, message=str(w.message)) for w in caught
    ]
    errors.extend(
        str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)
    )
    try:
        compiled = getattr(executable, "compiled", executable)
        stats = compiled.memory_analysis()
        compiler_memory = {
            name: int(getattr(stats, name))
            for name in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "temp_size_in_bytes",
                "alias_size_in_bytes",
            )
        }
    except (AttributeError, TypeError, RuntimeError):
        compiler_memory = None
    # Save outputs for every workload, including objective and gradient failures.
    outputs = []
    for result, _ in results.values():
        numerical = (
            (result[0], result[1][0]) if case["workload"] in ("mle", "map") else result
        )
        outputs.append(
            [np.asarray(x) for x in jax.tree.leaves(numerical) if eqx.is_array(x)]
        )
    if trace_dir and not errors:
        try:
            Path(trace_dir).mkdir(parents=True, exist_ok=True)
            with (
                jax.profiler.trace(str(trace_dir), create_perfetto_trace=True),
                jax.profiler.TraceAnnotation(case["workload"]),
            ):
                jax.block_until_ready(executable(*inputs[0]))
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
            evidence["trace_error"] = traceback.format_exc()
    return serializable(
        dict(
            **initialization,
            lower_ms=lower_ms,
            compile_ms=compile_ms,
            first_execute_ms=samples[0],
            first_call_ms=lower_ms + compile_ms + samples[0],
            samples_ms=samples[1:],
            memory=measured_memory,
            compiler_memory=compiler_memory,
            validation=evidence,
            outputs=outputs,
            device=str(jax.devices()[0]),
            validated=not errors,
            **({"error": " / ".join(errors)} if errors else {}),
        )
    )
