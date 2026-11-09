"""Result integrity tests; no hardware timing thresholds."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))

RUNNER_PATH = Path(__file__).resolve().parents[1] / "benchmarks/run.py"
SPEC = importlib.util.spec_from_file_location("benchmark_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def workloads(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNER_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "benchmark_workloads",
        Path(__file__).resolve().parents[1] / "benchmarks/workloads.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case():
    return {"workload": "dense", "n": 16, "repeats": 2}


def row():
    return {
        "schema_version": 3,
        "status": "complete",
        "case": case(),
        "fingerprint": runner.digest({}),
        "validated": True,
        "first_call_ms": 1.0,
        "samples_ms": [0.1, 0.2],
    }


def report():
    return {
        "schema_version": 3,
        "synthetic": False,
        "status": "complete",
        "provenance": {},
        "fingerprint": runner.digest({}),
        "cases": [case()],
        "rows": [row()],
    }


def test_failed_attempt_cannot_relabel_old_result(tmp_path, monkeypatch):
    component = tmp_path / "component.json"
    runner.write(component, row())

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="worker failed")

    monkeypatch.setattr(runner.subprocess, "run", fail)
    result = runner.execute(case(), "new-hash", component, 5)
    assert result["status"] == "failed"
    assert result["stderr"] == "worker failed"
    assert "samples_ms" not in result
    assert json.loads(component.read_text()) == result


def test_timeout_is_recorded(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1, stderr=b"timed out")

    monkeypatch.setattr(runner.subprocess, "run", timeout)
    result = runner.execute(case(), "hash", tmp_path / "out.json", 1)
    assert result["status"] == "failed"
    assert result["stderr"] == "timed out"


def test_failed_worker_preserves_outputs_and_validation_evidence(tmp_path, monkeypatch):
    failure = {
        "schema_version": 3,
        "status": "failed",
        "case": case(),
        "fingerprint": "hash",
        "validated": False,
        "error": "negative posterior variance",
        "outputs": [[[0.1], [[-2.9e-7]]]],
        "validation": {"minimum_variance": -2.9e-7},
    }

    def failed_validation(command, **kwargs):
        runner.write(command[-1], failure)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", failed_validation)
    component = tmp_path / "failure.json"
    result = runner.execute(case(), "hash", component, 5)
    assert result["status"] == "failed"
    assert result["outputs"] == failure["outputs"]
    assert result["validation"] == failure["validation"]
    assert not runner.valid(result, case(), "hash")
    assert json.loads(component.read_text()) == result


def test_prediction_validation_retains_failed_outputs(workloads, monkeypatch):
    import jax.numpy as jnp

    prepare = workloads.prepare

    def prepare_invalid(case):
        operation, inputs, validate, evidence = prepare(case)

        def predict(*args):
            mean, covariance = operation(*args)
            return mean, jnp.full_like(covariance, -2.9e-7)

        return predict, inputs, validate, evidence

    monkeypatch.setattr(workloads, "prepare", prepare_invalid)
    monkeypatch.setitem(sys.modules, "run", runner)
    result = workloads.measure(
        dict(
            workload="dense",
            n=8,
            queries=1,
            covariance_tile=None,
            backend="cpu",
            repeats=1,
        )
    )
    assert not result["validated"]
    assert result["first_call_ms"] > 0 and result["memory"]
    evidence = result["validation"]
    expected = float(jnp.asarray(-2.9e-7))
    assert result["outputs"][0][1] == [[expected]]
    assert "predictions" not in result
    assert evidence["minimum_variance"] == expected


def test_worker_writes_failed_validation_evidence(tmp_path, monkeypatch, workloads):
    monkeypatch.setitem(sys.modules, "workloads", workloads)
    evidence = {"minimum_variance": -2.9e-7}
    outputs = [[[0.1], [[-2.9e-7]]]]

    def fail(case, trace_dir, **kwargs):
        return dict(
            validated=False,
            error="negative posterior variance",
            outputs=outputs,
            validation=evidence,
            samples_ms=[1.0],
            memory={"process_peak_rss_bytes": 1024},
        )

    monkeypatch.setattr(workloads, "measure", fail)
    monkeypatch.setattr(runner, "provenance", lambda: {})
    request, output = tmp_path / "request.json", tmp_path / "result.json"
    runner.write(
        request,
        dict(
            case=case(),
            fingerprint=runner.digest(runner.measurement_identity({})),
            trace_dir=None,
        ),
    )
    runner.worker(request, output)
    result = json.loads(output.read_text())
    assert result["status"] == "failed"
    assert result["validated"] is False
    assert result["outputs"] == outputs
    assert result["validation"] == evidence


@pytest.mark.parametrize(
    "change",
    [
        {"status": "failed"},
        {"fingerprint": "other"},
        {"validated": False},
        {"samples_ms": [float("nan"), 0.1]},
        {"samples_ms": [0.1]},
        {"samples_ms": None},
        {"samples_ms": [True, False]},
        {"first_call_ms": -1},
        {"case": {"n": 100}},
    ],
)
def test_resume_rejects_invalid_components(change):
    assert not runner.valid(row() | change, case(), runner.digest({}))


def test_valid_report_and_component():
    assert runner.valid(row(), case(), runner.digest({}))
    runner.check_report(report())


def test_resume_rejects_nonobject_json():
    assert not runner.valid([], case(), runner.digest({}))


@pytest.mark.parametrize(
    "change",
    [
        {"synthetic": True},
        {"cases": [], "rows": []},
        {"rows": []},
        {"cases": [case(), case()], "rows": [row(), row()]},
        {"fingerprint": "altered"},
    ],
)
def test_rejects_invalid_publication_reports(change):
    with pytest.raises(ValueError):
        runner.check_report(report() | change)


def test_source_manifest_covers_measurement_packages(monkeypatch):
    monkeypatch.setattr(runner, "hardware", lambda: {})
    source = runner.provenance()["sources"]
    assert {
        "src/gpex/linalg/base.py",
        "src/gpex/linalg/dense.py",
        "src/gpex/gp.py",
        "src/gpex/feature_paths.py",
        "src/gpex/prepared_query.py",
        "src/gpex/kernels/features.py",
        "src/gpex/covariance_systems.py",
        "src/gpex/results.py",
        "src/gpex/moments/__init__.py",
        "src/gpex/moments/assembly.py",
        "src/gpex/moments/measurements.py",
        "src/gpex/moments/operator_covariance.py",
        "src/gpex/operators/base.py",
        "src/gpex/operators/local.py",
        "src/gpex/operators/differences.py",
        "src/gpex/operators/quadrature.py",
        "src/gpex/blocks/__init__.py",
        "src/gpex/blocks/functionals.py",
        "src/gpex/blocks/queries.py",
        "src/gpex/blocks/designs.py",
        "src/gpex/blocks/datasets.py",
        "src/gpex/blocks/indexing.py",
    } <= source.keys()
    assert "src/gpex/operators.py" not in source


@pytest.mark.parametrize("workload", ["mle", "map"])
@pytest.mark.parametrize("tile_size", [None, 3])
def test_fitting_workloads_preserve_covariance_tile(workload, tile_size, workloads):
    operation, inputs, validate, _ = workloads.prepare(
        {
            "workload": workload,
            "n": 8,
            "queries": 1,
            "steps": 1,
            "covariance_tile": tile_size,
        }
    )
    (model,) = inputs[0]
    assert model[0].covariance_tile_size == tile_size
    result = operation(model)
    assert result[1][0].covariance_tile_size == tile_size
    validate(result, inputs[0])


def test_runtime_failure_keeps_earlier_measurements(workloads, monkeypatch):
    import jax
    import jax.numpy as jnp

    def fail_second(value):
        if float(value) > 1:
            raise ValueError("injected numerical failure")

    def operation(value):
        jax.debug.callback(fail_second, value)
        return jnp.array([value]), jnp.ones((1, 1))

    monkeypatch.setattr(
        workloads,
        "prepare",
        lambda *args: (
            operation,
            ((jnp.array(1.0),), (jnp.array(2.0),)),
            lambda *args: None,
            {},
        ),
    )
    monkeypatch.setitem(sys.modules, "run", runner)
    result = workloads.measure(dict(workload="dense", backend="cpu", repeats=2))
    assert result["validated"] is False
    assert "injected numerical failure" in result["error"]
    assert result["outputs"] == [[[1.0], [[1.0]]]]
    assert "predictions" not in result
    assert result["first_call_ms"] > 0
    assert result["compile_ms"] > 0
    assert result["memory"]["process_peak_rss_bytes"] > 0
    assert "injected numerical failure" in result["validation"]["execution_error"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_gradient_rejected(workloads, monkeypatch, bad):
    import jax.numpy as jnp

    monkeypatch.setattr(
        workloads,
        "prepare",
        lambda case: (
            lambda x: (x, jnp.asarray(bad)),
            ((jnp.asarray(1.0),),),
            None,
            {},
        ),
    )
    monkeypatch.setitem(sys.modules, "run", runner)
    result = workloads.measure(dict(workload="design", backend="cpu", repeats=1))
    assert not result["validated"]
    assert "non-finite" in result["error"]


def test_fixed_step_fit_does_not_require_loss_improvement(workloads):
    import jax.numpy as jnp

    _, inputs, validate, evidence = workloads.prepare(
        dict(workload="mle", n=8, queries=1, steps=2, covariance_tile=None)
    )
    validate((jnp.asarray([1.0, 2.0]), inputs[0][0]), inputs[0])
    assert evidence["final_objective"] == 2.0


@pytest.mark.parametrize("workload", ["predict", "predict-marginals"])
@pytest.mark.parametrize("queries", [1, 16])
def test_prediction_workloads_reuse_prepared_posterior(
    workloads, monkeypatch, workload, queries
):
    import equinox as eqx
    import jax
    import numpy as np

    case = dict(
        workload=workload,
        n=8,
        queries=queries,
        covariance_tile=None,
        dtype="float64" if jax.config.read("jax_enable_x64") else "float32",
    )
    operation, inputs, validate, evidence = workloads.prepare(case)
    assert evidence["conditioning_included"] is False

    def unexpected_condition(*args, **kwargs):
        raise AssertionError("conditioning must be outside the timed operation")

    from gpex import GaussianProcess

    monkeypatch.setattr(GaussianProcess, "condition", unexpected_condition)
    compiled = eqx.filter_jit(operation)
    for args in inputs:
        posterior, query = args
        reference = posterior.predict(query)
        mean, uncertainty = jax.block_until_ready(compiled(*args))
        expected = reference.covariance.full
        if workload == "predict-marginals":
            expected = np.diag(expected)
        tolerance = 3e-6 if case["dtype"] == "float32" else 1e-12
        np.testing.assert_allclose(
            mean, reference.mean.flat, rtol=tolerance, atol=tolerance
        )
        np.testing.assert_allclose(
            uncertainty, expected, rtol=tolerance, atol=tolerance
        )
        assert mean.shape == (queries,)
        assert uncertainty.shape == (
            (queries,) if workload == "predict-marginals" else (queries, queries)
        )
        validate((mean, uncertainty), args)


def test_query_sweep_plan_has_distinct_cases(tmp_path):
    command = [
        sys.executable,
        str(RUNNER_PATH),
        "--plan",
        "--workloads",
        "predict",
        "predict-marginals",
        "--sizes",
        "4096",
        "--queries",
        "1",
        "16",
        "64",
        "256",
        "1024",
        "--process-samples",
        "3",
        "--repeats",
        "20",
        "--output",
        str(tmp_path / "unused.json"),
    ]
    cases = json.loads(subprocess.check_output(command, text=True))["cases"]
    assert len(cases) == len({runner.digest(case) for case in cases}) == 30
    assert {case["queries"] for case in cases} == {1, 16, 64, 256, 1024}
    assert {case["n"] for case in cases} == {4096}
    assert not (tmp_path / "unused.json").exists()


def test_dataset_sequence_sizes_and_independence(workloads):
    datasets = __import__("datasets")
    case = dict(n=512, min_size=16, sequence_length=100, pattern="distinct")
    sizes, inputs = datasets.datasets(case)
    assert len(sizes) == len(set(sizes)) == 100
    assert min(sizes) == 16 and max(sizes) == 512
    for n, (x, y) in zip(sizes, inputs, strict=True):
        assert x.shape == (n, 1) and y.shape == (n,)
    with pytest.raises(ValueError, match="distinct sizes"):
        datasets.dataset_sizes(case | dict(pattern="repeated"))


@pytest.mark.parametrize(
    ("kind", "library"),
    [
        (kind, library)
        for kind in ("datasets", "sequential")
        for library in ("gpex", "compact", "gpjax", "gpytorch")
    ]
    + [("sequential", "multiblock"), ("sequential", "multiblock-rebuild")],
)
def test_sequences_match_independent_dense_reference(
    workloads, monkeypatch, kind, library
):
    import importlib

    import jax
    import numpy as np

    query_points = __import__("_data").query_points
    measure = __import__("_sequence").measure

    if library in ("gpjax", "gpytorch"):
        pytest.importorskip(library)
    names = {
        "datasets": [
            "datasets",
            "datasets-compact",
            "datasets-gpjax",
            "datasets-gpytorch",
        ],
        "sequential": [
            "sequential",
            "sequential-rebuild",
            "sequential-gpjax",
            "sequential-gpytorch",
            "sequential-multiblock",
            "sequential-rebuild-multiblock",
        ],
    }
    method = names[kind][
        [
            "gpex",
            "compact",
            "gpjax",
            "gpytorch",
            "multiblock",
            "multiblock-rebuild",
        ].index(library)
    ]
    case = dict(
        workload=method,
        n=6,
        start_size=2,
        min_size=2,
        sequence_length=4,
        pattern="distinct",
        queries=3,
        backend="cpu",
        dtype="float64" if jax.config.x64_enabled else "float32",
    )
    module = importlib.import_module(kind)
    monkeypatch.setitem(sys.modules, "run", runner)
    events = []
    result = measure(case, module.prepare(case), phase=events.append)
    assert result["validated"], result.get("error")
    assert result["timing_protocol"] == "initialized-runtime-v1"
    assert result["runtime_initialization_ms"] >= 0
    assert events == ["measure_start", "measure_end"]
    assert len(result["step_ms"]) == 4
    expected_traces = (
        (1 if kind == "datasets" else 2)
        if library in ("gpex", "compact")
        else 4 + (kind == "sequential")
    )
    if library == "compact" and kind == "datasets":
        expected_traces = 4
    if library.startswith("multiblock"):
        expected_traces = 3  # Initial conditioning and the two static block targets.
    if library != "gpytorch":
        assert result["trace_count"] == expected_traces
    if kind == "datasets":
        _, inputs = module.datasets(case)
    else:
        x = np.random.default_rng(2026).uniform(-4.0, 4.0, (6, 1))
        y = np.sin(1.7 * x[:, 0]) + 0.1 * np.cos(3.1 * x[:, 0])
        inputs = [(x[:n], y[:n]) for n in range(2, 7)]
    query = query_points(3)

    def kernel(a, b):
        return np.exp(
            -0.5 * ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1) / 0.8**2
        )

    for (x, y), output in zip(inputs, result["outputs"], strict=True):
        k = kernel(x, x) + (0.05 + 1e-6) * np.eye(len(x))
        cross = kernel(x, query)
        mean = cross.T @ np.linalg.solve(k, y)
        covariance = kernel(query, query) - cross.T @ np.linalg.solve(k, cross)
        if library == "gpjax":
            covariance += 1e-6 * np.eye(3)
        # These small well-conditioned systems differ only by float32 arithmetic.
        tol = 3e-6 if case["dtype"] == "float32" else 2e-10
        np.testing.assert_allclose(
            np.asarray(output[0]).reshape(-1), mean, atol=tol, rtol=tol
        )
        np.testing.assert_allclose(output[1], covariance, atol=tol, rtol=tol)
    row = dict(
        result,
        schema_version=runner.SCHEMA,
        status="complete",
        case=case,
        fingerprint="same",
    )
    assert runner.valid(row, case, "same")
    assert not runner.valid(row | dict(step_ms=result["step_ms"][:-1]), case, "same")
    assert not runner.valid(row | dict(total_ms=-1), case, "same")


@pytest.mark.parametrize("failure", ["nan", "negative", "exception"])
def test_sequence_failure_keeps_partial_evidence(workloads, monkeypatch, failure):
    import jax.numpy as jnp

    Sequence, measure = (
        __import__("_sequence").Sequence,
        __import__("_sequence").measure,
    )

    def step(state, value):
        if failure == "exception":
            raise ValueError("injected failure")
        covariance = jnp.full((1, 1), jnp.nan if failure == "nan" else -1.0)
        return None, (value, covariance)

    case = dict(workload="datasets", backend="cpu", dtype="float64")
    monkeypatch.setitem(sys.modules, "run", runner)
    result = measure(case, Sequence(step, [(jnp.ones(1),)] * 2, [2, 3], "gpex"))
    assert not result["validated"]
    assert result["error"]
    assert len(result["step_ms"]) < 2
    if failure != "exception":
        assert result["outputs"]


@pytest.mark.parametrize("method", ["datasets", "datasets-gpjax"])
def test_distinct_sizes_reuse_only_reserved_jax_trace(workloads, monkeypatch, method):
    import jax

    if method.endswith("gpjax"):
        pytest.importorskip("gpjax")
    module = __import__("datasets")
    measure = __import__("_sequence").measure
    case = dict(
        workload=method,
        n=12,
        min_size=2,
        sequence_length=10,
        pattern="distinct",
        queries=1,
        backend="cpu",
        dtype="float64" if jax.config.x64_enabled else "float32",
    )
    monkeypatch.setitem(sys.modules, "run", runner)
    result = measure(case, module.prepare(case))
    assert result["validated"], result.get("error")
    assert result["trace_count"] == (10 if method.endswith("gpjax") else 1)


def test_initialization_does_not_execute_or_cache_the_workload(workloads, monkeypatch):
    """Initialization must not turn the target's first call into a warm call."""
    import jax.numpy as jnp

    traces = []

    def operation(x):
        traces.append(x.shape)
        return x * 2

    monkeypatch.setattr(
        workloads, "prepare", lambda case: (operation, ((jnp.ones(3),),), None, {})
    )
    monkeypatch.setitem(sys.modules, "run", runner)
    result = workloads.measure(dict(workload="dense", backend="cpu", repeats=2))
    assert result["validated"]
    assert traces == [(3,)]
    assert result["timing_protocol"] == "initialized-runtime-v1"
    assert result["runtime_initialization_ms"] >= 0
    assert result["first_call_ms"] == pytest.approx(
        result["lower_ms"] + result["compile_ms"] + result["first_execute_ms"]
    )
