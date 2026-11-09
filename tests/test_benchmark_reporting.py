"""Publication statistics must weight independent processes equally."""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
spec = importlib.util.spec_from_file_location("benchmark_plots", BENCHMARKS / "plot.py")
assert spec is not None and spec.loader is not None
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)


def load(tmp_path, workload, samples, steps=100):
    fingerprint = plots.digest({})
    rows = []
    for sample, warm in enumerate(samples):
        case = dict(
            workload=workload,
            dtype="float64",
            n=16,
            backend="cpu",
            repeats=len(warm),
            steps=steps,
            sample=sample,
        )
        rows.append(
            dict(
                schema_version=3,
                status="complete",
                validated=True,
                fingerprint=fingerprint,
                case=case,
                samples_ms=warm,
                first_call_ms=123.0 + sample,
                lower_ms=2.0,
                compile_ms=3.0,
                memory=dict(process_peak_rss_bytes=1024),
            )
        )
    report = dict(
        schema_version=3,
        synthetic=False,
        status="complete",
        provenance={},
        fingerprint=fingerprint,
        label="test",
        cases=[r["case"] for r in rows],
        rows=rows,
    )
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    return plots.read_reports([path])


def test_each_process_has_equal_weight_despite_different_repeat_counts(tmp_path):
    rows = load(tmp_path, "dense", [[1.0, 3.0], [100.0] * 20])
    assert [r["samples_ms"] for r in rows] == [[2.0], [100.0]]
    assert plots.np.median([x for r in rows for x in r["samples_ms"]]) == 51.0


def test_fitting_reports_complete_fit(tmp_path):
    rows = load(tmp_path, "map", [[100.0, 300.0]])
    assert rows[0]["samples_ms"] == [200.0]


@pytest.mark.parametrize("method", ["append", "rebuild"])
def test_updates_keep_process_first_call(tmp_path, method):
    rows = load(tmp_path, method, [[1.0, 3.0], [2.0, 4.0]])
    assert [r["first_call_ms"] for r in rows] == [[123.0], [124.0]]
    assert [r["samples_ms"] for r in rows] == [[2.0], [3.0]]


def test_new_allocation_can_extend_an_identical_measurement_protocol():
    from copy import deepcopy

    report: dict[str, Any] = dict(
        fingerprint="first-job",
        provenance=dict(
            sources={"workloads.py": "same-source"},
            packages={"jax": "0.10.2"},
            environment={
                "SLURM_JOB_ID": "1",
                "CUDA_VISIBLE_DEVICES": "0",
                "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                "OMP_NUM_THREADS": "32",
            },
        ),
    )
    second = deepcopy(report)
    second["fingerprint"] = "second-job"
    second["provenance"]["environment"].update(
        SLURM_JOB_ID="2", CUDA_VISIBLE_DEVICES="3", JAX_ENABLE_X64="false"
    )
    case = dict(workload="dense", n=1024, sample=0, backend="gpu", dtype="float32")
    assert plots.comparison_protocol(report, case) == plots.comparison_protocol(
        second, dict(case, n=256)
    )
    second["provenance"]["environment"]["OMP_NUM_THREADS"] = "16"
    assert plots.comparison_protocol(report, case) != plots.comparison_protocol(
        second, case
    )
    second = deepcopy(report)
    second["provenance"]["sources"]["workloads.py"] = "changed-source"
    assert plots.comparison_protocol(report, case) != plots.comparison_protocol(
        second, case
    )


def test_partial_report_exposes_missing_and_failed_cases(tmp_path):
    load(tmp_path, "dense", [[1.0, 3.0]])
    path = tmp_path / "report.json"
    report = json.loads(path.read_text())
    failed_case = report["cases"][0] | {"n": 32}
    missing_case = report["cases"][0] | {"n": 64}
    report["cases"].extend([failed_case, missing_case])
    report["rows"].append(
        report["rows"][0]
        | {
            "case": failed_case,
            "status": "failed",
            "validated": False,
            "error": "non-finite covariance",
        }
    )
    report["status"] = "partial"
    path.write_text(json.dumps(report))
    coverage = []
    rows = plots.read_reports([path], coverage)
    assert len(rows) == 1
    assert [r["status"] for r in coverage] == ["complete", "failed", "missing"]
    plots.write_report(rows, coverage, tmp_path, [])
    text = (tmp_path / "report.html").read_text()
    assert "non-finite covariance" in text
    assert "Not measured" in text


def test_mixed_precision_reports_are_never_pooled(tmp_path):
    load(tmp_path, "dense", [[2.0]])
    first = tmp_path / "report.json"
    report = json.loads(first.read_text())
    for case in report["cases"]:
        case["dtype"] = "float32"
    for row in report["rows"]:
        row["case"]["dtype"] = "float32"
        row["samples_ms"] = [1.0]
    second = tmp_path / "float32.json"
    second.write_text(json.dumps(report))
    rows = plots.read_reports([first, second])
    assert [(r["dtype"], r["samples_ms"]) for r in rows] == [
        ("float64", [2.0]),
        ("float32", [1.0]),
    ]


@pytest.mark.parametrize("workload", ["predict", "dense"])
def test_query_counts_have_separate_results_and_coverage(tmp_path, workload):
    load(tmp_path, workload, [[2.0]])
    path = tmp_path / "report.json"
    report = json.loads(path.read_text())
    first = report["rows"][0]
    first["case"]["queries"] = 1
    second = dict(first, case=first["case"] | {"queries": 1024}, samples_ms=[20.0])
    report["rows"] = [first, second]
    report["cases"] = [r["case"] for r in report["rows"]]
    path.write_text(json.dumps(report))
    coverage = []
    rows = plots.read_reports([path], coverage)
    plots.write_report(rows, coverage, tmp_path, [])
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert [
        (r["queries"], r["median_ms"], r["processes"]) for r in summary["measurements"]
    ] == [(1, 2.0, 1), (1024, 20.0, 1)]
    assert [r["queries"] for r in summary["coverage"]] == [1, 1024]
    if workload == "dense":
        with pytest.raises(ValueError, match="queries count"):
            plots.figure(rows, tmp_path, "wrong", "Mixed queries", {workload})


@pytest.mark.parametrize("schema", [4, 5])
def test_reporting_preserves_disagreeing_results(tmp_path, schema):
    load(tmp_path, "dense", [[1.0]])
    path = tmp_path / "report.json"
    report = json.loads(path.read_text())
    report["schema_version"] = schema
    report["provenance"] = {"environment": {}}
    fingerprint = plots.digest(report["provenance"])
    report["fingerprint"] = fingerprint
    first = report["rows"][0]
    first.update(
        schema_version=schema,
        fingerprint=fingerprint,
        outputs=[[[1.0], [[1e-5]]]],
        validation={},
    )
    other = dict(
        first,
        case=first["case"] | {"workload": "gpytorch"},
        outputs=[[[1.0], [[1e-3]]]],
    )
    report["rows"].append(other)
    report["cases"].append(other["case"])
    path.write_text(json.dumps(report))
    original = path.read_bytes()
    coverage = []
    assert len(plots.read_reports([path], coverage)) == 2
    assert all(row["status"] == "complete" for row in coverage)
    assert path.read_bytes() == original


def test_sequence_totals_and_sizes_are_not_pooled_as_warm_samples(tmp_path):
    datasets = __import__("datasets")
    SCHEMA = __import__("run").SCHEMA

    fingerprint = plots.digest(__import__("run").measurement_identity({}))
    rows = []
    for maximum in (32, 64):
        case = dict(
            workload="datasets",
            pattern="distinct",
            n=maximum,
            min_size=2,
            sequence_length=20,
            queries=1,
            dtype="float64",
            backend="cpu",
            sample=0,
            repeats=5,
        )
        rows.append(
            dict(
                schema_version=SCHEMA,
                status="complete",
                validated=True,
                measurement="sequence",
                fingerprint=fingerprint,
                case=case,
                initial_ms=0.0,
                total_ms=20.0,
                step_ms=[1.0] * 20,
                cumulative_ms=list(range(1, 21)),
                observation_counts=datasets.dataset_sizes(case),
                validation=dict(checked_predictions=20),
                memory=dict(process_peak_rss_bytes=1024),
            )
        )
    report = dict(
        schema_version=SCHEMA,
        synthetic=False,
        status="complete",
        provenance={},
        fingerprint=fingerprint,
        label="test",
        cases=[r["case"] for r in rows],
        rows=rows,
    )
    path = tmp_path / "sequences.json"
    path.write_text(json.dumps(report))
    coverage, sequences = [], []
    ordinary = plots.read_reports([path], coverage, sequences)
    assert ordinary == []
    assert len(sequences) == 2
    assert {r["observations"] for r in coverage} == {32, 64}
    figures, summary = plots.sequence_figures(sequences, tmp_path)
    assert len(figures) == len(summary) == 2
    assert all(r["total_ms"] == 20 for r in summary)
    plots.write_report(ordinary, coverage, tmp_path, figures, summary)
    html = (tmp_path / "report.html").read_text()
    assert "complete experiment timings" in html
    assert "distinct" in html


def test_sequence_summary_retains_every_supported_method(tmp_path):
    methods = [
        name for name in __import__("run").WORKLOADS if name.startswith("sequential")
    ]
    rows = [
        dict(
            label="test",
            case=dict(
                workload=method,
                n=3,
                start_size=2,
                queries=1,
                dtype="float64",
                backend="cpu",
            ),
            initial_ms=0.0,
            cumulative_ms=[elapsed],
            observation_counts=[3],
            total_ms=elapsed,
            memory={},
        )
        for elapsed, method in enumerate(methods, 1)
    ]
    _, summary = plots.sequence_figures(rows, tmp_path)
    totals = sorted(row["total_ms"] for row in summary)
    assert totals == list(range(1, len(methods) + 1))


def test_runtime_protocols_are_not_pooled(tmp_path):
    load(tmp_path, "dense", [[2.0]])
    legacy = tmp_path / "report.json"
    report = json.loads(legacy.read_text())
    report["provenance"]["timing_protocol"] = "initialized-runtime-v1"
    report["fingerprint"] = plots.digest(report["provenance"])
    for case in report["cases"]:
        case["sample"] += 1
    for row in report["rows"]:
        row["case"]["sample"] += 1
        row["fingerprint"] = report["fingerprint"]
        row["timing_protocol"] = "initialized-runtime-v1"
        row["runtime_initialization_ms"] = 75.0
    initialized = tmp_path / "initialized.json"
    initialized.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="different protocols"):
        plots.read_reports([legacy, initialized])
    rows = plots.read_reports([initialized])
    assert rows[0]["runtime_initialization_ms"] == 75.0
    plots.write_report(rows, [], tmp_path, [])
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["measurements"][0]["runtime_initialization_ms"] == [75.0]
    assert "exclude a separate tiny numerical" in (tmp_path / "report.html").read_text()
