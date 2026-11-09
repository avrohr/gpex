"""Execution integrity and worker environment checks."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))

spec = importlib.util.spec_from_file_location(
    "execution_runner", Path(__file__).parents[1] / "benchmarks/run.py"
)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_identity_excludes_attempt_not_hardware():
    source: dict[str, Any] = dict(
        sources={"run.py": "hash"},
        packages={"jax": "1"},
        hardware={"gpu": "H100"},
        host="host1",
        environment={
            "SLURM_JOB_ID": "1",
            "SLURM_CPUS_PER_TASK": "32",
            "JAX_ENABLE_X64": "0",
            "OMP_NUM_THREADS": "32",
            "CUDA_VISIBLE_DEVICES": "0",
        },
    )
    other = source | {
        "host": "host2",
        "environment": source["environment"]
        | {"SLURM_JOB_ID": "2", "CUDA_VISIBLE_DEVICES": "3"},
    }
    assert runner.measurement_identity(source) == runner.measurement_identity(other)
    for replacement in (
        {"hardware": {"gpu": "H200"}},
        {"packages": {"jax": "2"}},
        {"environment": source["environment"] | {"JAX_ENABLE_X64": "1"}},
        {
            "environment": source["environment"]
            | {"XLA_FLAGS": "--xla_gpu_autotune_level=4"}
        },
    ):
        assert runner.measurement_identity(source) != runner.measurement_identity(
            source | replacement
        )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_inherited_flags_reach_workers_and_resume(dtype, tmp_path, monkeypatch):
    inherited = "--xla_gpu_autotune_level=4"
    monkeypatch.setenv("XLA_FLAGS", inherited)
    output = tmp_path / "report.json"
    command = [
        sys.executable,
        str(Path(__file__).parents[1] / "benchmarks/run.py"),
        "--output",
        str(output),
        "--dtype",
        dtype,
        "--workloads",
        "dense",
        "--sizes",
        "8",
        "--repeats",
        "1",
        "--resume",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    report = json.loads(output.read_text())
    assert report["provenance"]["environment"]["XLA_FLAGS"] == inherited
    assert report["rows"][0]["status"] == "complete"
    assert (
        report["rows"][0]["timing_protocol"] == report["provenance"]["timing_protocol"]
    )
    assert report["rows"][0]["runtime_initialization_ms"] >= 0
    component = next((tmp_path / "report_components").glob("*.json"))
    modified = component.stat().st_mtime_ns
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    resumed = json.loads(output.read_text())
    assert resumed["fingerprint"] == report["fingerprint"]
    assert resumed["rows"] == report["rows"]
    assert component.stat().st_mtime_ns == modified


@pytest.fixture
def gpu_probe(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "gpu_probe", Path(__file__).parents[1] / "benchmarks/_gpu.py"
    )
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    class Driver:
        # These are already CUDA-visible devices, in CUDA's resolved order.
        devices = [("H100", 80 * 2**30, 132, 9, 0), ("A100", 40 * 2**30, 108, 8, 0)]

        def cuInit(self, flags):
            return 0

        def cuDeviceGetCount(self, output):
            output._obj.value = len(self.devices)
            return 0

        def cuDeviceGet(self, output, ordinal):
            output._obj.value = 100 + ordinal
            return 0

        def cuDeviceGetName(self, output, length, device):
            output.value = self.devices[device.value - 100][0].encode()
            return 0

        def cuDeviceTotalMem_v2(self, output, device):
            output._obj.value = self.devices[device.value - 100][1]
            return 0

        def cuDeviceGetAttribute(self, output, attribute, device):
            output._obj.value = self.devices[device.value - 100][
                {16: 2, 75: 3, 76: 4}[attribute]
            ]
            return 0

    driver = Driver()
    monkeypatch.setattr(probe.ctypes, "CDLL", lambda name: driver)
    monkeypatch.setattr(
        probe.subprocess, "check_output", lambda *args, **kwargs: "570.133.20\n"
    )
    return probe, driver


def test_gpu_probe_preserves_visible_order_partition_size_and_driver(gpu_probe):
    probe, driver = gpu_probe
    original = probe.hardware()
    assert [gpu["name"] for gpu in original["devices"]] == ["H100", "A100"]
    assert original["devices"][0]["memory_bytes"] == 80 * 2**30
    assert original["driver_version"] == "570.133.20"
    driver.devices = driver.devices[::-1]
    assert probe.hardware()["devices"] == original["devices"][::-1]
    # MIG partitions can share a GPU model while exposing different resources.
    driver.devices = [("H100", 10 * 2**30, 16, 9, 0)]
    partition = probe.hardware()["devices"][0]
    assert partition["memory_bytes"] == 10 * 2**30
    assert partition["multiprocessors"] == 16
    assert partition != original["devices"][0]


def test_gpu_probe_errors_remain_visible(gpu_probe, monkeypatch):
    probe, driver = gpu_probe
    driver.devices = []
    with pytest.raises(RuntimeError, match="no visible devices"):
        probe.hardware()
    monkeypatch.setattr(driver, "cuInit", lambda flags: 100)
    with pytest.raises(RuntimeError, match="cuInit failed with CUDA error 100"):
        probe.hardware()


@pytest.mark.parametrize("visible", ["2,0", "GPU-bbbb,GPU-aaaa", "MIG-GPU-aaaa/1/2"])
def test_gpu_hardware_uses_an_isolated_probe_with_original_selection(
    visible, monkeypatch
):
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    selected = {"devices": [{"name": "H100"}], "driver_version": "570.133.20"}

    def output(command, **kwargs):
        if command[0] == "sysctl":
            return "CPU"
        assert command == [
            sys.executable,
            str(Path(__file__).parents[1] / "benchmarks/_gpu.py"),
        ]
        assert "env" not in kwargs  # Inherit CUDA selection; do not reinterpret it.
        assert runner.os.environ["CUDA_VISIBLE_DEVICES"] == visible
        assert runner.os.environ["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"
        return json.dumps(selected)

    # Leave platform.processor() free to run its own CPU-discovery subprocess.
    monkeypatch.setattr(runner, "subprocess", SimpleNamespace(check_output=output))
    assert runner.hardware()["gpu"] == selected


def test_gpu_identity_distinguishes_hardware_order_but_not_equivalent_allocations():
    h100 = dict(name="H100", memory_bytes=80 * 2**30, multiprocessors=132)
    a100 = dict(name="A100", memory_bytes=40 * 2**30, multiprocessors=108)

    def identity(devices, visible, release="570.133.20"):
        return runner.measurement_identity(
            dict(
                hardware={"gpu": dict(devices=devices, driver_version=release)},
                environment={"CUDA_VISIBLE_DEVICES": visible},
            )
        )

    assert identity([h100], "0") == identity([h100], "GPU-another")
    assert identity([h100], "0") != identity([a100], "1")
    assert identity([h100, a100], "0,1") != identity([a100, h100], "1,0")
    assert identity([h100], "0") != identity([h100, h100], "0,2")
    assert identity([h100], "0") != identity([h100], "0", release="570.133.21")


def test_guard_excludes_validation_interval():
    events = [dict(phase="measure_start", time=10), dict(phase="measure_end", time=20)]
    observations = [dict(time=t, processes=[{"pid": 7}]) for t in (5, 15, 25)]
    assert runner.timing_overlap(events, observations) == observations[1:2]


def test_idle_machine_does_not_wait_between_cases(monkeypatch):
    monkeypatch.setattr(runner, "competing_processes", lambda: [])

    def unexpected_sleep(seconds):
        raise AssertionError("idle cases should start immediately")

    monkeypatch.setattr(runner.time, "sleep", unexpected_sleep)
    assert runner.wait_for_quiet()


def test_cpu_guard_waits_for_settling_after_competing_work(monkeypatch):
    clock = [0]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        runner.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    monkeypatch.setattr(
        runner, "competing_processes", lambda: [{"pid": 7}] if clock[0] < 5 else []
    )
    assert runner.wait_for_quiet(timeout=20, quiet_seconds=10)
    assert clock[0] >= 14


def test_partial_paused_and_complete_coverage():
    case = dict(workload="dense", n=8, repeats=1)
    source = {"environment": {}}
    fingerprint = runner.digest(runner.measurement_identity(source))
    report = dict(
        schema_version=4,
        synthetic=False,
        status="paused",
        provenance=source,
        fingerprint=fingerprint,
        cases=[case],
        rows=[],
    )
    runner.check_report(report)
    with pytest.raises(ValueError, match="incomplete"):
        runner.check_report(report | {"status": "complete"})
    row: dict[str, Any] = dict(
        schema_version=4,
        case=case,
        fingerprint=fingerprint,
        status="failed",
        error="numerical failure",
    )
    runner.check_report(report | {"status": "complete", "rows": [row]})
    with pytest.raises(ValueError, match="duplicate"):
        runner.check_report(report | {"rows": [row, row]})


def test_failure_keeps_measurement_and_source_checked_after(tmp_path, monkeypatch):
    case = dict(workload="dense", n=8, repeats=1)
    calls = []

    def provenance():
        calls.append(1)
        return {"environment": {}}

    monkeypatch.setattr(runner, "provenance", provenance)
    result = dict(
        validated=False,
        error="variance",
        outputs=[-0.1],
        memory={"peak": 5},
        samples_ms=[1],
        first_call_ms=2,
        validation={"error": 0.1},
    )
    monkeypatch.setitem(
        sys.modules, "workloads", SimpleNamespace(measure=lambda *a, **k: result)
    )
    request, output = tmp_path / "request.json", tmp_path / "out.json"
    runner.write(
        request,
        dict(
            case=case,
            fingerprint=runner.digest(runner.measurement_identity(provenance())),
        ),
    )
    calls.clear()
    runner.worker(request, output)
    row = json.loads(output.read_text())
    assert row["status"] == "failed"
    assert row["memory"] == result["memory"]
    assert row["samples_ms"] == [1]
    assert len(calls) == 2


def test_source_change_detected_even_after_measure_raises(tmp_path, monkeypatch):
    sources = iter(
        [{"environment": {}}, {"environment": {}, "sources": {"changed": "hash"}}]
    )
    monkeypatch.setattr(runner, "provenance", lambda: next(sources))

    def measure(*args, **kwargs):
        raise ValueError("original error")

    monkeypatch.setitem(sys.modules, "workloads", SimpleNamespace(measure=measure))
    request = tmp_path / "request.json"
    runner.write(
        request,
        dict(
            case={"workload": "dense"}, fingerprint=runner.digest({"environment": {}})
        ),
    )
    with pytest.raises(ValueError, match="source or environment changed"):
        runner.worker(request, tmp_path / "out.json")


def test_query_sizes_have_independent_allocation_failures_and_traces(
    tmp_path, monkeypatch
):
    output = tmp_path / "report.json"
    monkeypatch.setattr(runner, "provenance", lambda: {"environment": {}})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--output",
            str(output),
            "--workloads",
            "predict",
            "--sizes",
            "8",
            "--queries",
            "1024",
            "1",
            "--process-samples",
            "2",
            "--trace",
            str(tmp_path / "traces"),
        ],
    )
    calls = []

    def execute(case, fingerprint, component, timeout, trace, **kwargs):
        calls.append((case["queries"], case["sample"], trace))
        failed = case["queries"] == 1024
        result = dict(
            schema_version=runner.SCHEMA,
            case=case,
            fingerprint=fingerprint,
            status="failed" if failed else "complete",
            validated=not failed,
            first_call_ms=1.0,
            samples_ms=[1.0] * case["repeats"],
        )
        if failed:
            result["error"] = "out of memory"
        return result

    monkeypatch.setattr(runner, "execute", execute)
    with pytest.raises(SystemExit, match="1"):
        runner.main()
    assert [(q, sample) for q, sample, _ in calls] == [(1024, 0), (1, 0), (1, 1)]
    assert len({trace for _, _, trace in calls}) == 3
    rows = json.loads(output.read_text())["rows"]
    assert [row["status"] for row in rows] == [
        "failed",
        "failed",
        "complete",
        "complete",
    ]


@pytest.mark.parametrize("source", ["cli", "manifest"])
@pytest.mark.parametrize("workload", ["gpjax", "gpytorch"])
def test_trace_rejects_comparison_cases_before_dispatch(
    source, workload, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    arguments = [
        "run.py",
        "--output",
        str(tmp_path / "report.json"),
        "--trace",
        str(tmp_path / "traces"),
    ]
    if source == "manifest":
        manifest = tmp_path / "manifest.json"
        runner.write(
            manifest,
            dict(cases=[dict(workload=workload, dtype="float64", backend="cpu")]),
        )
        arguments += ["--manifest", str(manifest)]
    else:
        arguments += ["--workloads", workload]
    monkeypatch.setattr(sys, "argv", arguments)

    def unexpected_dispatch(*args, **kwargs):
        raise AssertionError("unsupported trace must fail before launching a worker")

    monkeypatch.setattr(runner, "execute", unexpected_dispatch)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert (
        "--trace supports single-operation gpex workloads only"
        in capsys.readouterr().err
    )


def test_manifest_trace_validation_uses_selected_cases(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    manifest = tmp_path / "manifest.json"
    cases = [
        dict(workload=workload, dtype="float64", backend="cpu")
        for workload in ("dense", "gpjax")
    ]
    runner.write(manifest, dict(cases=cases))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--output",
            str(tmp_path / "report.json"),
            "--manifest",
            str(manifest),
            "--workloads",
            "dense",
            "--trace",
            str(tmp_path / "traces"),
            "--plan",
        ],
    )
    runner.main()
    assert json.loads(capsys.readouterr().out)["cases"] == cases[:1]
