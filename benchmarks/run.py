"""Isolated, resumable performance measurements. See README.md for examples."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _runtime import TIMING_PROTOCOL

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = (
    "dense",
    "mixed",
    "predict",
    "predict-marginals",
    "marginals",
    "sample",
    "kernel-gradient",
    "feature-gradient",
    "construction",
    "append",
    "rebuild",
    "mle",
    "map",
    "design",
    "design-prepared",
    "gpjax",
    "gpytorch",
    "datasets",
    "datasets-compact",
    "datasets-gpjax",
    "datasets-gpytorch",
    "sequential",
    "sequential-rebuild",
    "sequential-multiblock",
    "sequential-rebuild-multiblock",
    "sequential-gpjax",
    "sequential-gpytorch",
)
SCHEMA = 6


def sequence_module(case):
    name = case["workload"].split("-")[0]
    return name if name in ("datasets", "sequential") else None


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def provenance():
    paths = sorted(
        [
            *(ROOT / "src/gpex").rglob("*.py"),
            *(p for p in (ROOT / "benchmarks").glob("*.py") if p.name != "plot.py"),
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
        ]
    )
    packages = {}
    for name in (
        "gpex",
        "cholops",
        "jax",
        "jaxlib",
        "equinox",
        "numpy",
        "optax",
        "gpjax",
        "torch",
        "gpytorch",
        "scipy",
        "linear-operator",
        "jax-cuda13-plugin",
        "jax-cuda13-pjrt",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(
        sources={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
        timing_protocol=TIMING_PROTOCOL,
        packages=packages,
        python=sys.version,
        platform=platform.platform(),
        host=socket.gethostname(),
        hardware=hardware(),
        environment={
            key: value
            for key, value in sorted(os.environ.items())
            if key == "GPEX_BENCHMARK_RUNTIME_SHA256"
            or key.startswith(
                ("JAX_", "XLA_", "OMP_", "MKL_", "OPENBLAS_", "CUDA_VISIBLE", "SLURM_")
            )
        },
    )


def hardware():
    """Actual execution hardware; no scheduler IDs or machine names."""
    if sys.platform == "darwin":
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    else:
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            platform.processor(),
        )
    info: dict[str, Any] = dict(
        cpu=cpu,
        architecture=platform.machine(),
        cpu_count=os.cpu_count(),
        affinity_count=len(os.sched_getaffinity(0))  # pyright: ignore[reportAttributeAccessIssue]
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count(),
    )
    if {"cuda", "gpu"} & set(os.environ.get("JAX_PLATFORMS", "").split(",")):
        # Let CUDA resolve numeric indices, UUIDs, MIG and enumeration order.
        # The probe creates no contexts and never initializes CUDA in this process.
        info["gpu"] = json.loads(
            subprocess.check_output(
                [sys.executable, str(Path(__file__).with_name("_gpu.py"))],
                text=True,
            )
        )
    return info


def measurement_identity(source):
    """Reusable identity; retain full provenance separately for auditing."""
    resources = {
        "SLURM_CPUS_PER_TASK",
        "SLURM_CPUS_ON_NODE",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
        "SLURM_JOB_GPUS",
    }
    environment = {
        k: v
        for k, v in source.get("environment", {}).items()
        if (not k.startswith("SLURM_") or k in resources)
        and k not in {"CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS"}
    }
    return {k: v for k, v in source.items() if k not in {"host", "environment"}} | {
        "environment": environment
    }


def memory():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "process_peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024)
    }
    if "jax" in sys.modules:
        stats = sys.modules["jax"].devices()[0].memory_stats() or {}
        result.update(
            {
                k: int(stats[k])
                for k in ("peak_bytes_in_use", "peak_pool_bytes")
                if k in stats
            }
        )
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        result["peak_bytes_in_use"] = int(
            sys.modules["torch"].cuda.max_memory_allocated()
        )
    return result


def valid(row, case, fingerprint):
    """Only successful results from this exact case and environment are reusable."""
    if isinstance(row, dict) and "timing_protocol" in row:
        elapsed = row.get("runtime_initialization_ms")
        if (
            row["timing_protocol"] != TIMING_PROTOCOL
            or not isinstance(elapsed, (float, int))
            or isinstance(elapsed, bool)
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            return False
    if isinstance(row, dict) and sequence_module(case):
        return valid_sequence(row, case, fingerprint)
    if not isinstance(row, dict) or not isinstance(row.get("samples_ms"), list):
        return False
    return (
        row.get("schema_version") in (3, 4, 5, SCHEMA)
        and row.get("status") == "complete"
        and row.get("case") == case
        and row.get("fingerprint") == fingerprint
        and row.get("validated") is True
        and len(row["samples_ms"]) == case["repeats"]
        and all(
            type(x) in (float, int) and math.isfinite(x) and x >= 0
            for x in [row.get("first_call_ms", -1), *row.get("samples_ms", [])]
        )
    )


def valid_sequence(row, case, fingerprint):
    """Require all timed steps and predictions, retaining failed evidence separately."""
    from datasets import dataset_sizes

    counts = (
        dataset_sizes(case)
        if sequence_module(case) == "datasets"
        else list(range(case["start_size"] + 1, case["n"] + 1))
    )
    times = row.get("step_ms", [])
    initial, total = row.get("initial_ms", -1), row.get("total_ms", -1)
    if not (
        row.get("schema_version") == SCHEMA
        and row.get("measurement") == "sequence"
        and row.get("status") == "complete"
        and row.get("validated") is True
        and row.get("case") == case
        and row.get("fingerprint") == fingerprint
        and row.get("observation_counts") == counts
        and isinstance(times, list)
        and len(times) == len(counts)
        and all(
            type(x) in (int, float) and math.isfinite(x) and x >= 0
            for x in [initial, total, *times]
        )
        and row.get("validation", {}).get("checked_predictions")
        == (len(counts) + (sequence_module(case) == "sequential"))
        and isinstance(row.get("cumulative_ms"), list)
        and len(row["cumulative_ms"]) == len(times)
    ):
        return False
    elapsed = initial
    for delta, cumulative in zip(times, row["cumulative_ms"], strict=True):
        elapsed += delta
        if not isinstance(cumulative, (int, float)) or not math.isclose(
            elapsed, cumulative, rel_tol=1e-12
        ):
            return False
    return math.isclose(total, elapsed, rel_tol=1e-12)


def competing_processes():
    """Identify active Python/test computation, excluding our entire process tree."""
    output = subprocess.check_output(
        ["ps", "-axo", "pid=,ppid=,%cpu=,command="], text=True
    )
    processes = {}
    for line in output.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) == 4:
            pid, parent, cpu, command = fields
            processes[int(pid)] = (int(parent), float(cpu), command)
    owned = {os.getpid()}
    while True:
        expanded = owned | {pid for pid, row in processes.items() if row[0] in owned}
        if expanded == owned:
            break
        owned = expanded
    return [
        dict(pid=pid, cpu_percent=cpu, command=command)
        for pid, (_, cpu, command) in processes.items()
        if pid not in owned
        and cpu >= 5
        and any(
            name in Path(command.split()[0]).name.lower()
            for name in ("python", "pytest", "jupyter")
        )
    ]


def wait_for_quiet(timeout=300, quiet_seconds=30):
    # Settle after competing work, not before every already-idle case.
    if not competing_processes():
        return True
    deadline = time.monotonic() + timeout
    quiet = time.monotonic()
    while time.monotonic() < deadline:
        if competing_processes():
            quiet = time.monotonic()
        if time.monotonic() - quiet >= quiet_seconds:
            return True
        time.sleep(1)
    return False


def timing_overlap(events, observations):
    intervals, start = [], None
    for event in events:
        if event["phase"] == "measure_start":
            start = event["time"]
        elif event["phase"] == "measure_end" and start is not None:
            intervals.append((start, event["time"]))
            start = None
    if start is not None:
        intervals.append((start, float("inf")))
    return [
        o
        for o in observations
        if o["processes"] and any(a <= o["time"] <= b for a, b in intervals)
    ]


def execute(
    case,
    fingerprint,
    component,
    timeout,
    trace_dir=None,
    *,
    guard=False,
) -> dict[str, Any]:
    component = Path(component)
    component.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=component.parent))
    request, output = attempt / "request.json", attempt / "result.json"
    phases = attempt / "phases.jsonl"
    write(
        request,
        dict(
            case=case,
            fingerprint=fingerprint,
            trace_dir=trace_dir,
            phases=str(phases),
        ),
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        str(request),
        "--output",
        str(output),
    ]
    start = time.monotonic()
    observations = []
    try:
        if guard:
            with (
                (attempt / "stdout.txt").open("w") as stdout,
                (attempt / "stderr.txt").open("w") as stderr,
            ):
                process = subprocess.Popen(
                    command, cwd=ROOT, stdout=stdout, stderr=stderr
                )
                try:
                    while process.poll() is None:
                        if time.monotonic() - start > timeout:
                            raise subprocess.TimeoutExpired(command, timeout)
                        try:
                            active = competing_processes()
                        except (
                            OSError,
                            ValueError,
                            subprocess.SubprocessError,
                        ) as error:
                            active = [dict(monitor_error=str(error))]
                        observations.append(
                            dict(time=time.monotonic(), processes=active)
                        )
                        time.sleep(1)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
            completed = subprocess.CompletedProcess(
                command,
                process.returncode,
                (attempt / "stdout.txt").read_text(),
                (attempt / "stderr.txt").read_text(),
            )
            completed.check_returncode()
        else:
            scratch = os.environ.get("GPEX_BENCHMARK_SCRATCH")
            case_directory = (
                tempfile.TemporaryDirectory(prefix="case-", dir=scratch)
                if scratch and case.get("backend") == "gpu"
                else nullcontext()
            )
            with case_directory as temporary:
                if temporary is None:
                    completed = subprocess.run(
                        command,
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        check=True,
                    )
                else:
                    cache = Path(temporary) / "cache"
                    cache.mkdir()
                    environment = os.environ | {
                        "TMPDIR": temporary,
                        "CUDA_CACHE_PATH": str(cache),
                    }
                    with (
                        (attempt / "stdout.txt").open("w") as stdout,
                        (attempt / "stderr.txt").open("w") as stderr,
                    ):
                        completed = subprocess.run(
                            command,
                            cwd=ROOT,
                            env=environment,
                            stdout=stdout,
                            stderr=stderr,
                            text=True,
                            timeout=timeout,
                            check=True,
                        )
        row: dict[str, Any] = json.loads(output.read_text())
        failed = (
            isinstance(row, dict)
            and row.get("status") == "failed"
            and row.get("case") == case
            and row.get("fingerprint") == fingerprint
        )
        if not valid(row, case, fingerprint) and not failed:
            raise ValueError("worker returned an invalid result")
        if completed.stdout is not None:
            (attempt / "stdout.txt").write_text(completed.stdout)
            (attempt / "stderr.txt").write_text(completed.stderr)
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        stderr = getattr(error, "stderr", "") or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        if not stderr and (attempt / "stderr.txt").exists():
            stderr = (attempt / "stderr.txt").read_text()
        (attempt / "stderr.txt").write_text(stderr)
        row = dict(
            schema_version=SCHEMA,
            status="failed",
            case=case,
            fingerprint=fingerprint,
            validated=False,
            error=str(error),
            stderr=stderr,
            elapsed_seconds=time.monotonic() - start,
        )
    if guard:
        events = (
            [json.loads(line) for line in phases.read_text().splitlines()]
            if phases.exists()
            else []
        )
        overlap = timing_overlap(events, observations)
        write(attempt / "activity.json", observations)
        if overlap:
            row.update(
                status="failed",
                validated=False,
                timing_invalid=True,
                error="Timing rejected: concurrent CPU work during measurement",
                contention=overlap,
            )
        elif row.get("status") == "complete" and not events:
            row.update(
                status="failed",
                validated=False,
                timing_invalid=True,
                error="Timing rejected: measurement phase signals missing",
            )
    published = os.environ.get("GPEX_BENCHMARK_OUTPUT_ROOT")
    row["attempt"] = str(
        Path(published) / component.parent.name / attempt.name
        if published and case.get("backend") == "gpu"
        else attempt
    )
    write(output, row)
    write(component, row)
    return row


def worker(request, output):
    payload = json.loads(request.read_text())
    case = payload["case"]
    if digest(measurement_identity(provenance())) != payload["fingerprint"]:
        raise ValueError("source or environment changed before worker started")

    def phase(name):
        if payload.get("phases"):
            with Path(payload["phases"]).open("a") as stream:
                stream.write(json.dumps(dict(phase=name, time=time.monotonic())) + "\n")

    if sequence_module(case):
        import importlib

        from _sequence import measure as measure_sequence

        def measure(case, trace_dir=None, *, phase=None):
            module = importlib.import_module(case["workload"].split("-")[0])
            return measure_sequence(case, module.prepare(case), phase=phase)
    elif case["workload"] == "gpytorch":
        from compare import measure
    else:
        from workloads import measure
    try:
        result = measure(
            case,
            payload.get("trace_dir"),
            phase=phase,
        )
    finally:
        if digest(measurement_identity(provenance())) != payload["fingerprint"]:
            raise ValueError("source or environment changed during measurement")
    accepted = result.get("validated") is True
    row = dict(
        schema_version=SCHEMA,
        status="complete" if accepted else "failed",
        case=case,
        fingerprint=payload["fingerprint"],
        **result,
    )
    if accepted and not valid(row, case, payload["fingerprint"]):
        raise ValueError("invalid timing samples")
    write(output, row)


def check_report(report):
    schema = report.get("schema_version")
    if schema not in (3, 4, 5, SCHEMA) or report.get("synthetic") is not False:
        raise ValueError("expected a measured schema-3 through schema-6 report")
    source = report.get("provenance", {})
    identity = measurement_identity(source) if schema >= 4 else source
    if digest(identity) != report.get("fingerprint"):
        raise ValueError("provenance fingerprint mismatch")
    cases, rows = report.get("cases", []), report.get("rows", [])
    protocol = source.get("timing_protocol")
    if protocol is not None and any(
        row.get("status") == "complete" and row.get("timing_protocol") != protocol
        for row in rows
    ):
        raise ValueError("measurement timing protocol differs from provenance")
    ids = {digest(c) for c in cases}
    row_ids = [digest(r.get("case")) for r in rows]
    if (
        not cases
        or len(ids) != len(cases)
        or len(set(row_ids)) != len(rows)
        or not set(row_ids) <= ids
    ):
        raise ValueError("empty, duplicate or unexpected case coverage")
    if report.get("status") not in ("complete", "partial", "paused"):
        raise ValueError("unknown report status")
    if report.get("status") == "complete" and len(rows) != len(cases):
        raise ValueError("incomplete case coverage")
    for row in rows:
        if row.get("fingerprint") != report["fingerprint"]:
            raise ValueError("case/provenance mismatch")
        if row["status"] == "complete":
            if not valid(row, row["case"], report["fingerprint"]):
                raise ValueError("unvalidated row")
        elif row["status"] != "failed" or not row.get("error"):
            raise ValueError("missing failure evidence")


def main(*, default_workloads=None, default_sizes=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--guard-cpu", action="store_true")
    parser.add_argument("--contention-retries", type=int, default=2)
    parser.add_argument("--quiet-timeout", type=float, default=300)
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=WORKLOADS,
        default=None,
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=default_sizes or [32, 128]
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=100,
        help="Total independent datasets in each sequence",
    )
    parser.add_argument("--min-size", type=int, default=16)
    parser.add_argument(
        "--start-size",
        type=int,
        default=512,
        help="Initial observations for sequential updates; --sizes is final size",
    )
    parser.add_argument("--queries", nargs="+", type=int, default=[1])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--process-samples",
        type=int,
        default=1,
        help="Independent processes per case or complete sequence",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--label", default=platform.node())
    parser.add_argument("--covariance-tile", type=int)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trace", type=Path, help="Separate untimed warm JAX traces")
    args = parser.parse_args()
    dtype = (
        json.loads(args.worker.read_text())["case"]["dtype"]
        if args.worker
        else args.dtype
    )
    os.environ["JAX_ENABLE_X64"] = "1" if dtype == "float64" else "0"
    if args.worker:
        worker(args.worker, args.output)
        return
    selected_workloads = args.workloads
    args.workloads = (
        args.workloads
        or default_workloads
        or [
            "dense",
            "mixed",
            "append",
            "rebuild",
            "map",
            "design",
        ]
    )
    if (
        min(
            *args.sizes,
            *args.queries,
            args.repeats,
            args.process_samples,
            args.steps,
        )
        < 1
        or args.timeout <= 0
    ):
        parser.error("counts and timeout must be positive")
    if "mixed" in args.workloads and min(args.sizes) < 3:
        parser.error("mixed measurements need at least three observations")
    if (
        len(set(args.sizes)) != len(args.sizes)
        or len(set(args.workloads)) != len(args.workloads)
        or len(set(args.queries)) != len(args.queries)
    ):
        parser.error("duplicate sizes, workloads or query counts")
    common = {
        k: getattr(args, k)
        for k in (
            "repeats",
            "steps",
            "backend",
            "covariance_tile",
        )
    }
    cases = []
    for w in args.workloads:
        for n in args.sizes:
            for q in args.queries:
                for sample in range(args.process_samples):
                    case = dict(
                        common,
                        workload=w,
                        n=n,
                        queries=q,
                        sample=sample,
                        dtype=args.dtype,
                    )
                    if sequence_module(case) == "datasets":
                        from datasets import dataset_sizes

                        item = case | dict(
                            pattern="distinct",
                            min_size=args.min_size,
                            sequence_length=args.sequence_length,
                        )
                        try:
                            dataset_sizes(item)
                        except ValueError as error:
                            parser.error(str(error))
                        cases.append(item)
                    elif sequence_module(case) == "sequential":
                        if not 0 < args.start_size < n:
                            parser.error(
                                "sequential requires 0 < --start-size < final --sizes"
                            )
                        cases.append(case | dict(start_size=args.start_size))
                    else:
                        cases.append(case)
    # GPU work remains the default; CPU is also needed for error callbacks.
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        cases = [
            c
            for c in manifest["cases"]
            if not selected_workloads or c["workload"] in selected_workloads
        ]
        if not cases or len({(c["dtype"], c["backend"]) for c in cases}) != 1:
            raise ValueError("Manifest execution requires one backend and dtype")
        dtype, backend = cases[0]["dtype"], cases[0]["backend"]
        os.environ["JAX_ENABLE_X64"] = "1" if dtype == "float64" else "0"
        os.environ["JAX_PLATFORMS"] = "cuda,cpu" if backend == "gpu" else "cpu"
    else:
        os.environ["JAX_PLATFORMS"] = "cuda,cpu" if args.backend == "gpu" else "cpu"
    if args.trace and any(
        c["workload"].startswith("gp") or sequence_module(c) for c in cases
    ):
        parser.error("--trace supports single-operation gpex workloads only")
    if args.plan:
        print(json.dumps(dict(cases=cases), indent=2))
        return
    source = provenance()
    if args.manifest and source["sources"] != manifest["sources"]:
        raise ValueError("Manifest source hashes differ from executable source")
    fingerprint = digest(measurement_identity(source))
    folder = args.output.parent / (args.output.stem + "_components")
    folder.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = dict(
        schema_version=SCHEMA,
        synthetic=False,
        status="partial",
        created_at=datetime.now(UTC).isoformat(),
        label=args.label,
        provenance=source,
        fingerprint=fingerprint,
        cases=cases,
        rows=[],
    )
    exhausted = {}
    for case in cases:
        component = folder / (
            digest(dict(case=case, fingerprint=fingerprint)) + ".json"
        )
        row: dict[str, Any] | None = None
        if args.resume and component.exists() and not args.trace:
            try:
                cached = json.loads(component.read_text())
                if valid(cached, case, fingerprint):
                    row = cached
            except (ValueError, OSError):
                pass
        if row is None:
            trace = (
                str(
                    args.trace.resolve()
                    / (
                        f"{case['workload']}-{case['n']}"
                        f"-q{case['queries']}-{case['sample']}"
                    )
                )
                if args.trace
                else None
            )
            setting = digest(
                {k: v for k, v in case.items() if k not in ("sample", "repeats")}
            )
            guard = args.guard_cpu and case["backend"] == "cpu"
            if setting in exhausted:
                row = dict(
                    schema_version=SCHEMA,
                    status="failed",
                    case=case,
                    fingerprint=fingerprint,
                    validated=False,
                    not_executed=True,
                    error="Skipped repeat after allocation failure",
                    evidence=exhausted[setting],
                )
            else:
                for _retry in range(args.contention_retries + 1):
                    if guard and not wait_for_quiet(args.quiet_timeout):
                        report.update(
                            status="paused",
                            pause_reason="CPU remained busy before measurement",
                        )
                        write(args.output, report)
                        raise SystemExit(2)
                    row = execute(
                        case,
                        fingerprint,
                        component,
                        args.timeout,
                        trace,
                        guard=guard,
                    )
                    if not row.get("timing_invalid"):
                        break
                assert row is not None
                if row.get("timing_invalid"):
                    report["rows"].append(row)
                    report.update(
                        status="paused",
                        pause_reason="CPU contention retry limit reached",
                    )
                    write(args.output, report)
                    raise SystemExit(2)
            assert row is not None
            failure = row.get("error", "") + row.get("stderr", "")
            if row["status"] == "failed" and any(
                text in failure.lower()
                for text in (
                    "out of memory",
                    "resource_exhausted",
                    "failed to allocate",
                    "std::bad_alloc",
                )
            ):
                exhausted[setting] = row.get("attempt", str(component))
        assert row is not None
        report["rows"].append(row)
        write(args.output, report)
        print(
            f"{case['workload']} n={case['n']} "
            f"queries={case['queries']} sample={case['sample']} "
            f"{case.get('pattern', '')}: {row['status']}",
            flush=True,
        )
    report["status"] = "complete"
    check_report(report)
    write(args.output, report)
    if any(row["status"] == "failed" for row in report["rows"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
