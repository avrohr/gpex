"""Coverage and interrupted-publication tests for grouped GPU profiling."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pytest
from campaign import cases, gpu_jobs

ROOT = Path(__file__).resolve().parents[1]


def test_gpu_job_groups_preserve_the_case_matrix_and_filters():
    plan = json.loads((ROOT / "profiling/plan.json").read_text())
    rows = [line.split() for line in gpu_jobs(plan)]
    assert len(rows) == 8
    for dtype in ("float32", "float64"):
        planned = cases(plan, "gpu", dtype)
        assigned = [
            case
            for precision, _, _, *workloads in rows
            if precision == dtype
            for workload in workloads
            for case in planned
            if case["workload"] == workload
        ]
        assert len(assigned) == 318
        assert Counter(map(json.dumps, assigned)) == Counter(map(json.dumps, planned))
    assert [line.split() for line in gpu_jobs(plan, "float64", ("dense", "gpjax"))] == [
        ["float64", "conditioning", "45", "dense", "gpjax"]
    ]
    with pytest.raises(ValueError, match="Unknown GPU workloads"):
        list(gpu_jobs(plan, workloads=("typo",)))


@pytest.fixture
def worker(tmp_path):
    if shutil.which("rsync") is None:
        pytest.skip("Grouped worker publishes with rsync")
    tools = tmp_path / "bin"
    tools.mkdir()
    for name, program in {
        "module": "#!/bin/sh\nexit 0\n",
        "findmnt": "#!/bin/sh\nexit 0\n",
        "nvidia-smi": "#!/bin/sh\nexec sleep 30\n",
        "setsid": (
            f"#!{sys.executable}\nimport os, sys\n"
            "os.setsid()\nos.execv(sys.argv[1], sys.argv[1:])\n"
        ),
    }.items():
        tool = tools / name
        tool.write_text(program)
        tool.chmod(0o755)

    run = tmp_path / "run"
    run.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    fake_runner = tmp_path / "fake_runner.py"
    fake_runner.write_text(
        """
import json, sys, time
from pathlib import Path
out = Path(sys.argv[sys.argv.index("--output") + 1])
attempt = out.parent / (out.stem + "_components") / "attempt-test"
attempt.mkdir(parents=True)
(attempt / "stderr.txt").write_text("evidence")
out.write_text(json.dumps({"status": "partial" if out.stem == "slow" else "complete"}))
if out.stem == "slow":
    time.sleep(30)
raise SystemExit(out.stem == "failed")
"""
    )
    (run / "stage.sh").write_text(
        f"""
mkdir -p "$2/source/.venv/bin" "$2/source/benchmarks"
ln -s "{sys.executable}" "$2/source/.venv/bin/python"
cp "{fake_runner}" "$2/source/benchmarks/run.py"
printf 'tree\\tarchive_bytes\\tsha256\\nvenv\\t1\\tfixture\\n' > "$3"
"""
    )
    return (
        run,
        scratch,
        os.environ
        | {
            "PATH": f"{tools}:{os.environ['PATH']}",
            "TMPDIR": str(scratch),
            "SLURM_JOB_ID": "fixture",
            "SLURM_CPUS_PER_TASK": "32",
        },
    )


def test_failed_workload_publishes_evidence_and_allows_following_work(worker):
    run, _, env = worker
    process = subprocess.run(
        [
            "bash",
            ROOT / "cluster/worker.sh",
            run,
            "float32",
            "fixture",
            "failed",
            "next",
        ],
        env=env,
        timeout=15,
        check=False,
    )
    assert process.returncode != 0
    results = run / "gpu/float32"
    assert (
        results / "failed_components/attempt-test/stderr.txt"
    ).read_text() == "evidence"
    assert (results / "failed.json").is_file()
    assert (results / "next.json").is_file()


def test_warning_publishes_active_partial_workload(worker):
    run, scratch, env = worker
    process = subprocess.Popen(
        ["bash", ROOT / "cluster/worker.sh", run, "float32", "fixture", "slow", "next"],
        env=env,
    )
    try:
        deadline = time.monotonic() + 10
        while not list(scratch.glob("*/results/slow.json")):
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.02)
        process.send_signal(signal.SIGUSR1)
        assert process.wait(timeout=10) != 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
    results = run / "gpu/float32"
    assert json.loads((results / "slow.json").read_text())["status"] == "partial"
    assert (
        results / "slow_components/attempt-test/stderr.txt"
    ).read_text() == "evidence"
    assert not (results / "next.json").exists()
