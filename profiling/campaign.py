"""Freeze committed source and expand the CPU or GPU workload manifests."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def cases(plan, backend, dtype):
    grids = plan["sizes"][backend] | (
        plan.get("size_overrides", {}).get(backend, {}).get(dtype, {})
    )
    tile_overrides = plan.get("tile_overrides", {}).get(backend, {}).get(dtype, {})
    result = [
        dict(
            plan["settings"],
            backend=backend,
            dtype=dtype,
            tile=tile_overrides.get(workload, plan["tile"][backend]),
            workload=workload,
            n=n,
            queries=queries,
            sample=sample,
            repeats=2
            if workload in ("append", "rebuild")
            else 5
            if workload in ("mle", "map")
            else 20,
        )
        for workload, sizes in grids.items()
        for n in sizes
        for queries in plan.get("query_sweeps", {}).get(
            workload, [plan["settings"]["queries"]]
        )
        for sample in range(3)
    ]
    expanded = []
    for case in result:
        family = case["workload"].split("-")[0]
        options = plan.get("sequences", {}).get(family, {})
        expanded.append(case | options)
    return expanded


def source_manifest(source):
    paths = [
        *(source / "src/gpex").rglob("*.py"),
        *(p for p in (source / "benchmarks").glob("*.py") if p.name != "plot.py"),
        source / "pyproject.toml",
        source / "uv.lock",
    ]
    return {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }


def complete_workload(run, dtype, workload):
    from run import check_report

    manifest = json.loads((run / f"gpu-{dtype}-{workload}.json").read_text())
    try:
        report = json.loads((run / "gpu" / dtype / f"{workload}.json").read_text())
        check_report(report)
    except (OSError, ValueError, AttributeError, KeyError, TypeError):
        return False
    return (
        report["status"] == "complete"
        and report["cases"] == manifest["cases"]
        and report.get("provenance", {}).get("sources") == manifest["sources"]
        and all(row["status"] == "complete" for row in report["rows"])
    )


def gpu_jobs(plan, dtype=None, workloads=(), *, unfinished_in=None):
    groups = plan["helma"]["jobs"]
    assigned = Counter(
        workload for group in groups.values() for workload in group["workloads"]
    )
    if assigned != Counter(plan["sizes"]["gpu"].keys()):
        raise ValueError("HELMA job groups must cover each GPU workload once")
    unknown = set(workloads) - assigned.keys()
    if unknown:
        raise ValueError(f"Unknown GPU workloads: {', '.join(sorted(unknown))}")
    for precision in (dtype,) if dtype else ("float32", "float64"):
        for name, group in groups.items():
            selected = [
                workload
                for workload in group["workloads"]
                if (not workloads or workload in workloads)
                and (
                    unfinished_in is None
                    or not complete_workload(unfinished_in, precision, workload)
                )
            ]
            if selected:
                yield f"{precision}\t{name}\t{group['minutes']}\t{' '.join(selected)}\n"


def prepare(run, backend, resume=False):
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "src",
            "benchmarks",
            "pyproject.toml",
            "uv.lock",
            "profiling/plan.json",
            "profiling/campaign.py",
            "profiling/cpu.sh",
            "cluster/worker.sh",
            *(("cluster/stage.sh",) if backend == "gpu" else ()),
        ],
        cwd=ROOT,
        text=True,
    )
    if dirty:
        raise ValueError(
            "Commit or stash source, plan and worker changes before profiling"
        )
    plan = json.loads((ROOT / "profiling/plan.json").read_text())
    plan["source_commit"] = revision
    jobs = list(gpu_jobs(plan)) if backend == "gpu" else []
    scripts = (
        ("cluster/worker.sh", "cluster/stage.sh")
        if backend == "gpu"
        else ("profiling/cpu.sh",)
    )
    source = run / "source"
    if run.exists():
        if not resume or json.loads((run / "plan.json").read_text()) != plan:
            raise ValueError(
                "Existing run requires --resume and the same committed source/plan"
            )
        manifest = json.loads((run / f"{backend}-float32.json").read_text())
        if source_manifest(source) != manifest["sources"]:
            raise ValueError("Frozen source has changed")
        if backend == "gpu":
            for script in scripts:
                if (run / Path(script).name).read_bytes() != (
                    ROOT / script
                ).read_bytes():
                    raise ValueError(f"Frozen script has changed: {script}")
        return
    run.mkdir(parents=True)
    archive = subprocess.check_output(
        [
            "git",
            "archive",
            revision,
            "src",
            "benchmarks",
            "pyproject.toml",
            "uv.lock",
            "README.md",
            "LICENSE",
        ],
        cwd=ROOT,
    )
    source.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(source, filter="data")
    hashes = source_manifest(source)
    for dtype in ("float32", "float64"):
        manifest = dict(
            source_commit=revision, sources=hashes, cases=cases(plan, backend, dtype)
        )
        save(run / f"{backend}-{dtype}.json", manifest)
        if backend == "gpu":
            for workload in plan["sizes"]["gpu"]:
                save(
                    run / f"gpu-{dtype}-{workload}.json",
                    manifest
                    | {
                        "cases": [
                            c for c in manifest["cases"] if c["workload"] == workload
                        ]
                    },
                )
        print(f"{backend} {dtype}: {len(manifest['cases'])} cases")
    for script in scripts:
        (run / Path(script).name).write_bytes((ROOT / script).read_bytes())
    if jobs:
        (run / "jobs.tsv").write_text("".join(jobs))
    save(run / "plan.json", plan)
    print(f"Source: {revision}\nCampaign: {run}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list-gpu-jobs", action="store_true")
    parser.add_argument("--dtype", choices=("float32", "float64"))
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument("--unfinished-in", type=Path)
    args = parser.parse_args()
    if args.list_gpu_jobs:
        if args.backend or args.output or args.resume:
            parser.error("Job listing and campaign preparation are separate")
        plan = json.loads((ROOT / "profiling/plan.json").read_text())
        if args.unfinished_in:
            sys.path.insert(0, str(ROOT / "benchmarks"))
        print(
            "".join(
                gpu_jobs(
                    plan, args.dtype, args.workload, unfinished_in=args.unfinished_in
                )
            ),
            end="",
        )
        return
    if (
        not (args.backend and args.output)
        or args.dtype
        or args.workload
        or args.unfinished_in
    ):
        parser.error(
            "Use --backend and --output to prepare a campaign. "
            "Filters require --list-gpu-jobs."
        )
    prepare(args.output.resolve(), args.backend, args.resume)


if __name__ == "__main__":
    main()
