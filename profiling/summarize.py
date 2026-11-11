"""Keep compact measurements and provenance from a complete campaign."""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from run import check_report, sequence_module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    """Weight independent processes equally, preserving sequence positions."""
    values = np.asarray(values)
    return dict(
        median=np.median(values, axis=0).tolist(),
        min=np.min(values, axis=0).tolist(),
        max=np.max(values, axis=0).tolist(),
        q25=np.percentile(values, 25, axis=0).tolist(),
        q75=np.percentile(values, 75, axis=0).tolist(),
    )


def sequence_summary(rows):
    case = rows[0]["case"]
    counts = rows[0]["observation_counts"]
    if any(r["observation_counts"] != counts for r in rows):
        raise ValueError("Sequence observation counts differ between processes")
    memory_key = (
        "peak_bytes_in_use" if case["backend"] == "gpu" else "process_peak_rss_bytes"
    )
    return dict(
        measurement="sequence",
        settings={k: v for k, v in case.items() if k != "sample"},
        processes=len(rows),
        observation_counts=counts,
        timing="complete sequence including compilation and initial conditioning",
        **{
            name: distribution([r[name] for r in rows])
            for name in ("total_ms", "initial_ms", "step_ms", "cumulative_ms")
        },
        process_totals_ms=[r["total_ms"] for r in rows],
        trace_counts=[r.get("trace_count") for r in rows],
        memory=max(r["memory"][memory_key] for r in rows),
        validation=[r["validation"] for r in rows],
    )


def load(paths):
    groups, coverage, sources, seen = defaultdict(list), Counter(), [], set()
    sequences = defaultdict(list)
    for path in paths:
        report = json.loads(path.read_text())
        check_report(report)
        sources.append(
            dict(
                path=str(path),
                sha256=digest(path),
                fingerprint=report["fingerprint"],
                provenance=report["provenance"],
                backend=report["cases"][0]["backend"],
            )
        )
        rows = {json.dumps(r["case"], sort_keys=True): r for r in report["rows"]}
        for case in report["cases"]:
            identity = json.dumps(case, sort_keys=True)
            assert identity not in seen, case
            seen.add(identity)
            row = rows[json.dumps(case, sort_keys=True)]
            coverage[(case["backend"], case["dtype"], row["status"])] += 1
            assert row["status"] == "complete" and row["validated"], (path, case)
            if "outputs" in row:

                def finite_values(value):
                    if isinstance(value, list):
                        return all(finite_values(v) for v in value)
                    return isinstance(value, (int, float)) and np.isfinite(value)

                assert finite_values(row["outputs"]), (path, case)
            if sequence_module(case):
                settings = {k: v for k, v in case.items() if k != "sample"}
                sequences[json.dumps(settings, sort_keys=True)].append(row)
                continue
            assert np.isfinite(row["samples_ms"]).all() and min(row["samples_ms"]) > 0
            key = tuple(
                case[k] for k in ("backend", "dtype", "workload", "n", "queries")
            )
            groups[key].append(row)
    points = {}
    for key, rows in groups.items():
        backend, dtype, workload, n, queries = key
        divisor = rows[0]["case"]["steps"] if workload in ("mle", "map") else 1
        warm = [float(np.median(r["samples_ms"])) / divisor for r in rows]
        first = [r["first_call_ms"] for r in rows]
        displayed = first if workload in ("append", "rebuild") else warm
        compilation = (
            [r["lower_ms"] + r["compile_ms"] for r in rows]
            if "compile_ms" in rows[0]
            else None
        )
        memory_key = (
            "peak_bytes_in_use" if backend == "gpu" else "process_peak_rss_bytes"
        )
        peaks = [r["memory"][memory_key] for r in rows]
        metrics = dict(warm_ms=warm, first_call_ms=first, memory=peaks)
        if compilation is not None:
            metrics.update(
                setup_ms=compilation,
                lower_ms=[r["lower_ms"] for r in rows],
                compile_ms=[r["compile_ms"] for r in rows],
            )
        points[key] = dict(
            backend=backend,
            dtype=dtype,
            workload=workload,
            n=n,
            queries=queries,
            ms=float(np.median(displayed)),
            q25=float(np.percentile(displayed, 25)),
            q75=float(np.percentile(displayed, 75)),
            warm_ms=float(np.median(warm)),
            warm_q25=float(np.percentile(warm, 25)),
            warm_q75=float(np.percentile(warm, 75)),
            first_call_ms=float(np.median(first)),
            setup_ms=float(np.median(compilation)) if compilation else None,
            lower_ms=float(np.median([r["lower_ms"] for r in rows]))
            if compilation
            else None,
            compile_ms=float(np.median([r["compile_ms"] for r in rows]))
            if compilation
            else None,
            memory=max(peaks),
            process_ranges={
                name: dict(min=min(values), max=max(values))
                for name, values in metrics.items()
            },
            processes=len(rows),
            settings={
                k: v
                for k, v in rows[0]["case"].items()
                if k not in ("sample", "repeats", "reference_limit")
            },
            timing="first call including compilation"
            if workload in ("append", "rebuild")
            else "complete 100-step fit / 100"
            if workload in ("mle", "map")
            else "warm execution",
        )
    return points, coverage, sources, [sequence_summary(r) for r in sequences.values()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profiling-commit", required=True)
    args = parser.parse_args()
    plan = json.loads((args.campaign / "plan.json").read_text())
    manifests = [
        json.loads((args.campaign / f"{b}-{d}.json").read_text())
        for b in ("cpu", "gpu")
        for d in ("float32", "float64")
    ]
    paths = sorted((args.campaign / "cpu").glob("*/report.json"))
    paths += sorted((args.campaign / "gpu").glob("*/*.json"))
    points, coverage, sources, sequences = load(paths)
    expected = {json.dumps(c, sort_keys=True) for m in manifests for c in m["cases"]}
    actual = {
        json.dumps(c, sort_keys=True)
        for p in paths
        for c in json.loads(p.read_text())["cases"]
    }
    if actual != expected:
        raise ValueError("Report coverage does not match the full campaign manifests")
    environments, reports = [], []
    for source in sources:
        provenance = source["provenance"]
        if provenance["sources"] != manifests[0]["sources"]:
            raise ValueError("Report source hashes do not match the campaign")
        environment = {
            k: provenance[k] for k in ("packages", "python", "platform", "hardware")
        }
        environment["environment"] = {
            k: v
            for k, v in provenance["environment"].items()
            if not k.startswith("SLURM_") and k != "CUDA_VISIBLE_DEVICES"
        }
        if environment not in environments:
            environments.append(environment)
        report = json.loads(Path(source["path"]).read_text())
        reports.append(
            dict(
                report=str(Path(source["path"]).relative_to(args.campaign)),
                sha256=source["sha256"],
                environment=environments.index(environment),
                backend=report["cases"][0]["backend"],
                dtype=report["cases"][0]["dtype"],
                workloads=sorted({c["workload"] for c in report["cases"]}),
                cases=len(report["cases"]),
            )
        )
    summary = dict(
        schema_version=2,
        source=plan["source_commit"],
        profiling_commit=args.profiling_commit,
        source_hashes=manifests[0]["sources"],
        coverage=[
            dict(backend=b, dtype=d, status=s, cases=n)
            for (b, d, s), n in sorted(coverage.items())
        ],
        points=list(points.values()),
        sequences=sequences,
        environments=environments,
        reports=reports,
        units=dict(
            warm_ms="milliseconds; complete fit divided by steps for MLE/MAP",
            first_call_ms="milliseconds; lowering + compilation + first execution",
            memory="bytes; CPU process peak RSS, GPU peak live allocation",
            sequences="milliseconds; complete experiments including compilation; "
            "median, quartiles and min/max across processes at each sequence position",
            process_ranges="min/max across independent processes; warm ranges use "
            "one median per process; memory ranges use one peak per process",
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(
        f"Saved {len(points)} points and {len(sequences)} sequences "
        f"from {sum(coverage.values())} accepted cases"
    )


if __name__ == "__main__":
    main()
