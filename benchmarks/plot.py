"""Plot accepted measurements with explicit precision and campaign coverage."""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from run import check_report, digest, sequence_module

NAMES = {
    "dense": "gpex dense",
    "gpjax": "GPJax dense",
    "gpytorch": "GPyTorch dense",
    "append": "gpex append",
    "rebuild": "gpex rebuild",
    "map": "MAP",
    "mle": "MLE",
    "design": "Design",
    "design-prepared": "Prepared design",
    "mixed": "Mixed measurements",
    "marginals": "Marginals",
    "sample": "Samples",
    "kernel-gradient": "Kernel gradient",
    "feature-gradient": "Feature gradient",
    "construction": "Construction",
    "predict": "Mean + full covariance",
    "predict-marginals": "Mean + marginal variances",
}
PREDICTION_WORKLOADS = {"predict", "predict-marginals"}
COLORS = ["#1F4E79", "#D97706", "#6D597A", "#2A9D8F", "#B23A48"]
FIGURES = [
    (
        "01_dense_inference",
        "Conditioning and prediction",
        {"dense", "gpjax", "gpytorch"},
    ),
    ("02_fitting", "MLE and MAP fitting", {"map", "mle"}),
    ("03_dense_streaming", "Append and rebuild", {"append", "rebuild"}),
    (
        "05_design",
        "Design objective and coordinate gradient",
        {"design", "design-prepared"},
    ),
    ("06_mixed", "Values, slopes and averages", {"mixed"}),
    ("08_prediction", "Prediction from an existing posterior", PREDICTION_WORKLOADS),
    (
        "07_diagnostics",
        "Additional workloads",
        set(NAMES)
        - {
            "dense",
            "gpjax",
            "gpytorch",
            "map",
            "mle",
            "append",
            "rebuild",
            "design",
            "design-prepared",
            "mixed",
            *PREDICTION_WORKLOADS,
        },
    ),
]


def comparison_protocol(report, case):
    """Do not pool different algorithms, dependencies, devices or source revisions."""
    provenance = report["provenance"]
    swept = "queries" if case["workload"] in PREDICTION_WORKLOADS else "n"
    protocol = {k: v for k, v in case.items() if k not in (swept, "sample", "repeats")}
    protocol["timing_protocol"] = provenance.get(
        "timing_protocol", "process-first-call"
    )
    protocol["sources"] = provenance.get("sources", report["fingerprint"])
    for key in ("packages", "python", "platform", "hardware"):
        protocol[key] = provenance.get(key)
    ignored = {"CUDA_VISIBLE_DEVICES", "JAX_ENABLE_X64"}
    protocol["environment"] = {
        key: value
        for key, value in provenance.get("environment", {}).items()
        if not key.startswith("SLURM_") and key not in ignored
    }
    return protocol


def read_reports(paths, coverage=None, sequences=None):
    reports = [json.loads(path.read_text()) for path in paths]
    for report in reports:
        check_report(report)
    rows, seen, protocols = [], set(), {}
    for path, report in zip(paths, reports, strict=True):
        measured = {digest(row["case"]): row for row in report["rows"]}
        for case in report["cases"]:
            precision = case.get("dtype", "unspecified")
            key = (report["label"], digest(case))
            if key in seen:
                raise ValueError(f"duplicate measurement: {key}")
            seen.add(key)
            method = case["workload"]
            fixed_size = (
                case["n"] if method in PREDICTION_WORKLOADS else case.get("queries", 1)
            )
            protocol_key = (
                report["label"],
                precision,
                method,
                fixed_size,
                case.get("pattern"),
                case.get("start_size"),
                case.get("sequence_length"),
            )
            protocol = comparison_protocol(report, case)
            if protocols.setdefault(protocol_key, protocol) != protocol:
                raise ValueError(
                    "use different labels for different protocols or source revisions"
                )
            row = measured.get(digest(case))
            status = row["status"] if row else "missing"
            if coverage is not None:
                coverage.append(
                    dict(
                        device=report["label"],
                        dtype=precision,
                        workload=method,
                        pattern=case.get("pattern", ""),
                        observations=case["n"],
                        queries=case.get("queries", 1),
                        sample=case.get("sample", 0),
                        status=status,
                        error=row.get("error", "") if row else "not measured",
                        report=str(path),
                        schema=report["schema_version"],
                        protocol=digest(protocol),
                        sources=digest(report["provenance"].get("sources", {})),
                    )
                )
            if status != "complete":
                continue
            assert row is not None
            if sequence_module(case):
                if sequences is not None:
                    sequences.append(dict(row, label=report["label"]))
                continue
            rows.append(
                dict(
                    device=report["label"],
                    backend=case["backend"],
                    dtype=precision,
                    method=NAMES[method],
                    workload=method,
                    observations=case["n"],
                    queries=case.get("queries", 1),
                    # One median per process gives each process equal weight.
                    samples_ms=[float(np.median(row["samples_ms"]))],
                    first_call_ms=[row["first_call_ms"]],
                    timing_protocol=protocol["timing_protocol"],
                    runtime_initialization_ms=row.get("runtime_initialization_ms"),
                    compilation_ms=[row["lower_ms"] + row["compile_ms"]]
                    if "compile_ms" in row
                    else [],
                    peak_bytes=row["memory"].get("peak_bytes_in_use")
                    if case["backend"] == "gpu"
                    else row["memory"]["process_peak_rss_bytes"],
                    memory_kind="live device allocation"
                    if case["backend"] == "gpu"
                    else "process RSS",
                    steps=case.get("steps"),
                )
            )
    return rows


def memory_label(mib, _):
    return f"{mib / 1024:g} GiB" if mib >= 1024 else f"{mib:g} MiB"


def figure(rows, output, name, title, workloads, *, coverage=()):
    selected = [r for r in rows if r["workload"] in workloads]
    planned = [r for r in coverage if r["workload"] in workloads]
    if not selected and not planned:
        return False
    query_sweep = workloads <= PREDICTION_WORKLOADS
    varying = "queries" if query_sweep else "observations"
    fixed = "observations" if query_sweep else "queries"
    fixed_sizes = {r[fixed] for r in [*selected, *planned]}
    if len(fixed_sizes) != 1:
        raise ValueError(f"plot each {fixed} count in a separate figure")
    if query_sweep:
        title += f" · {next(iter(fixed_sizes)):,} observations"
    precisions = sorted({r["dtype"] for r in [*selected, *planned]})
    devices = sorted({r["device"] for r in [*selected, *planned]})
    methods = sorted({r["workload"] for r in [*selected, *planned]})
    colors = {method: COLORS[i % len(COLORS)] for i, method in enumerate(methods)}
    styles: dict[str, Any] = {
        device: [("--", "o"), ("-", "^"), (":", "s"), ("-.", "D")][i % 4]
        for i, device in enumerate(devices)
    }
    if selected and all(
        r.get("timing_protocol") == "initialized-runtime-v1" for r in selected
    ):
        title += " · initialized runtime"
    fig, axes = plt.subplots(
        3,
        len(precisions),
        figsize=(6 * len(precisions), 9),
        squeeze=False,
        sharex=True,
        sharey="row",
    )
    first_call = workloads == {"append", "rebuild"}
    metric = "first_call_ms" if first_call else "samples_ms"
    for column, precision in enumerate(precisions):
        timing, compilation, memory = axes[:, column]
        timing.set_title(precision)
        for device in devices:
            for method in methods:
                part = [
                    r
                    for r in selected
                    if (r["device"], r["dtype"], r["workload"])
                    == (device, precision, method)
                ]
                grouped = defaultdict(list)
                for row in part:
                    grouped[row[varying]].append(row)
                # A failed or missing size breaks the curve instead of bridging it.
                sizes = sorted(
                    set(grouped)
                    | {
                        r[varying]
                        for r in planned
                        if (r["device"], r["dtype"], r["workload"])
                        == (device, precision, method)
                    }
                )
                linestyle, marker = styles[device]
                style = dict(
                    color=colors[method],
                    linestyle=linestyle,
                    marker=marker,
                    markersize=4,
                )
                for axis, field in ((timing, metric), (compilation, "compilation_ms")):
                    samples = [sum((r[field] for r in grouped[n]), []) for n in sizes]
                    median = [np.median(s) if s else np.nan for s in samples]
                    axis.plot(sizes, median, **style)
                    axis.fill_between(
                        sizes,
                        [np.quantile(s, 0.25) if s else np.nan for s in samples],
                        [np.quantile(s, 0.75) if s else np.nan for s in samples],
                        color=colors[method],
                        alpha=0.10,
                    )
                peaks = [
                    max(
                        (
                            r["peak_bytes"]
                            for r in grouped[n]
                            if r["peak_bytes"] is not None
                        ),
                        default=np.nan,
                    )
                    / 2**20
                    for n in sizes
                ]
                memory.plot(sizes, peaks, **style)
        failed = sum(
            r["status"] == "failed" for r in planned if r["dtype"] == precision
        )
        missing = sum(
            r["status"] == "missing" for r in planned if r["dtype"] == precision
        )
        timing.text(
            0.98,
            0.04,
            f"{failed} failed · {missing} not measured",
            transform=timing.transAxes,
            ha="right",
            fontsize=9,
            color="#555555",
        )
        memory.set_xlabel("Query points" if query_sweep else "Scalar observations")
    fit_steps = sorted(
        {r["steps"] for r in selected if r["workload"] in ("mle", "map")}
    )
    time_label = (
        f"Complete {fit_steps[0]}-step fit (ms)"
        if len(fit_steps) == 1
        else "Warm execution (ms)"
    )
    if first_call:
        time_label = "First call incl. compilation (ms)"
    axes[0, 0].set_ylabel(time_label)
    axes[1, 0].set_ylabel("Tracing + compilation (ms)")
    axes[2, 0].set_ylabel("Peak memory")
    for axis in axes.flat:
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[2]:
        axis.set_yscale("log", base=2)
        axis.yaxis.set_major_locator(LogLocator(base=2, numticks=6))
        axis.yaxis.set_major_formatter(FuncFormatter(memory_label))
        axis.yaxis.set_minor_formatter(NullFormatter())
    handles = [Line2D([], [], color=colors[m], label=NAMES[m]) for m in methods]
    handles += [
        Line2D(
            [],
            [],
            color="#444444",
            linestyle=styles[d][0],
            marker=styles[d][1],
            label=d,
        )
        for d in devices
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncols=min(3, len(handles)),
        fontsize=9,
    )
    fig.suptitle(title, y=0.995)
    fig.text(
        0.5,
        0.012,
        "Memory: CPU process RSS · GPU live allocation.\nBand: IQR across processes. "
        "Coverage details in report.html.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.89))
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "png"):
        fig.savefig(output / f"{name}.{suffix}", dpi=150)
    plt.close(fig)
    return True


SEQUENCE_NAMES = {
    "datasets": "gpex reserved",
    "datasets-compact": "gpex compact",
    "datasets-gpjax": "GPJax",
    "datasets-gpytorch": "GPyTorch",
    "sequential": "gpex append",
    "sequential-rebuild": "gpex recondition",
    "sequential-gpjax": "GPJax recondition",
    "sequential-gpytorch": "GPyTorch incremental",
    "sequential-multiblock": "gpex multiblock append",
    "sequential-rebuild-multiblock": "gpex multiblock recondition",
}


def sequence_figures(rows, output):
    groups = defaultdict(list)
    for row in rows:
        case = row["case"]
        groups[
            (
                row["label"],
                sequence_module(case),
                case["n"],
                case["queries"],
                case.get("pattern", "append"),
                case.get("start_size"),
                case.get("sequence_length"),
                case.get("min_size"),
            )
        ].append(row)
    figures, summary = [], []
    output.mkdir(parents=True, exist_ok=True)
    for index, (key, group) in enumerate(groups.items()):
        label, kind, n, queries, pattern, start_size, length, minimum = key
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, squeeze=False)
        for ax, precision in zip(axes[0], ("float32", "float64"), strict=True):
            for method_index, (method, title) in enumerate(
                (m, t) for m, t in SEQUENCE_NAMES.items() if m.startswith(kind)
            ):
                color = COLORS[method_index % len(COLORS)]
                selected = [
                    r
                    for r in group
                    if r["case"]["dtype"] == precision
                    and r["case"]["workload"] == method
                ]
                if not selected:
                    continue
                values = (
                    np.asarray(
                        [[r["initial_ms"], *r["cumulative_ms"]] for r in selected]
                    )
                    / 1000
                )
                if kind == "datasets":
                    x = np.arange(length + 1)
                    x, values = x[1:], values[:, 1:]
                else:
                    x = [start_size, *selected[0]["observation_counts"]]
                ax.plot(x, np.median(values, axis=0), color=color, label=title)
                ax.fill_between(
                    x,
                    np.percentile(values, 25, axis=0),
                    np.percentile(values, 75, axis=0),
                    color=color,
                    alpha=0.12,
                )
                memory_key = (
                    "peak_bytes_in_use"
                    if selected[0]["case"]["backend"] == "gpu"
                    else "process_peak_rss_bytes"
                )
                summary.append(
                    dict(
                        label=label,
                        dtype=precision,
                        method=title,
                        pattern=pattern,
                        n=n,
                        queries=queries,
                        start_size=start_size,
                        sequence_length=length,
                        min_size=minimum,
                        processes=len(selected),
                        timing_protocol=selected[0].get(
                            "timing_protocol", "process-first-call"
                        ),
                        runtime_initialization_ms=[
                            r["runtime_initialization_ms"]
                            for r in selected
                            if r.get("runtime_initialization_ms") is not None
                        ],
                        total_ms=float(np.median([r["total_ms"] for r in selected])),
                        peak_bytes=max(
                            (
                                r["memory"][memory_key]
                                for r in selected
                                if memory_key in r["memory"]
                            ),
                            default=None,
                        ),
                    )
                )
            ax.set_title(precision)
            ax.set_xlabel(
                "Datasets conditioned" if kind == "datasets" else "Active observations"
            )
            ax.grid(alpha=0.2)
            if ax.lines:
                ax.legend(fontsize=8)
        axes[0, 0].set_ylabel("Cumulative time including compilation (s)")
        title = (
            f"{label} · {kind} · {pattern} · maximum {n:,} observations"
            f" · {queries} queries"
        )
        fig.suptitle(title)
        fig.tight_layout()
        name = f"09_sequence_{index}_{kind}_{pattern}"
        for suffix in ("svg", "png"):
            fig.savefig(output / f"{name}.{suffix}", dpi=150)
        plt.close(fig)
        figures.append((name, title))
    return figures, summary


def write_report(rows, coverage, output, figures, sequence_rows=()):
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["device"],
                row["dtype"],
                row["workload"],
                row["observations"],
                row["queries"],
            )
        ].append(row)
    for (device, precision, workload, n, queries), group in grouped.items():
        summary.append(
            dict(
                device=device,
                dtype=precision,
                workload=workload,
                observations=n,
                queries=queries,
                processes=len(group),
                timing_protocol=group[0].get("timing_protocol", "process-first-call"),
                runtime_initialization_ms=[
                    r["runtime_initialization_ms"]
                    for r in group
                    if r.get("runtime_initialization_ms") is not None
                ],
                median_ms=float(np.median([r["samples_ms"][0] for r in group])),
                first_call_ms=float(np.median([r["first_call_ms"][0] for r in group])),
                peak_bytes=max(
                    (r["peak_bytes"] for r in group if r["peak_bytes"] is not None),
                    default=None,
                ),
            )
        )
    (output / "summary.json").write_text(
        json.dumps(
            dict(measurements=summary, coverage=coverage), indent=2, allow_nan=False
        )
        + "\n"
    )
    (output / "sequences.json").write_text(
        json.dumps(sequence_rows, indent=2, allow_nan=False) + "\n"
    )
    sequence_table = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(cell))}</td>"
            for cell in (
                row["label"],
                row["dtype"],
                row["method"],
                row["pattern"],
                row["processes"],
                f"{row['total_ms'] / 1000:.3f}",
                memory_label(row["peak_bytes"] / 2**20, None)
                if row["peak_bytes"] is not None
                else "unavailable",
            )
        )
        + "</tr>"
        for row in sequence_rows
    )
    sequence_html = (
        (
            "<h2>Complete sequences</h2><p>Totals include compilation and "
            "all synchronized inference calls. Sequential totals also include initial "
            "conditioning and cache preparation. Checks and input preparation "
            "are excluded. "
            "These are complete experiment timings, not warm per-call averages.</p>"
            "<table><tr><th>Machine</th><th>Precision</th><th>Method</th><th>Pattern</th>"
            "<th>Processes</th><th>Total (s)</th><th>Peak memory</th></tr>"
            + sequence_table
            + "</table>"
        )
        if sequence_rows
        else ""
    )
    counts = defaultdict(lambda: defaultdict(int))
    reasons = defaultdict(set)
    for row in coverage:
        key = (
            row["device"],
            row["dtype"],
            row["workload"],
            row["observations"],
            row["queries"],
            row.get("pattern", ""),
        )
        counts[key][row["status"]] += 1
        if row["error"]:
            reasons[key].add(row["error"])
    table = []
    for key, statuses in sorted(counts.items()):
        cells = [*key, statuses["complete"], statuses["failed"], statuses["missing"]]
        details = (
            "<details><summary>Failure details</summary><pre>"
            + html.escape("\n".join(sorted(reasons[key])))
            + "</pre></details>"
            if reasons[key]
            else ""
        )
        table.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in cells)
            + f"<td>{details}</td></tr>"
        )
    images = "".join(
        f'<figure><img src="{name}.svg" alt="{html.escape(title)}"></figure>'
        for name, title in figures
    )
    legacy = any(row["schema"] < 5 for row in coverage)
    note = (
        "Historical reports retain their original numerical checks. "
        "They have not been revalidated with the new criteria."
        if legacy
        else "All accepted measurements passed the recorded numerical checks."
    )
    if len({row["sources"] for row in coverage}) > 1:
        note += (
            " These reports measure multiple source revisions. Differences "
            "are not an isolated comparison of arithmetic precision."
        )
    if any(
        r.get("timing_protocol") == "initialized-runtime-v1"
        for r in [*rows, *sequence_rows]
    ):
        note += (
            " Initialized-runtime measurements exclude a separate tiny numerical "
            "initialization pass. First-call and sequence times still include "
            "target-workload tracing, compilation and first execution. "
            "Initialization durations are retained separately in the JSON results."
        )
    inputs = sorted({row["report"] for row in coverage})
    input_list = "".join(f"<li>{html.escape(path)}</li>" for path in inputs)
    text = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>gpex performance</title><style>
body{{font:16px system-ui;max-width:1200px;margin:40px auto;
padding:0 24px;color:#253447}}img{{width:100%}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{padding:8px;text-align:left;border-bottom:1px solid #ddd}}
pre{{white-space:pre-wrap;max-width:600px}}figure{{margin:40px 0}}
li{{overflow-wrap:anywhere}}
</style><h1>gpex performance</h1><p>{html.escape(note)}</p>
<p>Float32 and float64 are plotted separately on shared scales.
Curves contain accepted processes only. A failed or unmeasured size
breaks the line. Sizes with fewer accepted repetitions remain visible
in the coverage table.</p>
<p>Fitting times cover the complete fit. Update times include compilation.
Tracing and compilation are also shown separately. CPU RSS includes the
process, while GPU memory is live device allocation.</p>
<p>Prediction-only figures vary query count at a fixed observation count.
Conditioning is outside their timed calls. Memory includes posterior preparation
and storage, not just the additional prediction allocations.</p>
<details><summary>Input reports</summary><ul>{input_list}</ul></details>
{sequence_html}{images}<h2>Coverage</h2><table><thead><tr>
<th>Machine</th><th>Precision</th><th>Workload</th><th>Observations</th>
<th>Queries</th><th>Pattern</th>
<th>Accepted</th><th>Failed</th><th>Not measured</th><th>Details</th>
</tr></thead><tbody>{"".join(table)}</tbody></table></html>"""
    (output / "report.html").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    coverage = []
    sequences = []
    rows = read_reports(args.results, coverage, sequences)
    figures = []
    for name, title, methods in FIGURES:
        query_sweep = methods <= PREDICTION_WORKLOADS
        fixed = "observations" if query_sweep else "queries"
        counts = sorted(
            {r[fixed] for r in [*rows, *coverage] if r["workload"] in methods}
        )
        for count in counts:
            suffix = f"_n{count}" if query_sweep else f"_q{count}"
            figure_name = name + suffix if query_sweep or len(counts) > 1 else name
            figure_title = title if query_sweep else f"{title} · {count:,} query points"
            if figure(
                [r for r in rows if r[fixed] == count],
                args.output_dir,
                figure_name,
                figure_title,
                methods,
                coverage=[r for r in coverage if r[fixed] == count],
            ):
                figures.append((figure_name, figure_title))
    sequence_figs, sequence_summary = sequence_figures(sequences, args.output_dir)
    figures.extend(sequence_figs)
    write_report(rows, coverage, args.output_dir, figures, sequence_summary)
    print(f"Saved {len(figures)} figures and coverage report to {args.output_dir}")


if __name__ == "__main__":
    main()
