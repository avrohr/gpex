"""Render the documentation figures from a compact profiling summary."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("summary", type=Path)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
summary = json.loads(args.summary.read_text())
points = summary["points"]
args.output_dir.mkdir(parents=True, exist_ok=True)
plt.rcParams["svg.hashsalt"] = "gpex-performance"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    }
)
figures = [
    (
        "01_dense_inference",
        "Conditioning and prediction",
        ["dense", "gpjax", "gpytorch"],
    ),
    ("02_fitting", "MLE and MAP fitting", ["mle", "map"]),
    ("03_dense_streaming", "Adding an observation", ["append", "rebuild"]),
    ("05_design", "Differentiable design", ["design", "design-prepared"]),
    ("06_mixed", "Mixed measurements", ["mixed"]),
    (
        "07_prediction",
        "Prediction from 4,096 observations",
        ["predict", "predict-marginals"],
    ),
]
names = {
    "dense": "gpex",
    "gpjax": "GPJax",
    "gpytorch": "GPyTorch",
    "mle": "MLE",
    "map": "MAP",
    "append": "Append",
    "rebuild": "Rebuild",
    "design": "Design",
    "design-prepared": "Prepared design",
    "mixed": "gpex",
    "predict": "Full covariance",
    "predict-marginals": "Marginal variances",
}


def plot_metric(axis, x, rows, metric, *, divisor=1, **style):
    """Draw the recorded statistic and its independent-process range."""
    axis.plot(x, [p[metric] / divisor for p in rows], **style)
    ranges = [p.get("process_ranges", {}).get(metric) for p in rows]
    # Old compact summaries do not retain these extrema. Omit their bands.
    if rows and all(r is not None for r in ranges):
        axis.fill_between(
            x,
            [r["min"] / divisor for r in ranges],
            [r["max"] / divisor for r in ranges],
            color=style["color"],
            alpha=0.09,
            linewidth=0,
        )


for stem, title, workloads in figures:
    methods = [
        (w, names[w], c)
        for w, c in zip(workloads, ["#087f86", "#bf542c", "#6858a4"], strict=False)
    ]
    prediction = stem == "07_prediction"
    cold = stem in (
        "01_dense_inference",
        "02_fitting",
        "03_dense_streaming",
        "06_mixed",
    )
    warm_row = 1 if cold else 0
    memory_row = warm_row + 1
    fitting = stem == "02_fitting"
    fig, axes = plt.subplots(
        memory_row + 1,
        2,
        figsize=(11.5, 10 if memory_row == 2 else 7.6),
        sharex=True,
        sharey="row",
        layout="constrained",
    )

    for col, (backend, device_title) in enumerate(
        [("cpu", "Apple M3 Max · CPU"), ("gpu", "NVIDIA H100 · GPU")]
    ):
        for method, _name, color in methods:
            for dtype, style, marker in [("float32", "-", "o"), ("float64", "--", "s")]:
                rows = sorted(
                    [
                        p
                        for p in points
                        if (p["backend"], p["dtype"], p["workload"])
                        == (backend, dtype, method)
                    ],
                    key=lambda p: p["queries"] if prediction else p["n"],
                )
                x = [p["queries"] if prediction else p["n"] for p in rows]
                kwargs = dict(
                    color=color,
                    ls=style,
                    marker=marker,
                    ms=5 if method == "dense" else 3.5,
                    lw=3.8 if method == "dense" else 1.5,
                    markerfacecolor="none",
                    markeredgewidth=1.2,
                    zorder=2 if method == "dense" else 3,
                )
                if cold:
                    plot_metric(
                        axes[0, col],
                        x,
                        rows,
                        "first_call_ms",
                        divisor=100 if fitting else 1,
                        **kwargs,
                    )
                plot_metric(axes[warm_row, col], x, rows, "warm_ms", **kwargs)
                plot_metric(axes[memory_row, col], x, rows, "memory", **kwargs)
        axes[0, col].set_title(device_title, fontweight="bold", pad=12)
        axes[memory_row, col].set_xlabel(
            "Prediction queries" if prediction else "Scalar observations"
        )
        for timing_row in range(memory_row):
            axes[timing_row, col].set_yscale("log")
            axes[timing_row, col].yaxis.set_major_formatter(
                FuncFormatter(
                    lambda v, pos: f"{v / 1000:g} s" if v >= 1000 else f"{v:g} ms"
                )
            )
        axes[memory_row, col].set_yscale("log", base=2)
        axes[memory_row, col].yaxis.set_major_locator(LogLocator(base=2, numticks=7))
        axes[memory_row, col].yaxis.set_major_formatter(
            FuncFormatter(
                lambda v, pos: (
                    f"{v / 2**30:g} GiB" if v >= 2**30 else f"{v / 2**20:g} MiB"
                )
            )
        )
        for ax in axes[:, col]:
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_locator(
                FixedLocator(
                    [1, 16, 64, 256, 1024]
                    if prediction
                    else [128, 512, 2048, 8192, 32768, 131072]
                )
            )
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(v):,}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(alpha=0.17)
    if cold:
        axes[0, 0].set_ylabel("Cold fit / 100" if fitting else "Cold-start runtime")
    axes[warm_row, 0].set_ylabel("Warm fit / 100" if fitting else "Warm runtime")
    axes[memory_row, 0].set_ylabel("Peak process RSS")
    axes[memory_row, 1].set_ylabel("Peak live device allocation")
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=True)
    handles = [Line2D([], [], color=c, lw=2, label=n) for _, n, c in methods]
    handles += [
        Line2D([], [], color="#444", ls="-", marker="o", label="float32"),
        Line2D(
            [],
            [],
            color="#444",
            ls="--",
            marker="s",
            markerfacecolor="white",
            label="float64",
        ),
    ]
    if any(p.get("process_ranges") for p in points if p["workload"] in workloads):
        handles.append(Patch(facecolor="#555", alpha=0.09, label="Process min–max"))
    fig.suptitle(title, fontsize=17)
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(methods) + 2,
        frameon=False,
    )
    fig.savefig(args.output_dir / f"{stem}.svg", metadata={"Date": None})
    plt.close(fig)

# Complete workflows retain their measured compilation and execution costs.
sequence_methods = [
    ("", "gpex reserved / append", "#087f86"),
    ("-compact", "gpex · exact-size arrays", "#6858a4"),
    ("-rebuild", "gpex rebuild", "#6858a4"),
    ("-gpjax", "GPJax", "#bf542c"),
    ("-gpytorch", "GPyTorch", "#426ab3"),
]
for kind, patterns, stem, title in [
    (
        "datasets",
        ["distinct"],
        "08_dataset_reuse",
        "Conditioning on changing datasets",
    ),
    ("sequential", [None], "09_sequential", "Sequential observations"),
]:
    rows = [
        p
        for p in summary.get("sequences", [])
        if p["settings"]["workload"].startswith(kind)
    ]
    if not rows:
        continue
    fig, axes = plt.subplots(
        len(patterns),
        2,
        figsize=(11.5, 8 if len(patterns) == 2 else 5),
        squeeze=False,
        sharey="row",
        layout="constrained",
    )
    present = set()
    for row, pattern in enumerate(patterns):
        for col, (backend, device) in enumerate(
            (("cpu", "Apple M3 Max · CPU"), ("gpu", "NVIDIA H100 · GPU"))
        ):
            ax = axes[row, col]
            for suffix, label, color in sequence_methods:
                for dtype, style in (("float32", "-"), ("float64", "--")):
                    selected = [
                        p
                        for p in rows
                        if (
                            p["settings"]["backend"],
                            p["settings"]["dtype"],
                            p["settings"]["workload"],
                            p["settings"].get("pattern"),
                        )
                        == (backend, dtype, kind + suffix, pattern)
                    ]
                    for p in selected:
                        settings = p["settings"]
                        present.add((suffix, label, color))
                        x = (
                            list(range(1, len(p["observation_counts"]) + 1))
                            if kind == "datasets"
                            else p["observation_counts"]
                        )
                        curves = {
                            k: [v / 1000 for v in p["cumulative_ms"][k]]
                            for k in ("median", "min", "max")
                        }
                        if kind == "sequential":
                            x = [settings["start_size"], *x]
                            curves = {
                                k: [p["initial_ms"][k] / 1000, *v]
                                for k, v in curves.items()
                            }
                        ax.plot(x, curves["median"], color=color, ls=style, lw=1.8)
                        ax.fill_between(
                            x,
                            curves["min"],
                            curves["max"],
                            color=color,
                            alpha=0.09,
                            linewidth=0,
                        )
            ax.set_title(device + (f" · {pattern} sizes" if pattern else ""))
            ax.set_xlabel(
                "Datasets processed" if kind == "datasets" else "Active observations"
            )
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(
                FuncFormatter(
                    lambda v, pos: f"{v:g} s" if v >= 1 else f"{v * 1000:g} ms"
                )
            )
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(alpha=0.17)
            ax.tick_params(labelleft=True)
        axes[row, 0].set_ylabel("Cumulative runtime, including compilation")
    handles = [
        Line2D(
            [],
            [],
            color=c,
            lw=2,
            label=("gpex reserved" if kind == "datasets" else "gpex append")
            if suffix == ""
            else label,
        )
        for suffix, label, c in sequence_methods
        if (suffix, label, c) in present
    ]
    handles += [
        Line2D([], [], color="#444", ls=style, label=dtype)
        for dtype, style in (("float32", "-"), ("float64", "--"))
    ]
    handles.append(Patch(facecolor="#555", alpha=0.09, label="Process min–max"))
    fig.suptitle(title, fontsize=17)
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    fig.savefig(args.output_dir / f"{stem}.svg", metadata={"Date": None})
    plt.close(fig)

# Homepage overview: complete float64 workflows, including first compilation.
home_methods = [
    ("sequential", "gpex", "#087f86"),
    ("sequential-gpjax", "GPJax", "#bf542c"),
    ("sequential-gpytorch", "GPyTorch", "#426ab3"),
]
home_rows = {
    (p["settings"]["backend"], p["settings"]["workload"]): p
    for p in summary.get("sequences", [])
    if p["settings"]["dtype"] == "float64"
    and p["settings"]["workload"] in {w for w, _, _ in home_methods}
    and p["settings"].get("start_size") == 512
    and p["settings"]["n"] == 1024
    and p["settings"]["queries"] == 1
}
if all(
    (device, method) in home_rows
    for device in ("cpu", "gpu")
    for method, _, _ in home_methods
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharex=True, layout="constrained")
    upper = max(p["total_ms"]["max"] / 1000 for p in home_rows.values())
    for ax, backend, title in zip(
        axes, ("cpu", "gpu"), ("Apple M3 Max · CPU", "NVIDIA H100 · GPU"), strict=True
    ):
        for i, (method, _label, color) in enumerate(home_methods):
            timing = home_rows[backend, method]["total_ms"]
            median, low, high = (timing[k] / 1000 for k in ("median", "min", "max"))
            ax.barh(i, median, color=color, height=0.5)
            ax.errorbar(
                median,
                i,
                xerr=[[median - low], [high - median]],
                fmt="none",
                ecolor="#333",
                capsize=3,
                lw=1,
            )
            ax.text(
                high + upper * 0.025,
                i,
                f"{median:.2f} s",
                va="center",
                fontweight="bold" if method == "sequential" else "normal",
            )
        ax.set_yticks(range(3), [name for _, name, _ in home_methods])
        ax.invert_yaxis()
        ax.set_xlim(0, upper * 1.2)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Complete workflow (seconds)")
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.15)
    fig.savefig(args.output_dir / "10_sequential_headline.svg", metadata={"Date": None})
    plt.close(fig)
