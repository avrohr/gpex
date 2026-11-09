"""Execute tutorial and example notebooks in isolated kernels.

Run from the checkout:
    python scripts/verify_tutorials.py [--write] [names...]

Use --write before building the documentation to save the executed outputs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*")
    parser.add_argument(
        "--write",
        action="store_true",
        help="save executed notebooks with their outputs",
    )
    args = parser.parse_args()

    paths = sorted(
        [
            *(ROOT / "docs/tutorials").glob("*.ipynb"),
            *(ROOT / "examples").glob("*.ipynb"),
        ]
    )
    if args.names:
        paths = [path for path in paths if path.stem in args.names]
        if len(paths) != len(args.names):
            parser.error("unknown or duplicate notebook name")

    with tempfile.TemporaryDirectory() as cache_directory:
        environment = {
            **os.environ,
            "MPLBACKEND": "module://matplotlib_inline.backend_inline",
            "PYTHONPATH": str(ROOT / "src"),
            "MPLCONFIGDIR": str(Path(cache_directory) / "matplotlib"),
            "XDG_CACHE_HOME": str(Path(cache_directory) / "cache"),
        }
        # Build fonts once, outside the kernels that capture tutorial outputs.
        subprocess.run(
            [sys.executable, "-c", "import matplotlib.font_manager"],
            env=environment,
            check=True,
        )

        for path in paths:
            start = time.monotonic()
            notebook = nbformat.read(path, as_version=4)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    cell.outputs = []
                    cell.execution_count = None
                    cell.metadata.pop("execution", None)

            with tempfile.TemporaryDirectory() as directory:
                client = NotebookClient(
                    notebook,
                    timeout=600,
                    kernel_name="python3",
                    record_timing=False,
                    resources={"metadata": {"path": directory}},
                )
                client.execute(env=environment)

            image_count = sum(
                "image/png" in output.get("data", {})
                for cell in notebook.cells
                for output in cell.get("outputs", [])
            )
            if image_count == 0:
                raise RuntimeError(
                    f"{path.name}: expected at least one figure, found none"
                )
            if args.write:
                nbformat.write(notebook, path)
            print(f"PASS {path.name} ({time.monotonic() - start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
