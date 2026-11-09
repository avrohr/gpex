"""Check that built tutorial pages contain every saved notebook output.

Run after verify_tutorials.py --write and mkdocs build --strict.
"""

from __future__ import annotations

import argparse
import base64
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


@dataclass
class Output:
    text: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)


class NotebookPage(HTMLParser):
    def __init__(self, html: str):
        super().__init__()
        self.outputs: dict[str, list[Output]] = {}
        self.stack: list[tuple[str, str | None, Output | None]] = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        cell = self.stack[-1][1] if self.stack else None
        output = self.stack[-1][2] if self.stack else None
        if "jp-CodeCell" in classes and attrs.get("id", "").startswith("cell-id="):
            cell = attrs["id"].removeprefix("cell-id=")
            self.outputs[cell] = []
        if cell and "jp-OutputArea-output" in classes:
            output = Output()
            self.outputs[cell].append(output)
        if output is not None and tag == "img":
            output.images.append(attrs.get("src", ""))
        if tag not in VOID_TAGS:
            self.stack.append((tag, cell, output))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        if self.stack and self.stack[-1][2] is not None:
            self.stack[-1][2].text.append(data)


def normalized(text: str) -> str:
    return " ".join(ANSI.sub("", text).split())


def check_notebook(path: Path, site: Path) -> tuple[int, int]:
    notebook = nbformat.read(path, as_version=4)
    page = NotebookPage((site / "tutorials" / path.stem / "index.html").read_text())
    output_count = image_count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        if cell.execution_count is None:
            raise ValueError(f"{path.name}: unexecuted cell {cell.id}")
        rendered = page.outputs.get(cell.id)
        if rendered is None:
            raise ValueError(f"{path.name}: missing code cell {cell.id}")
        if len(rendered) != len(cell.outputs):
            raise ValueError(f"{path.name}: output count differs in cell {cell.id}")
        for saved, output in zip(cell.outputs, rendered, strict=True):
            output_count += 1
            if saved.output_type == "error":
                raise ValueError(f"{path.name}: saved execution error in {cell.id}")
            data = saved.get("data", {})
            image_mime = next(
                (m for m in ("image/png", "image/jpeg") if m in data), None
            )
            if image_mime:
                prefix = f"data:{image_mime};base64,"
                expected = base64.b64decode(data[image_mime])
                if not any(
                    src.startswith(prefix)
                    and base64.b64decode(src.removeprefix(prefix)) == expected
                    for src in output.images
                ):
                    raise ValueError(
                        f"{path.name}: missing or changed image in {cell.id}"
                    )
                image_count += 1
            elif saved.output_type == "stream" or set(data) == {"text/plain"}:
                expected_text = saved.get("text", data.get("text/plain", ""))
                if normalized(expected_text) != normalized("".join(output.text)):
                    raise ValueError(f"{path.name}: changed text output in {cell.id}")
            else:
                raise ValueError(
                    f"{path.name}: unchecked output MIME types {list(data)}"
                )
    downloaded = site / "tutorials" / path.stem / path.name
    if nbformat.read(downloaded, as_version=4) != notebook:
        raise ValueError(f"{path.name}: downloadable notebook differs from source")
    return output_count, image_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, default=ROOT / "site")
    parser.add_argument("--source-version", help="version directory in a combined site")
    args = parser.parse_args()
    site = args.site_dir / args.source_version if args.source_version else args.site_dir
    paths = sorted((ROOT / "docs/tutorials").glob("*.ipynb"))
    if not paths:
        raise ValueError("No tutorial notebooks found")
    total_outputs = total_images = 0
    for path in paths:
        outputs, images = check_notebook(path, site)
        total_outputs += outputs
        total_images += images
    print(
        f"Verified {len(paths)} tutorials: {total_outputs} outputs, "
        f"{total_images} figures, and matching notebook downloads."
    )


if __name__ == "__main__":
    main()
