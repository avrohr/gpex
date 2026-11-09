"""Check source links and rendered site files/fragments after a strict build."""

from __future__ import annotations

import argparse
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import nbformat
import yaml

ROOT = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and "name" in attrs:
            self.ids.add(attrs["name"])
        for name in ("href", "src"):
            if name in attrs:
                self.links.append(attrs[name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, default=ROOT / "site")
    parser.add_argument(
        "--source-version", help="version matching this checkout in a combined site"
    )
    args = parser.parse_args()
    site = args.site_dir.resolve()
    source_site = site / args.source_version if args.source_version else site
    errors = []
    source_count = rendered_count = 0
    site_url = urlsplit(yaml.safe_load((ROOT / "mkdocs.yml").read_text())["site_url"])
    sources = [
        *sorted(ROOT.glob("*.md")),
        ROOT / "examples/README.md",
        *sorted((ROOT / "docs").rglob("*.md")),
        *sorted((ROOT / "docs").rglob("*.ipynb")),
    ]
    for path in sources:
        if path.suffix == ".ipynb":
            source = "\n".join(
                c.source
                for c in nbformat.read(path, 4).cells
                if c.cell_type == "markdown"
            )
        else:
            source = path.read_text()
        source = re.sub(r"```.*?```", "", source, flags=re.S)
        for link in re.findall(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)", source):
            url = urlsplit(link.strip("<>"))
            if url.scheme or url.netloc:
                if url.netloc == site_url.netloc and url.path.startswith(site_url.path):
                    # Published stable links must also resolve in this checkout's docs.
                    relative = url.path.removeprefix(site_url.path).removeprefix(
                        "stable/"
                    )
                    target = source_site / unquote(relative)
                    if target.is_dir():
                        target /= "index.html"
                else:
                    match = re.match(
                        r"https://github.com/avrohr/gpex/(?:blob|tree)/main/(.*)", link
                    )
                    if not match:
                        continue
                    target = ROOT / unquote(match.group(1).split("#")[0])
            else:
                target = path.parent / unquote(url.path) if url.path else path
            source_count += 1
            if not target.exists():
                errors.append(f"{path.relative_to(ROOT)}: missing {link}")
    pages = {p.resolve(): Page(p.read_text()) for p in site.rglob("*.html")}
    if not pages:
        errors.append("No built HTML found; run mkdocs build --strict first")
    for path, page in pages.items():
        for link in page.links:
            url = urlsplit(link)
            if (url.scheme or url.netloc) and (
                url.netloc != site_url.netloc or not url.path.startswith(site_url.path)
            ):
                continue
            if url.path.startswith("/"):
                if not url.path.startswith(site_url.path):
                    errors.append(f"{path.relative_to(site)}: outside site root {link}")
                    continue
                target = site / unquote(url.path.removeprefix(site_url.path))
            else:
                target = path.parent / unquote(url.path) if url.path else path
            if target.is_dir():
                target /= "index.html"
            target = target.resolve()
            rendered_count += 1
            if not target.exists():
                errors.append(f"{path.relative_to(site)}: missing {link}")
            elif (
                url.fragment
                and target in pages
                and unquote(url.fragment) not in pages[target].ids
            ):
                errors.append(f"{path.relative_to(site)}: missing fragment {link}")
    for error in sorted(set(errors)):
        print(error)
    print(
        f"Checked {source_count} source links and {rendered_count} rendered links; "
        f"{len(set(errors))} errors."
    )
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
