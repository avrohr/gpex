"""Build one documentation version on the local gh-pages branch.

Run with ``dev`` or the stable package version from a release tag checkout.
The caller configures Git, checks the generated site, and pushes the branch.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tomllib
from pathlib import Path

from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="dev or a stable package version, e.g. 0.1.0")
    args = parser.parse_args()
    if args.version != "dev":
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())
        package_version = project["project"]["version"]
        version = Version(args.version)
        if args.version != package_version or version.is_prerelease or version.local:
            parser.error(
                "release docs must match the checked-out stable package version"
            )

    title = "dev (unreleased)" if args.version == "dev" else args.version
    subprocess.run(
        ["mike", "deploy", "--title", title, args.version], cwd=ROOT, check=True
    )
    versions = json.loads(
        subprocess.check_output(["mike", "list", "--json"], cwd=ROOT, text=True)
    )
    releases = [entry["version"] for entry in versions if entry["version"] != "dev"]
    if releases:
        # Publishing an older maintenance release must not move stable backwards.
        latest = max(releases, key=Version)
        subprocess.run(
            ["mike", "alias", "--update-aliases", latest, "stable"],
            cwd=ROOT,
            check=True,
        )
    subprocess.run(
        ["mike", "set-default", "stable" if releases else "dev"],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
