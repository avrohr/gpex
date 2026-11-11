#!/bin/bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
if [ "$#" -ne 0 ]; then echo 'Usage: bash cluster/cleanup.sh' >&2; exit 1; fi
jobs=$(squeue -h -u "$USER" -n lingpax-profile)
if [ -n "$jobs" ]; then echo 'Profiling jobs are still active.' >&2; exit 1; fi
root=$(python3 -c 'import json; from pathlib import Path; print(Path(json.load(open("profiling/plan.json"))["helma"]["root"]).expanduser())')
test ! -L "$root"
test "$(cat "$root/.gpex-profiling")" = 'gpex profiling'
rm -rf -- "$root"
