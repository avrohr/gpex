#!/bin/bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
if [ "$#" -ne 0 ]; then echo 'Usage: bash cluster/install.sh' >&2; exit 1; fi
if [ -n "$(git status --porcelain -- src pyproject.toml uv.lock)" ]; then
    echo 'Commit or stash source/dependency changes before installing.' >&2
    exit 1
fi
export PATH="$HOME/.local/bin:$PATH"
export UV_BIN
UV_BIN=$(command -v uv) || { echo 'Install uv before running this script.' >&2; exit 1; }
uv python install 3.12
python=$(uv python find --managed-python 3.12)
root=$("$python" -c 'import json; from pathlib import Path; print(Path(json.load(open("profiling/plan.json"))["helma"]["root"]).expanduser())')
revision=$(git rev-parse HEAD)
runtime="$root/runtime/${revision:0:7}"
test ! -L "$root"
test ! -e "$root" || test -f "$root/.gpex-profiling"
if [ -e "$runtime" ]; then
    echo "Runtime already exists: $runtime. Reuse it or clean it before reinstalling." >&2
    exit 1
fi
mkdir -p "$runtime"
printf 'gpex profiling\n' > "$root/.gpex-profiling"
git archive "$revision" src pyproject.toml uv.lock README.md LICENSE | tar -xf - -C "$runtime"
printf '%s\n' "$revision" > "$runtime/source-commit"
cd "$runtime"
export UV_CACHE_DIR="$root/cache"
bash -lc 'set -eu; module load gpu-env/2025; exec "$UV_BIN" sync --locked --no-default-groups --group comparison --extra cuda13 --managed-python --python 3.12'
printf 'Installed %s at %s\n' "$revision" "$runtime"
