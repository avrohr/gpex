#!/bin/bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
export PATH="$HOME/.local/bin:$PATH"
if [ "$#" -ne 0 ]; then echo 'Usage: bash profiling/local/install.sh' >&2; exit 1; fi
if [ -n "$(git status --porcelain -- src pyproject.toml uv.lock)" ]; then
    echo 'Commit or stash source/dependency changes before installing.' >&2
    exit 1
fi
command -v uv >/dev/null || { echo 'Install uv before running this script.' >&2; exit 1; }
uv python install 3.12
revision=$(git rev-parse HEAD)
root="$repo/profiling/.local"
runtime="$root/runtime/${revision:0:7}"
test ! -L "$root"
test ! -e "$root" || test -f "$root/.gpex-cpu"
if [ -e "$runtime" ]; then
    echo "Runtime already exists: $runtime. Reuse it or clean it before reinstalling." >&2
    exit 1
fi
mkdir -p "$runtime"
printf 'gpex CPU profiling\n' > "$root/.gpex-cpu"
git archive "$revision" src pyproject.toml uv.lock README.md LICENSE | tar -xf - -C "$runtime"
printf '%s\n' "$revision" > "$runtime/source-commit"
cd "$runtime"
export UV_CACHE_DIR="$root/cache"
uv sync --locked --no-default-groups --group comparison --python 3.12
printf 'Installed %s at %s\n' "$revision" "$runtime"
