#!/bin/bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
export PATH="$HOME/.local/bin:$PATH"
python=$(uv python find 3.12)
revision=$(git rev-parse HEAD)
name=${1:-${revision:0:7}}
if [ "$#" -gt 2 ] || [[ "$name" == */* || "$name" == .* ]] || [ "${2:---resume}" != --resume ]; then echo 'Usage: bash profiling/local/run.sh [run-name [--resume]]' >&2; exit 1; fi
run="$repo/profiling/raw/$name"
runtime="$repo/profiling/.local/runtime/${revision:0:7}"
processes=$(ps -axo comm=,command=)
if printf '%s\n' "$processes" | awk -v root="$repo/profiling/" '$1 !~ /awk$/ && index($0,root) && /cpu.sh|benchmarks\/run.py/ {found=1} END {exit !found}'; then
    echo 'A CPU profiling controller or worker is already active.' >&2; exit 1
fi
test -x "$runtime/.venv/bin/python" || { echo 'Run profiling/local/install.sh first.' >&2; exit 1; }
for file in pyproject.toml uv.lock; do cmp "$file" "$runtime/$file"; done
test "$(cat "$runtime/source-commit")" = "$revision"
"$python" profiling/campaign.py --backend cpu --output "$run" ${2:+--resume}
if [ ! -L "$run/source/.venv" ]; then ln -s "$runtime/.venv" "$run/source/.venv"; fi
label=$("$python" -c 'import json; print(json.load(open("profiling/plan.json"))["cpu_label"])')
if [ "$(uname -s)" = Darwin ]; then
    "$python" profiling/local/launch.py "$run" "$runtime/.venv/bin/python" "$label"
else
    nohup bash "$run/cpu.sh" "$run" "$label" </dev/null >>"$run/cpu.log" 2>&1 &
    printf '%s\n' "$!" > "$run/cpu.pid"
    printf 'CPU controller %s, campaign %s\n' "$!" "$run"
fi
