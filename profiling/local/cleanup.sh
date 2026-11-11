#!/bin/bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
name=${1:-$(git rev-parse --short=7 HEAD)}
if [ "$#" -gt 1 ] || [[ "$name" == */* || "$name" == .* ]]; then echo 'Usage: bash profiling/local/cleanup.sh [run-name]' >&2; exit 1; fi
processes=$(ps -axo comm=,command=)
if printf '%s\n' "$processes" | awk -v root="$repo/profiling/" '$1 !~ /awk$/ && index($0,root) && /cpu.sh|benchmarks\/run.py/ {found=1} END {exit !found}'; then
    echo 'Stop CPU profiling controllers and workers before cleanup.' >&2; exit 1
fi
run="$repo/profiling/raw/$name"
root="$repo/profiling/.local"
test ! -L "$repo/profiling/raw" && test ! -L "$run" && test ! -L "$root"
test ! -e "$run" || test -f "$run/plan.json"
test ! -e "$root" || test "$(cat "$root/.gpex-cpu")" = 'gpex CPU profiling'
if [ "$(uname -s)" = Darwin ] && [ -f "$run/cpu-launchd-label" ]; then
    job=$(cat "$run/cpu-launchd-label")
    target="gui/$(id -u)/$job"
    if launchctl print "$target" >/dev/null 2>&1; then
        launchctl bootout "$target"
    fi
fi
rm -rf -- "$run" "$root"
