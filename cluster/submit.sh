#!/bin/bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
export PATH="$HOME/.local/bin:$PATH"

usage() {
    echo 'Usage: bash cluster/submit.sh [run-name] [--resume] [--dry-run] [--dtype float32|float64] [--workload NAME]...' >&2
    exit 1
}
name=
resume=0
dry_run=0
filters=(--list-gpu-jobs)
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --resume) resume=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        --dtype|--workload) [[ "$#" -ge 2 ]] || usage; filters+=("$1" "$2"); shift 2 ;;
        -*) usage ;;
        *) [[ -z "$name" && "$1" != */* && "$1" != .* ]] || usage; name=$1; shift ;;
    esac
done
python=$(uv python find 3.12)
revision=$(git rev-parse HEAD)
name=${name:-${revision:0:7}}
root=$("$python" -c 'import json; from pathlib import Path; print(Path(json.load(open("profiling/plan.json"))["helma"]["root"]).expanduser())')
runtime="$root/runtime/${revision:0:7}"
run="$root/runs/$name"
if [[ "$dry_run" == 1 ]]; then
    if [[ "$resume" == 1 ]]; then filters+=(--unfinished-in "$run"); fi
    exec "$python" profiling/campaign.py "${filters[@]}"
fi

rows=$("$python" profiling/campaign.py "${filters[@]}")
test -x "$runtime/.venv/bin/python" || { echo 'Run cluster/install.sh first.' >&2; exit 1; }
for file in pyproject.toml uv.lock; do cmp "$file" "$runtime/$file"; done
test "$(cat "$runtime/source-commit")" = "$revision"
prepare=(--backend gpu --output "$run")
if [[ "$resume" == 1 ]]; then prepare+=(--resume); fi
"$python" profiling/campaign.py "${prepare[@]}"
mkdir "$run/.submitting" || { echo "Submission lock exists: $run/.submitting" >&2; exit 1; }
trap 'rmdir "$run/.submitting"' EXIT
active=$(squeue -u "$USER" -n lingpax-profile -h -o '%Z')
if grep -Fxq "$run" <<< "$active"; then
    echo "Profiling jobs are still active for $run." >&2
    exit 1
fi
if [[ ! -L "$run/source/.venv" ]]; then ln -s "$runtime/.venv" "$run/source/.venv"; fi
test "$(realpath "$run/source/.venv")" = "$(realpath "$runtime/.venv")"
if [[ "$resume" == 1 ]]; then
    rows=$("$python" profiling/campaign.py "${filters[@]}" --unfinished-in "$run")
fi
if [[ -z "$rows" ]]; then
    echo "All selected workloads are complete."
    exit 0
fi
mkdir -p "$run/logs"
while IFS=$'\t' read -r dtype group minutes workloads; do
    read -r -a selected <<< "$workloads"
    job=$(env -u _LMFILES_ -u LOADEDMODULES sbatch --parsable --time="$minutes" \
        --chdir="$run" --output="$run/logs/%j.out" \
        "$run/worker.sh" "$run" "$dtype" "$group" "${selected[@]}")
    printf '%s\t%s\t%s\t%s\n' "$dtype" "$group" "$job" "$workloads" | tee -a "$run/launch.tsv"
done <<< "$rows"
