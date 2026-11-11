#!/bin/bash -l
#SBATCH --job-name=lingpax-profile
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:USR1@90
set -euo pipefail
run=${1:?Campaign directory}; dtype=${2:?Precision}; group=${3:?Job group}
shift 3
test "$#" -gt 0
work=$(mktemp -d "${TMPDIR:?Slurm node-local scratch}/gpex-profile.XXXXXX")
results="$run/gpu/$dtype"
logs="$run/logs/$SLURM_JOB_ID"
mkdir -p "$work/results" "$work/cases" "$results" "$logs"
module load gpu-env/2025
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32
# CPU enables runtime error callbacks; workers still require GPU computation.
export JAX_PLATFORMS=cuda,cpu JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_experimental_enable_fusion_autotuner=false
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONUSERBASE
module list 2> "$logs/modules.txt"
findmnt -T "$work" > "$logs/staging-filesystem.txt"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total --format=csv --loop=1 > "$work/gpu.csv" &
telemetry=$!
finish() {
    kill "$telemetry" 2>/dev/null || true
    wait "$telemetry" 2>/dev/null || true
    cp "$work/gpu.csv" "$logs/gpu.csv"
}
trap finish EXIT

active_pid=
workload=
publish() {
    local components="${workload}_components"
    if [[ -d "$work/results/$components" ]]; then
        rsync -a --checksum --exclude='*.tmp' "$work/results/$components/" "$results/$components/" || return
    fi
    if [[ -f "$work/results/$workload.json" ]]; then
        rsync -a --ignore-times "$work/results/$workload.json" "$results/"
    fi
}
interrupt() {
    trap '' USR1 TERM INT
    if [[ -n "$active_pid" ]]; then
        kill -TERM -- "-$active_pid" 2>/dev/null || true
        wait "$active_pid" || true
        publish || echo "Could not publish interrupted $workload" >&2
        printf '%s\tinterrupted\n' "$workload" >> "$logs/workloads.tsv"
    fi
    exit 1
}
trap interrupt USR1 TERM INT

bash "$run/stage.sh" "$run/source" "$work/staged" "$logs/staging.tsv"
source="$work/staged/source"
python="$source/.venv/bin/python"
export PYTHONPATH="$source/src"
export GPEX_BENCHMARK_SCRATCH="$work/cases" GPEX_BENCHMARK_OUTPUT_ROOT="$results"
export GPEX_BENCHMARK_RUNTIME_SHA256
GPEX_BENCHMARK_RUNTIME_SHA256=$(tail -n +2 "$logs/staging.tsv" | cut -f 1,3 | sha256sum | cut -d ' ' -f 1)
printf '%s\t%s\t%s\n' "$dtype" "$group" "$*" > "$logs/job.tsv"
cd "$source"
failed=0
for workload; do
    components="${workload}_components"
    if [[ -d "$results/$components" ]] &&
        ! rsync -a "$results/$components/" "$work/results/$components/"; then
        printf '%s\tresume-copy-failed\n' "$workload" >> "$logs/workloads.tsv"
        failed=1
        continue
    fi
    printf '%s\tstarted\n' "$workload" >> "$logs/workloads.tsv"
    setsid "$python" "$source/benchmarks/run.py" \
        --manifest "$run/gpu-$dtype-$workload.json" \
        --output "$work/results/$workload.json" --label 'HELMA H100' --timeout 1200 --resume &
    active_pid=$!
    status=0
    wait "$active_pid" || status=$?
    if ! publish; then
        printf '%s\tpublish-failed\n' "$workload" >> "$logs/workloads.tsv"
        failed=1
    fi
    active_pid=
    printf '%s\texit\t%s\n' "$workload" "$status" >> "$logs/workloads.tsv"
    if [[ "$status" != 0 ]]; then failed=1; fi
done
exit "$failed"
