#!/bin/bash
set -eu
run=${1:?Campaign directory}; label=${2:?CPU label}
source="$run/source"
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
export PYTHONPATH="$source/src" JAX_PLATFORMS=cpu
export JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_experimental_enable_fusion_autotuner=false"
for dtype in float32 float64; do
  mkdir -p "$run/cpu/$dtype"
  "$source/.venv/bin/python" "$source/benchmarks/run.py" \
    --manifest "$run/cpu-$dtype.json" --output "$run/cpu/$dtype/report.json" \
    --label "$label" --timeout 1200 --guard-cpu --resume || {
      code=$?; if [ "$code" -ne 1 ]; then exit "$code"; fi
    }
done
