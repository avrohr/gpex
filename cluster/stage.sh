#!/bin/bash
set -euo pipefail
original=$1
destination=$2
receipt=$3
mkdir "$destination"
venv=$(realpath "$original/.venv")
python=$(realpath "$venv/bin/python")
cpython=$(dirname "$(dirname "$python")")

printf 'tree\tarchive_bytes\tsha256\tcopy_seconds\textract_seconds\tverify_seconds\n' > "$receipt"
archive_args=(--sort=name --format=gnu --numeric-owner --owner=0 --group=0)
export TIMEFORMAT='%R'
stage_tree() {
    local name=$1 tree=$2
    shift 2
    local archive="$destination/$name.tar"
    local copy_seconds extract_seconds verify_seconds before after
    mkdir "$destination/$name"
    copy_seconds=$({ time tar "${archive_args[@]}" "$@" -C "$tree" -cf "$archive" .; } 2>&1)
    extract_seconds=$({ time tar --no-same-owner -p -C "$destination/$name" -xf "$archive"; } 2>&1)
    verify_seconds=$({
        time {
            sha256sum "$archive" > "$destination/original.sha"
            tar "${archive_args[@]}" "$@" -C "$destination/$name" -cf - . |
                sha256sum > "$destination/staged.sha"
        }
    } 2>&1)
    read -r before _ < "$destination/original.sha"
    read -r after _ < "$destination/staged.sha"
    if [[ "$before" != "$after" ]]; then
        echo "Staged $name archive checksum mismatch" >&2
        exit 1
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$(stat -c %s "$archive")" "$before" \
        "$copy_seconds" "$extract_seconds" "$verify_seconds" >> "$receipt"
    rm "$archive" "$destination/original.sha" "$destination/staged.sha"
}

stage_tree source "$original" --exclude='./.venv'
stage_tree cpython "$cpython"
stage_tree venv "$venv"

ln -sfn "../../cpython/bin/$(basename "$python")" "$destination/venv/bin/python"
sed -i "s|^home = .*|home = $destination/cpython/bin|" "$destination/venv/pyvenv.cfg"
printf '%s\n' "$destination/source/src" > "$destination/venv/lib/python3.12/site-packages/gpex.pth"
ln -s ../venv "$destination/source/.venv"

"$destination/source/.venv/bin/python" -I - "$destination" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
site = root / "venv/lib/python3.12/site-packages"
expected = [
    Path(sys.executable).resolve(),
    Path(sys.base_prefix).resolve(),
    *(Path(path).resolve() for path in sys.path if path),
    (site / "nvidia/cu13/bin/ptxas").resolve(strict=True),
    (site / "nvidia/cu13/nvvm/libdevice/libdevice.10.bc").resolve(strict=True),
    (site / "jax_plugins/xla_cuda13/xla_cuda_plugin.so").resolve(strict=True),
]
if any(not path.is_relative_to(root) for path in expected):
    raise ValueError(f"Python or CUDA tooling resolved outside staged runtime: {expected}")
PY
