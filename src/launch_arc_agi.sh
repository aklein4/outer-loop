#!/usr/bin/env bash

set -u

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/iTTT/src/.venv/bin/python}"
ARC_AGI_1_DATA_ROOT="${ARC_AGI_1_DATA_ROOT:-local_data/ARC-AGI/data}"
ARC_AGI_2_DATA_ROOT="${ARC_AGI_2_DATA_ROOT:-local_data/ARC-AGI-2/data}"

run_eval() {
    echo "Running: $*"
    if ! "$@"; then
        echo "FAILED: $*" >&2
    fi
}

common_args=(
    --data-root "$ARC_AGI_1_DATA_ROOT" "$ARC_AGI_2_DATA_ROOT"
    --splits validation
    --ttt-steps 128
    --batch-size 8
    --aux-weight 0.1
    --compile
)

run_eval "$PYTHON_BIN" evaluate_arc_agi.py \
    --fresh-config model/piano-llama3p2-1b-pre.yaml \
    "${common_args[@]}"

for base_lr in 3e-4 1e-4 3e-5 1e-5 1e-3; do
    run_eval "$PYTHON_BIN" evaluate_arc_agi.py \
        --fresh-config model/oloop-lora-llama3p2-1b-pre.yaml \
        --model-kwargs "{\"base_lr\":$base_lr}" \
        "${common_args[@]}"
done
