#!/usr/bin/env bash

# Run the standardized long-context evaluation matrix. A failed run is printed
# to stdout and recorded in its log, then the launcher continues with the next
# model/auxiliary-weight combination.

set -u
set -o pipefail

/home/ubuntu/iTTT/src/.venv/bin/python evaluate_standard_long_context.py \
    --fresh-config "model/oloop-lora-llama3p2-1b-pre.yaml" \
    --aux-weight 1.0 \
    --benchmark quality \
    --batch-size 16 \
    --compile
