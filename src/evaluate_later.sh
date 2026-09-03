#!/usr/bin/env bash

# sleep 4h

set -uo pipefail

PYTHON=/home/ubuntu/iTTT/src/.venv/bin/python

steps=(50 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500)

for evaluator in evaluate_policy; do

  "$PYTHON" "${evaluator}.py" \
    --checkpoint="aklein4/horizon-v2_piano-scaled" \
    --checkpoint-steps "${steps[@]}" \
    --batch-size 16 \
    --aux-weight 0.1 \
    --compile

  "$PYTHON" "${evaluator}.py" \
    --fresh-config="model/oloop-lora-llama3p2-1b-pre" \
    --checkpoint-steps "${steps[@]}" \
    --batch-size 16 \
    --aux-weight 0.0 \
    --compile

done