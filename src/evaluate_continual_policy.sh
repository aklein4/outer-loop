#!/usr/bin/env bash

set -euo pipefail

PYTHON=/home/ubuntu/iTTT/src/.venv/bin/python
eval_points=(0 16 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256)

"$PYTHON" evaluate_continual_policy.py \
  --fresh-config=model/piano-llama3p2-1b-pre \
  --checkpoint-steps 1600 \
  --n-tasks 4 \
  --num-examples "${eval_points[@]}" \
  --batch-size 16 \
  --aux-weight 0.1 \
  --compile

"$PYTHON" evaluate_continual_policy.py \
  --fresh-config=model/oloop-lora-llama3p2-1b-pre \
  --checkpoint-steps 1600 \
  --n-tasks 4 \
  --num-examples "${eval_points[@]}" \
  --batch-size 16 \
  --aux-weight 0.0 \
  --compile

"$PYTHON" plot_continual_policy.py --n-tasks 4 --boundaries 64 128 196
