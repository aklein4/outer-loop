#!/usr/bin/env bash

set -euo pipefail

PYTHON=/home/ubuntu/iTTT/src/.venv/bin/python

"$PYTHON" evaluate_lv.py \
  --fresh-config model/piano-llama3p2-1b-pre.yaml \
  --aux-weight 0.1 \
  --batch-size 32 \
  --length-levels 16k 32k 64k \
  --compile

"$PYTHON" evaluate_lv.py \
  --fresh-config model/oloop-lora-llama3p2-1b-pre.yaml \
  --aux-weight 1.0 \
  --batch-size 32 \
  --length-levels 16k 32k 64k \
  --compile
