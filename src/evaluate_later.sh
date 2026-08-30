#!/usr/bin/env bash

# sleep 4h

steps=(50 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600)

for evaluator in evaluate_acc; do

  # python "${evaluator}.py" \
  #   --checkpoint="aklein4/horizon-v2_piano-scaled" \
  #   --checkpoint-steps "${steps[@]}" \
  #   --batch-size 12 \
  #   --aux-weight 0.1 \
  #   --compile

  python "${evaluator}.py" \
    --fresh-config="model/oloop-lora-llama3p2-1b-pre" \
    --checkpoint-steps "${steps[@]}" \
    --batch-size 12 \
    --aux-weight 0.0 \
    --base-lrs 0.0001 \
    --compile

done