#!/usr/bin/env bash

# sleep 4h

steps=(1600 1700 1800 1900 2000 2100 2200 2300)

for evaluator in evaluate_icl evaluate_acc; do

  python "${evaluator}.py" \
    --checkpoint="aklein4/horizon-v2_piano-scaled" \
    --checkpoint-steps "${steps[@]}" \
    --batch-size 12 \
    --aux-weight 0.1 \
    --compile

  # python "${evaluator}.py" \
  #   --fresh-config="model/oloop-lora-llama3p2-1b-pre" \
  #   --checkpoint-steps "${steps[@]}" \
  #   --batch-size 12 \
  #   --aux-weight 0.0 \
  #   --base-lrs 0.0001 \
  #   --compile

done