#!/usr/bin/env bash

# sleep 4h

steps=(1500)

for evaluator in evaluate_persona; do

  # python "${evaluator}.py" \
  #   --checkpoint="aklein4/horizon-v2_piano-scaled" \
  #   --checkpoint-steps "${steps[@]}" \
  #   --batch-size 16 \
  #   --aux-weight 0.1 \
  #   --compile

  python "${evaluator}.py" \
    --fresh-config="model/oloop-lora-llama3p2-1b-pre" \
    --checkpoint-steps "${steps[@]}" \
    --batch-size 16 \
    --aux-weight 0.1 \
    --compile \

done