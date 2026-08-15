#!/usr/bin/env bash

# sleep 4h

oloop_steps=(1900 1500)
piano_steps=(350 400)

for evaluator in evaluate_icl evaluate_acc; do

  python "${evaluator}.py" \
    --checkpoint="aklein4/horizon-v2_oloop" \
    --checkpoint-steps "${oloop_steps[@]}" \
    --batch-size 12 \
    --aux-weight 0.0 \
    --compile

  # python "${evaluator}.py" \
  #   --checkpoint="aklein4/horizon-v2_piano" \
  #   --checkpoint-steps "${piano_steps[@]}" \
  #   --batch-size 12 \
  #   --aux-weight 0.0 \
  #   --compile

done