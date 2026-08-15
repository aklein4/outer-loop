#!/usr/bin/env bash

# sleep 4h

oloop_steps=(1900 1500)
forte_steps=(50 100 150 200 250 300 350 400 450)
piano_steps=(350 400)

for evaluator in evaluate_icl evaluate_acc; do

  # python "${evaluator}.py" \
  #   --checkpoint="aklein4/horizon-v2_oloop" \
  #   --checkpoint-steps "${oloop_steps[@]}" \
  #   --batch-size 12 \
  #   --aux-weight 0.0 \
  #   --compile

  python "${evaluator}.py" \
      --checkpoint="aklein4/horizon-v2_alpha" \
      --checkpoint-steps "${forte_steps[@]}" \
      --batch-size 12 \
      --aux-weight 0.1 \
      --compile

  # python "${evaluator}.py" \
  #   --checkpoint="aklein4/horizon-v2_piano" \
  #   --checkpoint-steps "${piano_steps[@]}" \
  #   --batch-size 12 \
  #   --aux-weight 0.0 \
  #   --compile

done