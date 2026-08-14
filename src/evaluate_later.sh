#!/usr/bin/env bash

sleep 4.5h

oloop_steps=(300 400 500 600 700 800 900 1000)
piano_steps=(100 150 200 250 300 350 400 450 500 550 600)

for evaluator in evaluate_icl evaluate_acc; do

  python "${evaluator}.py" \
    --checkpoint="aklein4/horizon-v2_oloop" \
    --checkpoint-steps "${oloop_steps[@]}" \
    --batch-size 12 \
    --aux-weight 0.0 \
    --compile

  python "${evaluator}.py" \
    --checkpoint="aklein4/horizon-v2_piano" \
    --checkpoint-steps "${piano_steps[@]}" \
    --batch-size 12 \
    --aux-weight 0.0 \
    --compile

done