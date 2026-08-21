#!/usr/bin/env bash

# Run the standardized long-context evaluation matrix. A failed run is printed
# to stdout and recorded in its log, then the launcher continues with the next
# model/auxiliary-weight combination.

set -u
set -o pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/ubuntu/iTTT/src/.venv/bin/python"
LOG_DIR="$REPO_DIR/launch_logs/long-context-testing"
RESULT_DIR="$REPO_DIR/local_data/standard_long_context_results"
RULER_DIR="$REPO_DIR/local_data/ruler_e2e"
EVALUATOR="$REPO_DIR/src/evaluate_standard_long_context.py"
RULER_NUM_SAMPLES="${RULER_NUM_SAMPLES:-64}"
ADAPTATION_BATCH_SIZE="${ADAPTATION_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"

mkdir -p "$LOG_DIR" "$RESULT_DIR"
cd "$REPO_DIR" || { return 1 2>/dev/null || exit 1; }

run_evaluation() {
    local label="$1"
    local output="$2"
    local log="$3"
    shift 3

    if [[ -s "$output" && "${FORCE:-0}" != "1" ]]; then
        echo "SKIP: $label (existing result: $output)"
        return 0
    fi

    echo "START: $label"
    echo "COMMAND: $*"
    "$@" 2>&1 | tee "$log"
    local command_status=${PIPESTATUS[0]}

    if (( command_status != 0 )); then
        echo "FAIL: $label (exit $command_status; log: $log)"
        return 0
    fi

    echo "DONE: $label (result: $output; log: $log)"
}

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "FATAL: Python environment is unavailable: $PYTHON_BIN"
    return 1 2>/dev/null || exit 1
fi

if ! "$PYTHON_BIN" -c "import datasets, torch" >/dev/null 2>&1; then
    echo "FATAL: required Python imports failed in $PYTHON_BIN"
    "$PYTHON_BIN" -c "import datasets, torch" 2>&1
    return 1 2>/dev/null || exit 1
fi

if [[ ! "$RULER_NUM_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
    echo "FATAL: RULER_NUM_SAMPLES must be a positive integer; got $RULER_NUM_SAMPLES"
    return 1 2>/dev/null || exit 1
fi

RULER_LENGTHS=(8192 16384 32768 65536 131072)
RULER_TASKS=(niah_single_1 niah_single_2 niah_single_3)
RULER_MANIFESTS=()
for length in "${RULER_LENGTHS[@]}"; do
    for task in "${RULER_TASKS[@]}"; do
        manifest="$RULER_DIR/$length/$task/validation.jsonl"
        if [[ ! -f "$manifest" ]]; then
            echo "FATAL: missing RULER manifest: $manifest"
            return 1 2>/dev/null || exit 1
        fi
        manifest_samples=$(wc -l < "$manifest")
        if (( manifest_samples < RULER_NUM_SAMPLES )); then
            echo "FATAL: $manifest has $manifest_samples samples; launcher requests $RULER_NUM_SAMPLES"
            return 1 2>/dev/null || exit 1
        fi
        RULER_MANIFESTS+=("$manifest")
    done
done
echo "RULER preflight: ${#RULER_MANIFESTS[@]} manifests, $RULER_NUM_SAMPLES samples per task/length"

MODELS=(
    "oloop-lora-llama3p2-1b-pre"
    "piano-llama3p2-1b-pre"
)
AUX_WEIGHTS=("0.1" "1.0")

# for model in "${MODELS[@]}"; do
#     for aux_weight in "${AUX_WEIGHTS[@]}"; do
#         run_name="${model}_aux-${aux_weight}_quality-full"
#         output="$RESULT_DIR/${run_name}.json"
#         log="$LOG_DIR/${run_name}.log"
#         run_evaluation \
#             "$run_name" "$output" "$log" \
#             "$PYTHON_BIN" "$EVALUATOR" \
#             --fresh-config "model/${model}.yaml" \
#             --aux-weight "$aux_weight" \
#             --benchmark quality \
#             --seed 42 \
#             --quality-article-batch-size "$ADAPTATION_BATCH_SIZE" \
#             --eval-batch-size "$EVAL_BATCH_SIZE" \
#             --output "$output"
#     done
# done

for model in "${MODELS[@]}"; do
    for aux_weight in "${AUX_WEIGHTS[@]}"; do
        run_name="${model}_aux-${aux_weight}_ruler-e2e-grid-n${RULER_NUM_SAMPLES}"
        output="$RESULT_DIR/${run_name}.json"
        log="$LOG_DIR/${run_name}.log"
        run_evaluation \
            "$run_name" "$output" "$log" \
            "$PYTHON_BIN" "$EVALUATOR" \
            --fresh-config "model/${model}.yaml" \
            --aux-weight "$aux_weight" \
            --benchmark ruler \
            --ruler-manifests "${RULER_MANIFESTS[@]}" \
            --ruler-num-samples "$RULER_NUM_SAMPLES" \
            --ruler-adaptation-batch-size "$ADAPTATION_BATCH_SIZE" \
            --seed 42 \
            --eval-batch-size "$EVAL_BATCH_SIZE" \
            --output "$output"
    done
done

echo "All scheduled evaluations have been attempted."
