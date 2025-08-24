#!/bin/bash
# run_all.sh
# Loop through sets of prompt/result files and run send_prompts_openrouter.py

MODEL="qwen/qwen3-235b-a22b-2507"
MAX_TOKENS=5000

for prefix in A B C; do
  for ver in V1 V2 V3 V4 V5; do
    PROMPT_FILE="judge_prompts/prompts_${prefix}_${ver}.txt"
    RESULT_FILE="judge_results/results_${prefix}_${ver}.json"

    if [ -f "$PROMPT_FILE" ]; then
      echo "=== Running $PROMPT_FILE -> $RESULT_FILE ==="
      python send_prompts_openrouter.py \
        --input "$PROMPT_FILE" \
        --output "$RESULT_FILE" \
        --model "$MODEL" \
        --start 1 \
        --max -1 \
        --max_tokens $MAX_TOKENS
    else
      echo "!!! Skipping $PROMPT_FILE (not found)"
    fi
  done
done
