#!/bin/bash
# run_normalized_scores.sh
# Run compute_normalized_scores.py for evaluators A1–A5, B1–B5, C1–C5

for prefix in A B C; do
  for num in 1 2 3 4 5; do
    EVAL_ID="${prefix}_V${num}"
    INPUT="judge_results_flash/matched/results_matched_${EVAL_ID}/ratings_wide_by_task.csv"
    OUTPUT="judge_results_flash/matched/results_matched_${EVAL_ID}/normalized_scores_by_task_model.csv"

    if [ -f "$INPUT" ]; then
      echo "=== Processing $EVAL_ID ==="
      python compute_normalized_scores.py \
        --input "$INPUT" \
        --output "$OUTPUT"
    else
      echo "!!! Skipping $EVAL_ID (missing $INPUT)"
    fi
  done
done
