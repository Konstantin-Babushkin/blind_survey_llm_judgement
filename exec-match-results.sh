#!/bin/bash
# run_extract_all.sh
# Run extract_model_ratings.py for A_V1–A_V5, B_V1–B_V5, C_V1–C_V5

for prefix in A B C; do
  for ver in V1 V2 V3 V4 V5; do
    RESULTS="judge_results_flash/results_${prefix}_${ver}.json"
    MATCHER="judge_prompts/matcher/matcher_${prefix}_${ver}.csv"
    OUTDIR="judge_results_flash/matched"

    if [ -f "$RESULTS" ] && [ -f "$MATCHER" ]; then
      echo "=== Processing $RESULTS with $MATCHER ==="
      python extract_model_ratings.py --results "$RESULTS" --matcher "$MATCHER" --outdir "$OUTDIR"
    else
      echo "!!! Skipping ${prefix}_${ver} (missing file)"
    fi
  done
done
