#!/bin/bash

# Script to run sanity check evaluations for all relevant result files.
# It finds files matching 'results_[A-C]_V[1-5].json' in the specified directory
# and runs the evaluation script for each, saving results to an output directory.
#
# Usage: ./run_all_evaluations.sh <results_directory> <output_directory>

# --- Configuration ---
RESULTS_DIR=$1
OUT_DIR=$2
SCRIPT_PATH="./evaluate_sanity_checks.py"

# --- Pre-flight Checks ---
if [ -z "$RESULTS_DIR" ] || [ -z "$OUT_DIR" ]; then
  echo "Usage: $0 <results_directory> <output_directory>"
  echo "Example: $0 judge_results/ all_sanity_scores/"
  exit 1
fi

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory not found at '$RESULTS_DIR'"
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Evaluation script not found at '$SCRIPT_PATH'"
    echo "Please ensure you are running this script from the project root directory."
    exit 1
fi

# --- Main Loop ---
# Create the output directory if it doesn't exist
mkdir -p "$OUT_DIR"

echo "Starting batch evaluation..."
echo "Results will be saved in: $OUT_DIR"

for f in "$RESULTS_DIR"/results_[A-C]_V[1-5].json; do
  if [ -e "$f" ]; then
    echo "----------------------------------------"
    echo "Processing file: $f"
    python3 "$SCRIPT_PATH" --results "$f" --outdir "$OUT_DIR"
    echo ""
  fi
done

echo "----------------------------------------"
echo "All evaluations complete."
