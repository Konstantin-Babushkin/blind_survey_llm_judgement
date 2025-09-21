#!/bin/bash

# This script calculates normalized scores for all JSON files in a directory.

PARSED_RESULTS_DIR="/home/kb/Documents/workspace/itmo/blind/human_survey/parsed_results"
SCRIPT_PATH="/home/kb/Documents/workspace/itmo/blind/human_survey/calculate_human_survey_scores.py"

# Check if the python script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: The script $SCRIPT_PATH does not exist."
    exit 1
fi

# Loop through all json files in the directory
for json_file in "$PARSED_RESULTS_DIR"/*.json; do
    if [ -f "$json_file" ]; then
        # Construct the output CSV file path
        csv_file="${json_file%.json}.csv"

        echo "Processing $json_file..."
        # Run the python script
        python "$SCRIPT_PATH" "$json_file" "$csv_file"
        echo "Created $csv_file"
    fi
done

echo "All files processed."
