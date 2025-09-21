import json
import sys
import os
import pandas as pd

def calculate_normalized_scores(input_file_path, output_file_path):
    """
    Calculates normalized scores from a JSON file, pivots the data,
    and writes the results to a single CSV file.
    """
    try:
        with open(input_file_path, 'r') as f:
            data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing {input_file_path}: {e}", file=sys.stderr)
        return

    category_mapping_file = os.path.join(os.path.dirname(__file__), '../', 'task_category_mapping.json')
    try:
        with open(category_mapping_file, 'r') as f:
            category_mapping = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load category mapping from {category_mapping_file}. 'category' column will be empty.", file=sys.stderr)
        category_mapping = {}

    evaluator_id = os.path.splitext(os.path.basename(input_file_path))[0]

    long_format_data = []
    for task, questions in data.items():
        category = category_mapping.get(task)
        if not category:
            continue

        for question, contestants in questions.items():
            n = len(contestants)
            if n > 1:
                for i, contestant in enumerate(contestants):
                    rank = i + 1
                    normalized_score = 1 - (rank - 1) / (n - 1)
                    long_format_data.append({
                        'evaluator': evaluator_id,
                        'task_id': task,
                        'task_type': category,
                        'question': question,
                        'model_name': contestant,
                        'normalized_score': normalized_score
                    })
            elif n == 1:
                contestant = contestants[0]
                rank = 1
                normalized_score = 1.0
                long_format_data.append({
                    'evaluator': evaluator_id,
                    'task_id': task,
                    'task_type': category,
                    'question': question,
                    'model_name': contestant,
                    'normalized_score': normalized_score
                })

    if not long_format_data:
        print("No data to process for this file.", file=sys.stderr)
        return

    df = pd.DataFrame(long_format_data)

    pivot_df = df.pivot_table(
        index=['evaluator', 'task_id', 'task_type', 'model_name'],
        columns='question',
        values='normalized_score'
    ).reset_index()

    pivot_df.rename(columns={
        'factual_accuracy': 'Accuracy',
        'fluency': 'Fluency',
        'coherence': 'Coherence',
        'conciseness': 'Conciseness',
        'constructiveness': 'Constructiveness',
        'final': 'Final',
        'motivational': 'Motivational',
        'sentiment': 'Sentiment'
    }, inplace=True)

    desired_columns = ['task_id', 'task_type', 'model_name', 'evaluator', 'Fluency', 'Coherence', 'Conciseness', 'Accuracy', 'Constructiveness', 'Final', 'Motivational', 'Sentiment']
    for col in desired_columns:
        if col not in pivot_df.columns:
            pivot_df[col] = None

    pivot_df = pivot_df[desired_columns]

    pivot_df.to_csv(output_file_path, index=False)
    print(f"Pivoted scores saved to {output_file_path}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python calculate_human_survey_scores.py <input_json_file> <output_csv_file>", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    calculate_normalized_scores(input_file, output_file)
