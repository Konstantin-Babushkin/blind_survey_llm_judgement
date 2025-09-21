import json
import os
from collections import defaultdict

def process_survey_results(file_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            _id = data.get("_id")
            task_id = data.get("fields", {}).get("taskId")

            if not _id or not task_id:
                continue

            aggregated_responses = defaultdict(lambda: defaultdict(list))
            responses = data.get("fields", {}).get("responses", {})
            for key, models in responses.items():
                if "_" not in key:
                    continue
                t_id, metric = key.split('_', 1)
                if not isinstance(models, list):
                    continue
                for model in models:
                    aggregated_responses[t_id][metric].append(model)

            output_filename = os.path.join(output_dir, f"{task_id}_{_id}.json")
            with open(output_filename, 'w') as out_f:
                json.dump(aggregated_responses, out_f, indent=4)

if __name__ == "__main__":
    process_survey_results('/home/kb/Documents/workspace/yt/survey_results.jsonl', '/home/kb/Documents/workspace/yt/parsed_results')