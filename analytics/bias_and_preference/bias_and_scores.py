import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict

MODELS = ["Gemini", "DeepSeek", "Flash", "Qwen"]

CRITERIA = [
    "Fluency",
    "Coherence",
    "Conciseness",
    "Accuracy",
    "Constructiveness",
    "Final choice",
    "Motivational tone",
    "Sentiment match"
]

MODEL_NAME_MAPPING = {
    'deepseek': 'DeepSeek',
    'flash': 'Flash',
    'gemini': 'Gemini',
    'qwen': 'Qwen'
}


def score_to_rank(score: float) -> int:
    if score == 1.0:
        return 4
    elif abs(score - 0.6666666667) < 0.01:
        return 3
    elif abs(score - 0.3333333333) < 0.01:
        return 2
    elif score == 0.0:
        return 1
    return 0


def rank_to_score(rank: int) -> float:
    if rank == 1:
        return 4.0
    elif rank == 2:
        return 3.0
    elif rank == 3:
        return 2.0
    elif rank == 4:
        return 1.0
    return 0.0


def load_form_data(form_file: Path) -> Dict[str, Dict[str, str]]:
    task_texts = {}

    with open(form_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = row.get('Task ID', '').strip()
            if not task_id:
                continue

            task_texts[task_id] = {}
            for model in MODELS:
                text = row.get(model, '').strip()
                if text:
                    task_texts[task_id][model] = text

    return task_texts


def load_rankings_data(rankings_file: Path) -> List[Dict[str, Any]]:
    rankings = []

    with open(rankings_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rankings.append(row)

    return rankings


def get_model_name_from_filename(filename: str) -> str:
    base_name = Path(filename).stem
    model_name_lower = base_name.split('_')[0].lower()
    return MODEL_NAME_MAPPING.get(model_name_lower, model_name_lower.capitalize())


def extract_ranking_questions(rankings: List[Dict], evaluator_model: str) -> List[Tuple[str, str, List[str]]]:
    questions = []

    grouped = defaultdict(lambda: defaultdict(dict))

    for row in rankings:
        if row['model_name'] != evaluator_model:
            continue

        task_id = row['task_id']
        evaluator = row['evaluator']

        for criterion in CRITERIA:
            score_str = row.get(criterion, '')
            if not score_str:
                continue

            try:
                score = float(score_str)
                grouped[(task_id, evaluator)][criterion][evaluator_model] = score
            except (ValueError, TypeError):
                continue

    for (task_id, evaluator), criteria_scores in grouped.items():
        all_models_scores = defaultdict(dict)

        for row in rankings:
            if row['task_id'] == task_id and row['evaluator'] == evaluator:
                model = row['model_name']
                for criterion in CRITERIA:
                    score_str = row.get(criterion, '')
                    if score_str:
                        try:
                            score = float(score_str)
                            all_models_scores[criterion][model] = score
                        except (ValueError, TypeError):
                            continue

        for criterion in CRITERIA:
            if criterion not in all_models_scores:
                continue

            model_scores = all_models_scores[criterion]
            if len(model_scores) != 4:
                continue

            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            ranking = [model for model, _ in sorted_models]

            questions.append((task_id, criterion, ranking))

    return questions


def calculate_model_bias(ranking_questions: List[Tuple[str, str, List[str]]]) -> Dict[str, float]:
    model_scores = {model: [] for model in MODELS}

    for _, _, ranking in ranking_questions:
        for i, model in enumerate(ranking):
            rank = i + 1
            score = rank_to_score(rank)
            model_scores[model].append(score)

    return {
        model: sum(scores) / len(scores) if scores else 0.0
        for model, scores in model_scores.items()
    }


def calculate_monotonicity_score(ranking_questions: List[Tuple[str, str, List[str]]]) -> float:
    if len(ranking_questions) < 2:
        return 0.0

    try:
        from scipy import stats
    except ImportError:
        return 0.0

    sequence_correlations = []

    task_rankings = defaultdict(list)
    for task_id, criterion, ranking in ranking_questions:
        task_rankings[task_id].append((criterion, ranking))

    task_ids = sorted(task_rankings.keys())
    for i in range(len(task_ids) - 1):
        task1 = task_ids[i]
        task2 = task_ids[i + 1]

        if task_rankings[task1] and task_rankings[task2]:
            _, ranking1 = task_rankings[task1][0]
            _, ranking2 = task_rankings[task2][0]

            positions1 = [ranking1.index(m) for m in MODELS if m in ranking1]
            positions2 = [ranking2.index(m) for m in MODELS if m in ranking2]

            if len(positions1) == len(positions2) == 4:
                corr, _ = stats.spearmanr(positions1, positions2)
                if not (corr != corr):
                    sequence_correlations.append(corr)

    if sequence_correlations:
        return float(sum(sequence_correlations) / len(sequence_correlations))

    return 0.0


def calculate_variance_score(ranking_questions: List[Tuple[str, str, List[str]]]) -> float:
    all_scores = []

    for _, _, ranking in ranking_questions:
        for i, model in enumerate(ranking):
            rank = i + 1
            score = rank_to_score(rank)
            all_scores.append(score)

    if not all_scores:
        return 0.0

    mean_score = sum(all_scores) / len(all_scores)
    variance = sum((x - mean_score) ** 2 for x in all_scores) / len(all_scores)

    return float(variance)


def calculate_verbosity_scores(model_lengths: Dict[str, int]) -> Dict[str, float]:
    if not model_lengths:
        return {}

    sorted_models = sorted(model_lengths.items(), key=lambda x: x[1])
    n = len(sorted_models)

    if n == 1:
        return {sorted_models[0][0]: 0.5}

    scores = {}
    for i, (model, _) in enumerate(sorted_models):
        scores[model] = i / (n - 1) if n > 1 else 0.0

    return scores


def calculate_verbosity_bias(
        ranking_questions: List[Tuple[str, str, List[str]]],
        form_data: Dict[str, Dict[str, str]]
) -> Dict[str, float]:
    first_place_verbosities = []
    weighted_verbosities = []
    all_lengths = []
    all_ranks = []

    for task_id, criterion, ranking in ranking_questions:
        if task_id not in form_data:
            continue

        model_lengths = {model: len(text) for model, text in form_data[task_id].items()}
        verbosity_scores = calculate_verbosity_scores(model_lengths)

        if not verbosity_scores:
            continue

        first_model = ranking[0]
        if first_model in verbosity_scores:
            first_place_verbosities.append(verbosity_scores[first_model])

        weights = [4, 3, 2, 1]
        weighted_sum = 0
        weight_total = 0
        for i, model in enumerate(ranking):
            if model in verbosity_scores:
                weighted_sum += verbosity_scores[model] * weights[i]
                weight_total += weights[i]

        if weight_total > 0:
            weighted_verbosities.append(weighted_sum / weight_total)

        for i, model in enumerate(ranking):
            if model in model_lengths:
                all_lengths.append(model_lengths[model])
                all_ranks.append(i + 1)

    correlation = 0.0
    if len(all_lengths) >= 4:
        try:
            from scipy import stats
            corr, _ = stats.spearmanr(all_lengths, all_ranks)
            if not (corr != corr):
                correlation = -corr
        except ImportError:
            pass

    return {
        'verbosity_first': sum(first_place_verbosities) / len(
            first_place_verbosities) if first_place_verbosities else 0.5,
        'verbosity_weighted': sum(weighted_verbosities) / len(weighted_verbosities) if weighted_verbosities else 0.5,
        'verbosity_correlation': correlation
    }


def analyze_llm_evaluator(
        rankings_file: Path,
        form_files: List[Path]
) -> Dict[str, Any]:
    evaluator_model = get_model_name_from_filename(rankings_file.name)
    rankings = load_rankings_data(rankings_file)
    ranking_questions = extract_ranking_questions(rankings, evaluator_model)

    all_form_data = {}
    for form_file in form_files:
        form_data = load_form_data(form_file)
        all_form_data.update(form_data)

    result = {
        'evaluator_model': evaluator_model,
        'rankings_file': rankings_file.name,
        'num_questions': len(ranking_questions),
    }

    if not ranking_questions:
        return result

    model_bias = calculate_model_bias(ranking_questions)
    for model, avg_score in model_bias.items():
        short_name = model.replace('-', '_').replace('.', '_')
        result[f'model_bias_{short_name}'] = round(avg_score, 3)

    result['monotonicity_score'] = round(calculate_monotonicity_score(ranking_questions), 3)
    result['variance_score'] = round(calculate_variance_score(ranking_questions), 3)
    verbosity_metrics = calculate_verbosity_bias(ranking_questions, all_form_data)
    result['verbosity_first'] = round(verbosity_metrics['verbosity_first'], 3)
    result['verbosity_weighted'] = round(verbosity_metrics['verbosity_weighted'], 3)
    result['verbosity_correlation'] = round(verbosity_metrics['verbosity_correlation'], 3)


    model_bias_values = list(model_bias.values())
    if model_bias_values:
        result['model_bias_std'] = round(
            (sum((x - sum(model_bias_values) / len(model_bias_values)) ** 2 for x in model_bias_values) / len(
                model_bias_values)) ** 0.5,
            3
        )

        max_model = max(model_bias.items(), key=lambda x: x[1])
        min_model = min(model_bias.items(), key=lambda x: x[1])
        result['preferred_model'] = max_model[0]
        result['least_preferred_model'] = min_model[0]
        result['preference_gap'] = round(max_model[1] - min_model[1], 3)

    return result


def main():
    base_path = Path(__file__).parent
    rankings_path = base_path / 'llm_self_rankings'
    forms_path = base_path / 'llm_forms'
    output_path = base_path / 'llm_bias_analysis_results.csv'

    rankings_files = sorted(rankings_path.glob('*.csv'))
    if not rankings_files:
        return

    form_files = sorted(forms_path.glob('*.csv'))

    if not form_files:
        return

    results = []
    for rankings_file in rankings_files:
        try:
            result = analyze_llm_evaluator(rankings_file, form_files)
            results.append(result)
        except Exception:
            import traceback
            traceback.print_exc()

    if results:
        fieldnames = list(results[0].keys())
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

if __name__ == '__main__':
    main()