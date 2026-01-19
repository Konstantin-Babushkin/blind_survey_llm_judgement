import json
import csv
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


MODELS_HUMAN = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "DeepSeek-R1-0528",
    "Qwen3-235B-A22B-2507"
]

MODELS_LLM = ["Gemini", "DeepSeek", "Flash", "Qwen"]

MODEL_MAPPING = {
    "gemini-2.5-pro": "Gemini",
    "gemini-2.5-flash": "Flash",
    "DeepSeek-R1-0528": "DeepSeek",
    "Qwen3-235B-A22B-2507": "Qwen"
}

CRITERIA_HUMAN = ["fluency", "coherence", "conciseness", "sentiment",
                  "motivational", "constructiveness", "final"]
CRITERIA_LLM = ["Fluency", "Coherence", "Conciseness", "Accuracy",
                "Constructiveness", "Final choice", "Motivational tone", "Sentiment match"]

CRITERIA_MAPPING = {
    "fluency": "Fluency",
    "coherence": "Coherence",
    "conciseness": "Conciseness",
    "sentiment": "Sentiment match",
    "motivational": "Motivational tone",
    "constructiveness": "Constructiveness",
    "final": "Final choice"
}


def load_human_responses(responses_dir: Path) -> Dict[str, Dict[str, List[str]]]:
    human_rankings = defaultdict(lambda: defaultdict(list))

    for response_file in responses_dir.glob('*.json'):
        with open(response_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        respondent_id = response_file.stem
        responses = data.get('responses', {})

        for key, value in responses.items():
            if 'sanity_check' in key or 'factual_accuracy' in key:
                continue

            if isinstance(value, list) and len(value) == 4:
                if all(model in MODELS_HUMAN for model in value):
                    
                    parts = key.split('_')
                    if len(parts) >= 2:
                        task_id = parts[0].upper()
                        criterion = '_'.join(parts[1:]).lower()
                        criterion_normalized = normalize_criterion_name(criterion)
                        normalized_ranking = [MODEL_MAPPING.get(m, m) for m in value]
                        human_rankings[f"{task_id}_{criterion_normalized}"][respondent_id] = normalized_ranking

    return human_rankings


def load_llm_rankings(rankings_dir: Path) -> Dict[str, Dict[str, List[str]]]:
    llm_rankings = defaultdict(lambda: defaultdict(list))

    
    for rankings_file in rankings_dir.glob('*.csv'):
        with open(rankings_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        grouped = defaultdict(lambda: defaultdict(dict))
        for row in rows:
            task_id = row['task_id']
            evaluator = row['evaluator']
            model = row['model_name']

            for criterion_llm in CRITERIA_LLM:
                score_str = row.get(criterion_llm, '')
                if score_str:
                    try:
                        score = float(score_str)
                        grouped[(task_id, evaluator)][criterion_llm][model] = score
                    except (ValueError, TypeError):
                        continue

        
        for (task_id, evaluator), criteria_scores in grouped.items():
            for criterion_llm, model_scores in criteria_scores.items():
                if len(model_scores) == 4:  
                    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
                    ranking = [model for model, _ in sorted_models]
                    criterion_normalized = normalize_criterion_name(criterion_llm)
                    key = f"{task_id}_{criterion_normalized}"
                    model_name_from_file = rankings_file.stem.split('_')[0]
                    evaluator_id = f"{evaluator}_{model_name_from_file}"
                    llm_rankings[key][evaluator_id] = ranking

    return llm_rankings


def ranking_to_ranks(ranking: List[str], models: List[str]) -> List[int]:
    ranks = [0] * len(models)
    for i, model in enumerate(ranking):
        if model in models:
            idx = models.index(model)
            ranks[idx] = i + 1  
    return ranks


def calculate_spearman_correlation(ranks1: List[int], ranks2: List[int]) -> float:
    n = len(ranks1)
    if n != len(ranks2) or n == 0:
        return 0.0

    
    d_squared_sum = sum((r1 - r2) ** 2 for r1, r2 in zip(ranks1, ranks2))

    
    if n <= 1:
        return 1.0 if ranks1 == ranks2 else 0.0

    correlation = 1 - (6 * d_squared_sum) / (n * (n ** 2 - 1))
    return correlation


def calculate_exact_agreement(ranking1: List[str], ranking2: List[str]) -> bool:
    return ranking1 == ranking2


def calculate_rank_distance(ranking1: List[str], ranking2: List[str], models: List[str]) -> float:
    ranks1 = ranking_to_ranks(ranking1, models)
    ranks2 = ranking_to_ranks(ranking2, models)

    if len(ranks1) != len(ranks2):
        return float('inf')

    distance = sum(abs(r1 - r2) for r1, r2 in zip(ranks1, ranks2)) / len(ranks1)
    return distance


def calculate_krippendorff_alpha(rankings_list: List[List[int]]) -> float:
    if len(rankings_list) < 2:
        return 0.0

    n_raters = len(rankings_list)
    n_items = len(rankings_list[0]) if rankings_list else 0

    if n_items == 0:
        return 0.0

    observed_disagreement = 0.0
    total_pairs = 0

    for i in range(n_items):
        for r1 in range(n_raters):
            for r2 in range(r1 + 1, n_raters):
                rank1 = rankings_list[r1][i]
                rank2 = rankings_list[r2][i]
                
                disagreement = (rank1 - rank2) ** 2
                observed_disagreement += disagreement
                total_pairs += 1

    if total_pairs == 0:
        return 0.0

    observed_disagreement /= total_pairs
    expected_disagreement = 5.0
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else 0.0

    alpha = 1 - (observed_disagreement / expected_disagreement)
    return max(0.0, min(1.0, alpha))  


def calculate_agreement_metrics(
        rankings_dict: Dict[str, List[str]],
        models: List[str]
) -> Dict[str, float]:
    if len(rankings_dict) < 2:
        return {
            'krippendorff_alpha': 0.0,
            'exact_agreement_pct': 0.0,
            'mean_spearman_correlation': 0.0,
            'mean_rank_distance': 0.0,
            'n_raters': len(rankings_dict)
        }

    rankings_ranks = []
    rankings_list = list(rankings_dict.values())

    for ranking in rankings_list:
        ranks = ranking_to_ranks(ranking, models)
        rankings_ranks.append(ranks)

    krippendorff_alpha = calculate_krippendorff_alpha(rankings_ranks)
    exact_agreements = 0
    total_pairs = 0
    for i in range(len(rankings_list)):
        for j in range(i + 1, len(rankings_list)):
            if calculate_exact_agreement(rankings_list[i], rankings_list[j]):
                exact_agreements += 1
            total_pairs += 1

    exact_agreement_pct = (exact_agreements / total_pairs * 100) if total_pairs > 0 else 0.0

    
    spearman_correlations = []
    for i in range(len(rankings_ranks)):
        for j in range(i + 1, len(rankings_ranks)):
            corr = calculate_spearman_correlation(rankings_ranks[i], rankings_ranks[j])
            spearman_correlations.append(corr)

    mean_spearman = sum(spearman_correlations) / len(spearman_correlations) if spearman_correlations else 0.0

    
    rank_distances = []
    for i in range(len(rankings_list)):
        for j in range(i + 1, len(rankings_list)):
            dist = calculate_rank_distance(rankings_list[i], rankings_list[j], models)
            rank_distances.append(dist)

    mean_rank_distance = sum(rank_distances) / len(rank_distances) if rank_distances else 0.0

    return {
        'krippendorff_alpha': krippendorff_alpha,
        'exact_agreement_pct': exact_agreement_pct,
        'mean_spearman_correlation': mean_spearman,
        'mean_rank_distance': mean_rank_distance,
        'n_raters': len(rankings_dict)
    }


def normalize_criterion_name(criterion: str) -> str:
    criterion = criterion.lower().replace(' ', '_').replace('_tone', '').replace('_match', '')
    if criterion == 'final_choice':
        criterion = 'final'
    return criterion


def main():
    """Основная функция."""
    base_path = Path(__file__).parent.parent
    human_responses_dir = base_path / 'bias' / 'responses'
    llm_rankings_dir = base_path / 'llm_bias' / 'llm_self_rankings'
    output_path = base_path / 'llm_bias' / 'inter_rater_agreement_results.csv'

    human_rankings = load_human_responses(human_responses_dir)
    llm_rankings = load_llm_rankings(llm_rankings_dir)
    all_tasks = set(human_rankings.keys()) | set(llm_rankings.keys())

    results = []

    
    for task_criterion in sorted(all_tasks):
        human_task = human_rankings.get(task_criterion, {})
        llm_task = llm_rankings.get(task_criterion, {})

        if not human_task and not llm_task:
            continue

        result = {
            'task_criterion': task_criterion,
            'n_human_raters': len(human_task),
            'n_llm_raters': len(llm_task),
        }

        
        if len(human_task) >= 2:
            human_metrics = calculate_agreement_metrics(human_task, MODELS_LLM)
            result['human_human_krippendorff_alpha'] = round(human_metrics['krippendorff_alpha'], 3)
            result['human_human_exact_agreement_pct'] = round(human_metrics['exact_agreement_pct'], 2)
            result['human_human_mean_spearman'] = round(human_metrics['mean_spearman_correlation'], 3)
            result['human_human_mean_rank_distance'] = round(human_metrics['mean_rank_distance'], 3)
        else:
            result['human_human_krippendorff_alpha'] = None
            result['human_human_exact_agreement_pct'] = None
            result['human_human_mean_spearman'] = None
            result['human_human_mean_rank_distance'] = None

        
        if len(llm_task) >= 2:
            llm_metrics = calculate_agreement_metrics(llm_task, MODELS_LLM)
            result['llm_llm_krippendorff_alpha'] = round(llm_metrics['krippendorff_alpha'], 3)
            result['llm_llm_exact_agreement_pct'] = round(llm_metrics['exact_agreement_pct'], 2)
            result['llm_llm_mean_spearman'] = round(llm_metrics['mean_spearman_correlation'], 3)
            result['llm_llm_mean_rank_distance'] = round(llm_metrics['mean_rank_distance'], 3)
        else:
            result['llm_llm_krippendorff_alpha'] = None
            result['llm_llm_exact_agreement_pct'] = None
            result['llm_llm_mean_spearman'] = None
            result['llm_llm_mean_rank_distance'] = None

        
        if human_task and llm_task:
            
            all_rankings = {**human_task, **llm_task}
            combined_metrics = calculate_agreement_metrics(all_rankings, MODELS_LLM)
            result['human_llm_krippendorff_alpha'] = round(combined_metrics['krippendorff_alpha'], 3)
            result['human_llm_exact_agreement_pct'] = round(combined_metrics['exact_agreement_pct'], 2)
            result['human_llm_mean_spearman'] = round(combined_metrics['mean_spearman_correlation'], 3)
            result['human_llm_mean_rank_distance'] = round(combined_metrics['mean_rank_distance'], 3)

            
            cross_correlations = []
            cross_distances = []
            for human_ranking in human_task.values():
                for llm_ranking in llm_task.values():
                    ranks_human = ranking_to_ranks(human_ranking, MODELS_LLM)
                    ranks_llm = ranking_to_ranks(llm_ranking, MODELS_LLM)
                    corr = calculate_spearman_correlation(ranks_human, ranks_llm)
                    dist = calculate_rank_distance(human_ranking, llm_ranking, MODELS_LLM)
                    cross_correlations.append(corr)
                    cross_distances.append(dist)

            result['human_llm_cross_mean_spearman'] = round(sum(cross_correlations) / len(cross_correlations),
                                                            3) if cross_correlations else None
            result['human_llm_cross_mean_rank_distance'] = round(sum(cross_distances) / len(cross_distances),
                                                                 3) if cross_distances else None
        else:
            result['human_llm_krippendorff_alpha'] = None
            result['human_llm_exact_agreement_pct'] = None
            result['human_llm_mean_spearman'] = None
            result['human_llm_mean_rank_distance'] = None
            result['human_llm_cross_mean_spearman'] = None
            result['human_llm_cross_mean_rank_distance'] = None

        results.append(result)

    
    if results:
        fieldnames = [
            'task_criterion', 'n_human_raters', 'n_llm_raters',
            'human_human_krippendorff_alpha', 'human_human_exact_agreement_pct',
            'human_human_mean_spearman', 'human_human_mean_rank_distance',
            'llm_llm_krippendorff_alpha', 'llm_llm_exact_agreement_pct',
            'llm_llm_mean_spearman', 'llm_llm_mean_rank_distance',
            'human_llm_krippendorff_alpha', 'human_llm_exact_agreement_pct',
            'human_llm_mean_spearman', 'human_llm_mean_rank_distance',
            'human_llm_cross_mean_spearman', 'human_llm_cross_mean_rank_distance'
        ]

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)


if __name__ == '__main__':
    main()