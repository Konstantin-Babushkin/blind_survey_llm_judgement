import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np
from collections import defaultdict
import warnings
import os
import sys

warnings.filterwarnings('ignore')

try:
    nltk.download('punkt', quiet=True)
except:
    pass


# -------------------- UTILS --------------------
def _to_text(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x)


def safe_split_lower(s: str):
    return _to_text(s).lower().split()


def lexical_diversity(tokens):
    return (len(set(tokens)) / len(tokens)) if tokens else 0.0


def distinct_n(tokens, n=1):
    if not tokens or len(tokens) < n:
        return 0.0
    ngrams = set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return len(ngrams) / (len(tokens) - n + 1)


# -------------------- METRICS --------------------
def calculate_metrics(reference, candidate):
    metrics = {}
    ref = _to_text(reference)
    cand = _to_text(candidate)

    ref_tokens = safe_split_lower(ref)
    cand_tokens = safe_split_lower(cand)

    metrics['word_count'] = len(cand.split())
    metrics['token_count'] = len(cand_tokens)

    smoothing = SmoothingFunction().method1
    try:
        metrics['bleu_1'] = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        metrics['bleu_2'] = sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5, 0, 0),
                                          smoothing_function=smoothing)
        metrics['bleu_3'] = sentence_bleu([ref_tokens], cand_tokens, weights=(1 / 3, 1 / 3, 1 / 3, 0),
                                          smoothing_function=smoothing)
        metrics['bleu_4'] = sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                                          smoothing_function=smoothing)
    except ZeroDivisionError:
        for n in range(1, 5):
            metrics[f'bleu_{n}'] = 0.0

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(ref, cand)
    metrics['rouge_l_precision'] = rouge_scores['rougeL'].precision
    metrics['rouge_l_recall'] = rouge_scores['rougeL'].recall
    metrics['rouge_l_f1'] = rouge_scores['rougeL'].fmeasure

    metrics['lexical_diversity'] = lexical_diversity(cand_tokens)
    metrics['length_ratio'] = len(cand_tokens) / (len(ref_tokens) + 1e-8)
    metrics['distinct_1'] = distinct_n(cand_tokens, 1)
    metrics['distinct_2'] = distinct_n(cand_tokens, 2)

    return metrics


# -------------------- ANALYSIS --------------------
def analyze_csv(file_path):
    df = pd.read_csv(file_path)
    base = os.path.splitext(os.path.basename(file_path))[0]

    exclude_cols = {'Original', 'Category', 'Volunteer', 'Task ID'}
    model_columns = [c for c in df.columns if c not in exclude_cols]

    all_results = defaultdict(lambda: defaultdict(list))

    for idx, row in df.iterrows():
        category = _to_text(row.get('Category'))
        reference = _to_text(row.get('Original'))

        for model in model_columns:
            candidate = _to_text(row.get(model))
            metrics = calculate_metrics(reference, candidate)
            for k, v in metrics.items():
                all_results[model][k].append(v)
            for meta in ['Category', 'Volunteer', 'Task ID']:
                all_results[model].setdefault(meta, []).append(_to_text(row.get(meta)))
            all_results[model].setdefault('Model', []).append(model)

    # BERTScore
    print("\nCalculating BERT scores (en)...")
    for model in model_columns:
        references = df['Original'].astype(str).tolist()
        candidates = df[model].astype(str).tolist()
        P, R, F1 = bert_score(candidates, references, lang='en', verbose=False)
        all_results[model]['bert_precision'] = [float(x) for x in P]
        all_results[model]['bert_recall'] = [float(x) for x in R]
        all_results[model]['bert_f1'] = [float(x) for x in F1]

    detailed_rows = []
    n = len(df)
    for model in model_columns:
        mres = all_results[model]
        for i in range(n):
            detailed_rows.append({
                'Model': mres['Model'][i],
                'Category': mres['Category'][i],
                'Volunteer': mres['Volunteer'][i],
                'Task_ID': mres['Task ID'][i],
                'word_count': mres['word_count'][i],
                'token_count': mres['token_count'][i],
                'bleu_1': mres['bleu_1'][i],
                'bleu_2': mres['bleu_2'][i],
                'bleu_3': mres['bleu_3'][i],
                'bleu_4': mres['bleu_4'][i],
                'rouge_l_precision': mres['rouge_l_precision'][i],
                'rouge_l_recall': mres['rouge_l_recall'][i],
                'rouge_l_f1': mres['rouge_l_f1'][i],
                'lexical_diversity': mres['lexical_diversity'][i],
                'length_ratio': mres['length_ratio'][i],
                'distinct_1': mres['distinct_1'][i],
                'distinct_2': mres['distinct_2'][i],
                'bert_precision': mres['bert_precision'][i],
                'bert_recall': mres['bert_recall'][i],
                'bert_f1': mres['bert_f1'][i],
            })

    detailed_df = pd.DataFrame(detailed_rows)

    numeric_cols = detailed_df.select_dtypes(include=[np.number]).columns.tolist()

    overall_summary = (
        detailed_df
        .groupby('Model')[numeric_cols]
        .mean()
        .reset_index()
    )

    by_category_summary = (
        detailed_df
        .groupby(['Category', 'Model'])[numeric_cols]
        .mean()
        .reset_index()
    )

    for cat in sorted(detailed_df['Category'].dropna().unique()):
        sub = by_category_summary[by_category_summary['Category'] == cat]
        if not sub.empty:
            best_sem = sub.loc[sub['bert_f1'].idxmax()]
            best_len = sub.loc[sub['word_count'].idxmax()]
            print(f"\n[{cat}] Best semantic (BERT F1): {best_sem['Model']} ({best_sem['bert_f1']:.4f})")
            print(f"[{cat}] Most verbose: {best_len['Model']} ({best_len['word_count']:.1f} words)")

    out_dir = os.path.dirname(file_path) or "."
    detailed_path = os.path.join(out_dir, f"{base}_detailed.csv")
    overall_path = os.path.join(out_dir, f"{base}_summary_overall.csv")
    bycat_path = os.path.join(out_dir, f"{base}_summary_by_category.csv")

    detailed_df.to_csv(detailed_path, index=False)
    overall_summary.to_csv(overall_path, index=False)
    by_category_summary.to_csv(bycat_path, index=False)

    print("\nSaved:")
    print("  -", detailed_path)
    print("  -", overall_path)
    print("  -", bycat_path)

    return overall_summary, by_category_summary, detailed_df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "taskC.csv"
        # csv_file = "taskB.csv"
        # csv_file = "taskA.csv"

    analyze_csv(csv_file)