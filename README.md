

- send prompts to LLM = [exec-prompt.sh](exec-prompt.sh)
```bash
./exec-prompt.sh
```

- match raw results to model names = [exec-match-results.sh](exec-match-results.sh)
```bash
./exec-match-results.sh
```

- normalize scores = [exec-normalize-scores.sh](exec-normalize-scores.sh)
```bash
./exec-normalize-scores.sh
```
- merge normalized scores into one csv = [merge_normalized_scores.py](merge_normalized_scores.py)
```bash
python merge_normalized_scores.py --indir judge_results/matched --outdir judge_results/matched
```

- aggregate sanity check results = [exec-sanity-check-aggregation.sh](exec-sanity-check-aggregation.sh)
```bash
./exec-sanity-check-aggregation.sh judge_results judge_results/matched/sanity_checks
```

- merge sanity check results = [merge_csv_files.py](merge_csv_files.py)
```bash
./merge_csv_files.py --indir judge_results/matched/sanity_checks --outfile judge_results/matched/sanity_checks
```

regex for blanks replacement
```bash
^\s*$
```

