"""
Human evaluator criterion-importance analytics (Correlation + Standardized Regression)

Goal:
- Quantify which criteria drive HUMAN "Final" scores, analogous to the LLM-judge script.
- Supports multiple CSV files. Evaluator ID is extracted from filename (by default: stem before first ".",
  or with a regex if you provide one).

Input CSV format (wide, already per model row), e.g. columns:
task_id, task_type, model_name, evaluator, Fluency, Coherence, Conciseness, Accuracy,
Constructiveness, Final, Motivational, Sentiment

Outputs (CSV):
1) human_criterion_importance_spearman.csv
   - Spearman ρ between each criterion and Final
   - computed at: ALL (pooled), by task, by evaluator, and by evaluator×task
2) human_criterion_importance_regression.csv
   - Standardized regression β predicting Final from all criteria
   - same scopes as above
3) optional: human_combined_rows.csv (all input rows concatenated with evaluator_id)

Usage:
  python human_importance.py --input_glob "/path/to/humans/*.csv" --outdir "/path/to/out"

If you have one folder:
  python human_importance.py --input_dir "/path/to/humans" --outdir "/path/to/out"

If evaluator id should come from filename with regex:
  python human_importance.py --input_dir ... --evaluator_regex "eval_(.+?)_results"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_CRITERIA = [
    "Fluency",
    "Coherence",
    "Conciseness",
    "Accuracy",
    "Constructiveness",
    "Motivational",
    "Sentiment",
]
DEFAULT_FINAL_COL = "Final"


def list_input_files(input_dir: str | None, input_glob: str | None) -> list[Path]:
    files: list[Path] = []
    if input_glob:
        files = [Path(p) for p in sorted(Path().glob(input_glob))]  # relative glob
        # If user passed an absolute glob, Path().glob won't work; fallback:
        if not files and any(ch in input_glob for ch in ["*", "?", "["]):
            import glob as _glob

            files = [Path(p) for p in sorted(_glob.glob(input_glob))]
    elif input_dir:
        d = Path(input_dir)
        if not d.exists():
            raise FileNotFoundError(f"input_dir not found: {d}")
        files = sorted(d.glob("*.csv"))
    else:
        raise ValueError("Provide either --input_dir or --input_glob")
    if not files:
        raise FileNotFoundError("No CSV files found.")
    return files


def extract_evaluator_id_from_filename(path: Path, evaluator_regex: str | None) -> str:
    stem = path.stem
    if evaluator_regex:
        m = re.search(evaluator_regex, stem)
        if not m:
            raise ValueError(f"Regex did not match filename stem '{stem}': {evaluator_regex}")
        return m.group(1)
    # Default: full stem
    return stem


def read_one_csv(path: Path, evaluator_id: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize columns (strip)
    df.columns = [str(c).strip() for c in df.columns]

    required = {"task_id", "task_type", "model_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing required columns: {sorted(missing)}")

    # Add evaluator_id (from filename)
    df["evaluator_id"] = evaluator_id

    # Coerce criteria to numeric if present
    for c in DEFAULT_CRITERIA + [DEFAULT_FINAL_COL]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def build_obs_matrix(df: pd.DataFrame, criteria: list[str], final_col: str) -> pd.DataFrame:
    """
    Observation unit: (evaluator_id, task_type, task_id, model_name)
    Each row already has criteria and Final in wide form, so just keep needed cols.
    """
    keep = ["evaluator_id", "task_type", "task_id", "model_name"] + criteria + [final_col]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in combined data: {missing}")
    out = df[keep].copy()
    return out


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    tmp = pd.concat([x, y], axis=1).dropna()
    if len(tmp) < 3:
        return np.nan
    rx = tmp.iloc[:, 0].rank(method="average")
    ry = tmp.iloc[:, 1].rank(method="average")
    if rx.nunique() < 2 or ry.nunique() < 2:
        return np.nan
    return float(rx.corr(ry))


def compute_spearman_importance(df_obs: pd.DataFrame, criteria: list[str], final_col: str) -> pd.DataFrame:
    rows: list[dict] = []

    def add_scope(scope: str, sub: pd.DataFrame, evaluator: str, task_type: str):
        for c in criteria:
            rows.append(
                {
                    "scope": scope,
                    "evaluator_id": evaluator,
                    "task_type": task_type,
                    "criterion": c,
                    "spearman_rho": spearman_corr(sub[c], sub[final_col]),
                    "N": int(sub[[c, final_col]].dropna().shape[0]),
                }
            )

    # ALL pooled
    add_scope("ALL", df_obs, "ALL", "ALL")

    # By task (pooled across evaluators)
    for task, sub in df_obs.groupby("task_type", dropna=False):
        add_scope("TASK", sub, "ALL", str(task))

    # By evaluator (pooled across tasks)
    for ev, sub in df_obs.groupby("evaluator_id", dropna=False):
        add_scope("EVALUATOR", sub, str(ev), "ALL")

    # By evaluator × task
    for (ev, task), sub in df_obs.groupby(["evaluator_id", "task_type"], dropna=False):
        add_scope("EVALUATOR_TASK", sub, str(ev), str(task))

    out = pd.DataFrame(rows).sort_values(
        ["scope", "evaluator_id", "task_type", "spearman_rho"],
        ascending=[True, True, True, False],
    )
    return out


def standardized_regression(sub: pd.DataFrame, criteria: list[str], final_col: str) -> pd.DataFrame:
    """
    OLS with standardized X and y:
      z(Final) ~ z(criteria...)
    Returns standardized betas.
    """
    use = sub[criteria + [final_col]].dropna().copy()
    n = len(use)

    # Basic guard: need enough rows vs predictors
    if n < max(6, len(criteria) + 2):
        return pd.DataFrame({"criterion": criteria, "std_beta": [np.nan] * len(criteria), "N": n, "note": "insufficient_data"})

    X = use[criteria].astype(float)
    y = use[final_col].astype(float)

    # Drop constant predictors within scope
    stds = X.std(ddof=0)
    keep = stds[stds > 0].index.tolist()
    dropped = [c for c in criteria if c not in keep]

    X = X[keep]
    Xz = (X - X.mean()) / X.std(ddof=0)
    yz = (y - y.mean()) / y.std(ddof=0)

    Xmat = np.column_stack([np.ones(len(Xz)), Xz.to_numpy()])
    coef, *_ = np.linalg.lstsq(Xmat, yz.to_numpy(), rcond=None)

    rows = []
    for c, b in zip(keep, coef[1:]):
        rows.append({"criterion": c, "std_beta": float(b), "N": n, "note": ""})
    for c in dropped:
        rows.append({"criterion": c, "std_beta": np.nan, "N": n, "note": "dropped_constant"})

    return pd.DataFrame(rows)


def compute_regression_importance(df_obs: pd.DataFrame, criteria: list[str], final_col: str) -> pd.DataFrame:
    rows: list[dict] = []

    def add_scope(scope: str, sub: pd.DataFrame, evaluator: str, task_type: str):
        res = standardized_regression(sub, criteria, final_col)
        for _, r in res.iterrows():
            rows.append(
                {
                    "scope": scope,
                    "evaluator_id": evaluator,
                    "task_type": task_type,
                    "criterion": r["criterion"],
                    "std_beta": r["std_beta"],
                    "N": int(r["N"]),
                    "note": r["note"],
                }
            )

    # ALL pooled
    add_scope("ALL", df_obs, "ALL", "ALL")

    # By task
    for task, sub in df_obs.groupby("task_type", dropna=False):
        add_scope("TASK", sub, "ALL", str(task))

    # By evaluator
    for ev, sub in df_obs.groupby("evaluator_id", dropna=False):
        add_scope("EVALUATOR", sub, str(ev), "ALL")

    # By evaluator × task
    for (ev, task), sub in df_obs.groupby(["evaluator_id", "task_type"], dropna=False):
        add_scope("EVALUATOR_TASK", sub, str(ev), str(task))

    out = pd.DataFrame(rows).sort_values(
        ["scope", "evaluator_id", "task_type", "std_beta"],
        ascending=[True, True, True, False],
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default='./data', help="Directory containing human CSV files.")
    ap.add_argument("--input_glob", default=None, help='Glob for CSV files, e.g. "data/human/*.csv"')
    ap.add_argument("--outdir", default=".", help="Output directory.")
    ap.add_argument("--evaluator_regex", default='^([a-z]\d+)_', help="Regex to extract evaluator id from filename stem.")
    ap.add_argument("--final_col", default=DEFAULT_FINAL_COL, help="Name of the final score column (default: Final).")
    ap.add_argument(
        "--criteria",
        default=",".join(DEFAULT_CRITERIA),
        help="Comma-separated list of criteria columns (exclude Final).",
    )
    ap.add_argument("--save_combined", action="store_true", help="Save concatenated input as human_combined_rows.csv")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    criteria = [c.strip() for c in args.criteria.split(",") if c.strip()]
    final_col = args.final_col

    files = list_input_files(args.input_dir, args.input_glob)

    dfs = []
    for f in files:
        evaluator_id = extract_evaluator_id_from_filename(f, args.evaluator_regex)
        dfs.append(read_one_csv(f, evaluator_id))

    combined = pd.concat(dfs, ignore_index=True)

    if args.save_combined:
        combined.to_csv(outdir / "human_combined_rows.csv", index=False)

    df_obs = build_obs_matrix(combined, criteria, final_col)

    corr_df = compute_spearman_importance(df_obs, criteria, final_col)
    reg_df = compute_regression_importance(df_obs, criteria, final_col)

    corr_path = outdir / "human_criterion_importance_spearman.csv"
    reg_path = outdir / "human_criterion_importance_regression.csv"
    corr_df.to_csv(corr_path, index=False)
    reg_df.to_csv(reg_path, index=False)

    print("Read files:", len(files))
    print("Saved:", corr_path)
    print("Saved:", reg_path)

    # Quick pooled summaries
    pooled_corr = corr_df[(corr_df["scope"] == "ALL")].sort_values("spearman_rho", ascending=False)
    pooled_reg = reg_df[(reg_df["scope"] == "ALL")].sort_values("std_beta", ascending=False)

    print("\nTop Spearman ρ with Final (ALL):")
    print(pooled_corr.head(10).to_string(index=False))

    print("\nTop standardized β predicting Final (ALL):")
    print(pooled_reg.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
