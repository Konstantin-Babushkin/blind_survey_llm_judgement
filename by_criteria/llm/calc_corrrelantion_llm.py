from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------
# Helpers: filename parsing
# ----------------------------
_SPLIT_RE = re.compile(r"[^a-zA-Z0-9]+")


def extract_evaluator_from_filename(path: Path) -> str:
    """
    Extract evaluator name from filename stem.
    Examples:
      "deepseek idea_to_text.csv" -> "deepseek"
      "gemini_interpret.csv"      -> "gemini"
      "qwen-summarize.csv"        -> "qwen"
    """
    stem = path.stem.strip()
    toks = [t for t in _SPLIT_RE.split(stem) if t]
    if not toks:
        raise ValueError(f"Could not parse evaluator name from filename: {path.name}")
    return toks[0]


def normalize_task_type(s: str) -> str:
    """
    Optional: normalize task type strings to a consistent set.
    Adjust mappings to match your project naming.
    """
    if s is None:
        return s
    x = str(s).strip().lower()
    mapping = {
        "idea-to-text": "Idea-to-Text",
        "idea_to_text": "Idea-to-Text",
        "ideatotext": "Idea-to-Text",
        "interpretation": "Interpretation",
        "interpret": "Interpretation",
        "summarization": "Summarization",
        "summarize": "Summarization",
        "summary": "Summarization",
    }
    return mapping.get(x, str(s).strip())


# ----------------------------
# Reading & tidying
# ----------------------------
REQUIRED = {"task_id", "task_type", "model_name"}


def read_one_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name}: missing required columns: {sorted(missing)}")

    # evaluator is taken from filename (requested), stored as 'judge'
    judge = extract_evaluator_from_filename(csv_path)
    df["judge"] = judge

    # normalize task_type
    df["task_type"] = df["task_type"].apply(normalize_task_type)

    # standardize model column
    df = df.rename(columns={"model_name": "model"})

    # ensure task_id is string (safe)
    df["task_id"] = df["task_id"].astype(str).str.strip()

    # Identify criterion columns: everything except ids
    id_cols = {"task_id", "task_type", "model", "judge", "evaluator"}
    crit_cols = [c for c in df.columns if c not in id_cols]

    # numeric conversion for criteria
    for c in crit_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep only relevant columns (id + criteria)
    keep = ["judge", "task_type", "task_id", "model"] + crit_cols
    df = df[keep].copy()

    return df


def read_dir(input_dir: Path, pattern: str = "*.csv") -> pd.DataFrame:
    paths = sorted(input_dir.glob(pattern))
    if not paths:
        raise ValueError(f"No files found in {input_dir} matching pattern '{pattern}'")

    frames = []
    for p in paths:
        frames.append(read_one_csv(p))
    return pd.concat(frames, ignore_index=True)


# ----------------------------
# Stats: Spearman rho + standardized OLS beta
# ----------------------------
def spearman_rho(x: pd.Series, y: pd.Series) -> float:
    tmp = pd.concat([x, y], axis=1).dropna()
    if len(tmp) < 3:
        return np.nan
    rx = tmp.iloc[:, 0].rank()
    ry = tmp.iloc[:, 1].rank()
    if rx.nunique() < 2 or ry.nunique() < 2:
        return np.nan
    return float(rx.corr(ry))


def spearman_corr_with_final(df: pd.DataFrame, final_col: str = "Final choice") -> pd.DataFrame:
    """
    Compute Spearman rho between each criterion and final choice under scopes:
      - JUDGE: per judge pooled across tasks
      - JUDGE_TASK: per (judge, task_type)
      - TASK: per task_type pooled across all judges  <-- you asked for this
      - ALL: pooled across all judges and tasks
    """
    if final_col not in df.columns:
        raise ValueError(f"'{final_col}' column not found. Available: {list(df.columns)}")

    id_cols = {"judge", "task_type", "task_id", "model"}
    crit_cols = [c for c in df.columns if c not in id_cols and c != final_col]

    rows: list[dict] = []

    # JUDGE
    for judge, sub in df.groupby("judge"):
        for c in crit_cols:
            rows.append(
                dict(
                    scope="JUDGE",
                    judge=judge,
                    task_type="ALL",
                    criterion=c,
                    spearman_rho=spearman_rho(sub[c], sub[final_col]),
                    n_obs=int(sub[[c, final_col]].dropna().shape[0]),
                )
            )

    # JUDGE_TASK
    for (judge, task), sub in df.groupby(["judge", "task_type"]):
        for c in crit_cols:
            rows.append(
                dict(
                    scope="JUDGE_TASK",
                    judge=judge,
                    task_type=task,
                    criterion=c,
                    spearman_rho=spearman_rho(sub[c], sub[final_col]),
                    n_obs=int(sub[[c, final_col]].dropna().shape[0]),
                )
            )

    # TASK (pooled across all judges)
    for task, sub in df.groupby("task_type"):
        for c in crit_cols:
            rows.append(
                dict(
                    scope="TASK",
                    judge="ALL",
                    task_type=task,
                    criterion=c,
                    spearman_rho=spearman_rho(sub[c], sub[final_col]),
                    n_obs=int(sub[[c, final_col]].dropna().shape[0]),
                )
            )

    # ALL
    for c in crit_cols:
        rows.append(
            dict(
                scope="ALL",
                judge="ALL",
                task_type="ALL",
                criterion=c,
                spearman_rho=spearman_rho(df[c], df[final_col]),
                n_obs=int(df[[c, final_col]].dropna().shape[0]),
            )
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["scope", "judge", "task_type", "spearman_rho"], ascending=[True, True, True, False])


def fit_standardized_ols(sub: pd.DataFrame, crit_cols: list[str], final_col: str) -> tuple[dict[str, float], int, str]:
    tmp = sub[crit_cols + [final_col]].dropna()
    n = len(tmp)
    if n < 3:
        return {c: np.nan for c in crit_cols}, n, "insufficient_data"

    X = tmp[crit_cols]
    y = tmp[final_col]

    # drop zero-variance predictors
    stds = X.std(ddof=0)
    keep = stds[stds > 0].index.tolist()
    if not keep:
        return {c: np.nan for c in crit_cols}, n, "no_variance_predictors"

    if n < len(keep) + 2:
        return {c: np.nan for c in crit_cols}, n, "insufficient_data"

    X = X[keep]
    Xz = (X - X.mean()) / X.std(ddof=0)
    yz = (y - y.mean()) / y.std(ddof=0)

    Xmat = np.column_stack([np.ones(len(Xz)), Xz.to_numpy()])
    coef, *_ = np.linalg.lstsq(Xmat, yz.to_numpy(), rcond=None)

    betas_keep = dict(zip(keep, coef[1:].tolist()))
    betas_all = {c: float(betas_keep.get(c, np.nan)) for c in crit_cols}
    return betas_all, n, ""


def standardized_regression_to_final(df: pd.DataFrame, final_col: str = "Final choice") -> pd.DataFrame:
    """
    Standardized OLS betas predicting final choice under scopes:
      - JUDGE
      - JUDGE_TASK
      - TASK (pooled across all judges)
      - ALL
    """
    if final_col not in df.columns:
        raise ValueError(f"'{final_col}' column not found. Available: {list(df.columns)}")

    id_cols = {"judge", "task_type", "task_id", "model"}
    crit_cols = [c for c in df.columns if c not in id_cols and c != final_col]

    rows: list[dict] = []

    def emit(scope: str, judge: str, task: str, sub: pd.DataFrame) -> None:
        betas, n, note = fit_standardized_ols(sub, crit_cols, final_col)
        for c in crit_cols:
            rows.append(
                dict(
                    scope=scope,
                    judge=judge,
                    task_type=task,
                    criterion=c,
                    std_beta=betas[c],
                    n_obs=int(n),
                    note=note,
                )
            )

    # JUDGE
    for judge, sub in df.groupby("judge"):
        emit("JUDGE", judge, "ALL", sub)

    # JUDGE_TASK
    for (judge, task), sub in df.groupby(["judge", "task_type"]):
        emit("JUDGE_TASK", judge, task, sub)

    # TASK (pooled across all judges)
    for task, sub in df.groupby("task_type"):
        emit("TASK", "ALL", task, sub)

    # ALL
    emit("ALL", "ALL", "ALL", df)

    out = pd.DataFrame(rows)
    return out.sort_values(["scope", "judge", "task_type", "std_beta"], ascending=[True, True, True, False])


# ----------------------------
# CLI
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default='./data', help="Directory containing CSV files.")
    ap.add_argument("--pattern", default="*.csv", help="Glob pattern for CSVs, default: *.csv")
    ap.add_argument("--outdir", default=".", help="Output directory.")
    ap.add_argument("--final_col", default="Final choice", help="Column name of the final preference label.")
    ap.add_argument(
        "--task_type",
        default=None,
        help="If set, print top rho/beta for this task type pooled across ALL evaluators (scope=TASK).",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_dir(input_dir, pattern=args.pattern)

    # Save combined tidy input (useful for debugging)
    combined_path = outdir / "combined_tidy.csv"
    df.to_csv(combined_path, index=False)

    corr_df = spearman_corr_with_final(df, final_col=args.final_col)
    corr_path = outdir / "criterion_importance_spearman.csv"
    corr_df.to_csv(corr_path, index=False)

    reg_df = standardized_regression_to_final(df, final_col=args.final_col)
    reg_path = outdir / "criterion_importance_regression.csv"
    reg_df.to_csv(reg_path, index=False)

    print("Saved:", combined_path)
    print("Saved:", corr_path)
    print("Saved:", reg_path)

    # Quick pooled summary
    pooled_corr = corr_df[corr_df["scope"] == "ALL"].sort_values("spearman_rho", ascending=False)
    print("\nTop Spearman rhos with Final choice (ALL):")
    print(pooled_corr.head(10).to_string(index=False))

    pooled_reg = reg_df[reg_df["scope"] == "ALL"].sort_values("std_beta", ascending=False)
    print("\nTop standardized betas predicting Final choice (ALL):")
    print(pooled_reg.head(10).to_string(index=False))

    # Task scope summary (what you asked for)
    if args.task_type:
        task = normalize_task_type(args.task_type)
        task_corr = corr_df[(corr_df["scope"] == "TASK") & (corr_df["task_type"] == task)].sort_values(
            "spearman_rho", ascending=False
        )
        task_reg = reg_df[(reg_df["scope"] == "TASK") & (reg_df["task_type"] == task)].sort_values(
            "std_beta", ascending=False
        )

        if task_corr.empty:
            print(f"\nNo TASK-scope rows found for task_type='{task}'. Available task types:")
            print(sorted(df["task_type"].dropna().unique().tolist()))
        else:
            print(f"\nTop Spearman rhos for task_type='{task}' (pooled across ALL evaluators):")
            print(task_corr.head(10).to_string(index=False))

            print(f"\nTop standardized betas for task_type='{task}' (pooled across ALL evaluators):")
            print(task_reg.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
