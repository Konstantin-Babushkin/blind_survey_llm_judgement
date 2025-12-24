from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_and_tidy(csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = [("" if c is None else str(c).strip()) for c in df.columns]

    if "Candidate" not in df.columns or "Criteria" not in df.columns:
        raise ValueError(f"Expected columns 'Candidate' and 'Criteria'. Got: {list(df.columns)}")

    # Find the unnamed column holding criterion names
    blank_cols = [c for c in df.columns if c == "" or str(c).lower().startswith("unnamed:")]
    if not blank_cols:
        raise ValueError("Could not find unnamed column containing criterion names.")

    crit_col = blank_cols[0]
    df = df.rename(columns={crit_col: "criterion"})

    # Model columns
    model_cols = [c for c in df.columns if c not in {"Candidate", "Criteria", "criterion", "AVERAGE"}]
    if not model_cols:
        raise ValueError("No model columns detected.")

    # Forward-fill structural columns
    df["Candidate"] = df["Candidate"].replace({"": pd.NA}).ffill()
    df["Criteria"] = df["Criteria"].replace({"": pd.NA}).ffill()
    df["criterion"] = df["criterion"].astype(str).str.strip()

    # Drop invalid rows
    df = df[df["criterion"].notna() & (df["criterion"] != "")]
    df = df[df["criterion"].str.upper() != "AVERAGE"]

    # Numeric conversion
    for m in model_cols:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    # Build long DF with judge
    long_df = (
        df.rename(columns={"Candidate": "judge", "Criteria": "task_type"})
        [["judge", "task_type", "criterion", *model_cols]]
        .copy()
    )

    return long_df, model_cols


def winners_by_task_and_criterion(long_df: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    scores = long_df[model_cols]
    out = long_df[["judge", "task_type", "criterion"]].copy()

    out["winner_score"] = scores.max(axis=1)
    out["winner"] = scores.idxmax(axis=1)

    out["is_tie_for_first"] = scores.eq(out["winner_score"], axis=0).sum(axis=1) > 1
    out["top_models"] = scores.apply(
        lambda row: ",".join([m for m in model_cols if pd.notna(row[m]) and row[m] == row.max()]),
        axis=1,
    )

    return out.sort_values(["judge", "task_type", "criterion"]).reset_index(drop=True)


def winners_overall_by_criterion(long_df: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    by_crit = (
        long_df
        .groupby(["judge", "criterion"], as_index=False)[model_cols]
        .mean(numeric_only=True)
    )

    scores = by_crit[model_cols]
    out = by_crit[["judge", "criterion"]].copy()

    out["winner_score"] = scores.max(axis=1)
    out["winner"] = scores.idxmax(axis=1)
    out["is_tie_for_first"] = scores.eq(out["winner_score"], axis=0).sum(axis=1) > 1
    out["top_models"] = scores.apply(
        lambda row: ",".join([m for m in model_cols if pd.notna(row[m]) and row[m] == row.max()]),
        axis=1,
    )

    return out.sort_values(["criterion", "judge"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./LLM judges result - by candidates.csv")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--save_long", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    long_df, model_cols = read_and_tidy(in_path)

    winners_task = winners_by_task_and_criterion(long_df, model_cols)
    winners_overall = winners_overall_by_criterion(long_df, model_cols)

    winners_task_path = outdir / "winners_by_task_and_criterion.csv"
    winners_overall_path = outdir / "winners_overall_by_criterion.csv"

    winners_task.to_csv(winners_task_path, index=False)
    winners_overall.to_csv(winners_overall_path, index=False)

    if args.save_long:
        long_df.to_csv(outdir / "tidy_long.csv", index=False)

    print("Detected model columns:", model_cols)
    print("Saved:", winners_task_path)
    print("Saved:", winners_overall_path)

    print("\nWinner counts by judge and task:")
    print(
        winners_task
        .groupby(["judge", "task_type", "winner"])
        .size()
        .sort_values(ascending=False)
        .to_string()
    )


if __name__ == "__main__":
    main()
