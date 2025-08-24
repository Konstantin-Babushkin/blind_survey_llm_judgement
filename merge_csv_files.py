#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merges all CSV files from a specified directory into a single CSV file.

This script iterates through all files ending with .csv in the input directory,
concatenates them, and saves the result to a new CSV file.

Usage:
  python ./merge_csv_files.py --indir /path/to/csv/dir --outfile merged.csv
"""

import argparse
import os
import pandas as pd
import glob

def main(indir: str, outfile: str):
    # --- 1. Validate Input Directory ---
    if not os.path.isdir(indir):
        raise SystemExit(f"Error: Input directory not found at '{indir}'")

    # --- 2. Find and Read CSV Files ---
    csv_files = glob.glob(os.path.join(indir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in '{indir}'. Nothing to merge.")
        return

    print(f"Found {len(csv_files)} CSV files to merge.")
    
    df_list = []
    for f in sorted(csv_files):
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not read or parse {f}. Skipping. Error: {e}")
            continue
            
    if not df_list:
        print("No valid CSV files could be read. Aborting.")
        return

    # --- 3. Concatenate and Save ---
    merged_df = pd.concat(df_list, ignore_index=True)
    
    # Ensure the output directory exists
    out_dir = os.path.dirname(outfile)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    merged_df.to_csv(outfile, index=False)
    
    print(f"✔ Successfully merged {len(df_list)} files into '{outfile}'.")
    print(f"Total rows in merged file: {len(merged_df)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge all CSV files in a directory.")
    ap.add_argument(
        "--indir",
        required=True,
        help="Input directory containing the CSV files to merge."
    )
    ap.add_argument(
        "--outfile",
        default="merged_results.csv",
        help="Path to the output merged CSV file."
    )
    args = ap.parse_args()
    main(args.indir, args.outfile)
