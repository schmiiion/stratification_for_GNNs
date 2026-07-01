import pandas as pd
import os
import warnings
from pathlib import Path

# Suppress the pandas warning about the python engine sniffing the delimiter
warnings.filterwarnings("ignore", message=".*Falling back to the 'python' engine.*")


def concatenate_and_verify(file1_abs, file2_abs):

    print(f"Processing:")
    print(f"  File 1 (Target): {file1_abs}")
    print(f"  File 2 (Source): {file2_abs}")

    if not os.path.exists(file1_abs):
        print(f"  [X] Error: File 1 does NOT exist at this exact location. Skipping.\n")
        return
    if not os.path.exists(file2_abs):
        print(f"  [X] Error: File 2 does NOT exist at this exact location. Skipping.\n")
        return

    # --- THE FIX: Let Pandas automatically detect ',' vs ';' ---
    df1 = pd.read_csv(file1_abs, sep=None, engine='python')
    df2 = pd.read_csv(file2_abs, sep=None, engine='python')

    # 1. Check if column headers match exactly
    if list(df1.columns) != list(df2.columns):
        print("  [X] Error: Column headers do not match! Concatenation aborted.")
        print(f"      File 1 headers: {list(df1.columns)}")
        print(f"      File 2 headers: {list(df2.columns)}\n")
        return
    else:
        print("  [✓] Column headers match.")

    # Combine dataframes to check for duplicates
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # 2. Check for row duplications
    if combined_df.duplicated().any():
        num_duplicates = combined_df.duplicated().sum()
        print(f"  [X] Error: Found {num_duplicates} duplicated row(s) between the files! Concatenation aborted.\n")
        return
    else:
        print("  [✓] No row duplications found.")

    # 3. Concatenate and save back to the first file's path (forcing standard commas)
    combined_df.to_csv(file1_abs, index=False, sep=',')
    print(f"  [✓] Success! Concatenated data saved to: {file1_abs}\n")


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[2]
    run_results_file1 = project_root / "src/logs/runs/0522-1916_FoldStatistics_CORA-CITE-CHAM-ACT-TX.csv"
    run_results_file2 = project_root / "src/logs/runs/0527-0850_FoldStatistics_CITE.csv"

    # run_results_file1 = project_root / "src/logs/archive/20260517_143150_FoldStatistics.csv"
    # run_results_file2 = project_root / "src/logs/archive/20260520_225756_FoldStatistics.csv"

    print("--- Checking Run Results ---")
    concatenate_and_verify(run_results_file1, run_results_file2)
