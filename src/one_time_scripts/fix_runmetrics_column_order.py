from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
USE_ORDERING_FROM_THIS_FILE = (
    PROJECT_ROOT / "src/logs/runs/0522-1916_FoldStatistics_CORA-CITE-CHAM-ACT-TX.csv"
)
ALIGN_THIS_FILE = Path(
    "/Users/jonas/Uni/SoSe26/Project/stratification_for_GNNs/src/logs/runs/0527-0850_FoldStatistics_CITE.csv"
)


desired_column_order = list(pd.read_csv(USE_ORDERING_FROM_THIS_FILE, nrows=0).columns)
file_to_align = pd.read_csv(ALIGN_THIS_FILE)

if set(file_to_align.columns) != set(desired_column_order):
    missing = sorted(set(desired_column_order) - set(file_to_align.columns))
    extra = sorted(set(file_to_align.columns) - set(desired_column_order))
    raise ValueError(f"Column sets differ. Missing={missing}, extra={extra}")

backup_file = ALIGN_THIS_FILE.with_suffix(".before_column_fix.csv")
file_to_align.to_csv(backup_file, index=False)
file_to_align[desired_column_order].to_csv(ALIGN_THIS_FILE, index=False)

print(f"Used column order from: {USE_ORDERING_FROM_THIS_FILE}")
print(f"Reordered columns in: {ALIGN_THIS_FILE}")
print(f"Backup written to: {backup_file}")
print(f"New header: {list(pd.read_csv(ALIGN_THIS_FILE, nrows=0).columns)}")
