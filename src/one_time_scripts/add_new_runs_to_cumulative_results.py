from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_cumulative_results import (
    CUMULATIVE_RESULTS_FILE,
    NEW_RUNS_DIR,
    collect_runmetrics_files,
    deduplicate_runmetrics,
    display_path,
    order_columns,
    read_runmetrics_file,
)


# Put folders or direct *RunMetrics*.csv files in this directory, then run this script.
# Nested folders are supported.
INPUT_PATHS = [NEW_RUNS_DIR]


def load_existing_cumulative_results():
    if not CUMULATIVE_RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"{display_path(CUMULATIVE_RESULTS_FILE)} does not exist yet. "
            "Run src/one_time_scripts/build_cumulative_results.py first."
        )
    return pd.read_csv(CUMULATIVE_RESULTS_FILE)


def add_new_runs_to_cumulative_results():
    NEW_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    files = collect_runmetrics_files(INPUT_PATHS, exclude_new_runs=False)
    files = [
        path for path in files
        if path.resolve() != CUMULATIVE_RESULTS_FILE.resolve()
    ]

    if not files:
        print(f"No new {display_path(NEW_RUNS_DIR)}/*RunMetrics*.csv files found.")
        return

    existing = load_existing_cumulative_results()
    new_frames = []
    for path in files:
        frame = read_runmetrics_file(path)
        if frame is not None:
            new_frames.append(frame)

    if not new_frames:
        print("No usable new RunMetrics files found.")
        return

    before = len(existing)
    combined = pd.concat([existing, *new_frames], ignore_index=True, sort=False)
    combined = deduplicate_runmetrics(combined)
    combined = order_columns(combined)
    combined.to_csv(CUMULATIVE_RESULTS_FILE, index=False)

    added = len(combined) - before
    print(f"Read {len(files)} new RunMetrics files from {display_path(NEW_RUNS_DIR)}.")
    print(f"Added {added} new unique run rows.")
    print(f"Cumulative file now has {len(combined)} rows: {display_path(CUMULATIVE_RESULTS_FILE)}")


if __name__ == "__main__":
    add_new_runs_to_cumulative_results()
