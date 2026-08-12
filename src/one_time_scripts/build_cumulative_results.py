from pathlib import Path
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from add_new_runs_to_cumulative_results import (
    CUMULATIVE_RESULTS_FILE,
    RUN_LOG_ROOT,
    collect_runmetrics_files,
    deduplicate_runmetrics,
    display_path,
    order_columns,
    read_runmetrics_file,
)


def build_cumulative_results():
    files = collect_runmetrics_files([RUN_LOG_ROOT])
    frames = []
    for path in files:
        frame = read_runmetrics_file(path)
        if frame is not None:
            frames.append(frame)

    if not frames:
        print(f"No RunMetrics CSV files found under {display_path(RUN_LOG_ROOT)}.")
        return

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = deduplicate_runmetrics(combined)
    combined = order_columns(combined)

    CUMULATIVE_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(CUMULATIVE_RESULTS_FILE, index=False)
    print(f"Wrote {len(combined)} rows to {display_path(CUMULATIVE_RESULTS_FILE)}.")


if __name__ == "__main__":
    build_cumulative_results()
