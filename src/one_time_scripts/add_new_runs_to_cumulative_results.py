from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.dataset_reference_metrics import canonical_dataset_key


RUN_LOG_ROOT = SRC_ROOT / "logs/runs"
NEW_RUNS_DIR = RUN_LOG_ROOT / "new_runs"
CUMULATIVE_RESULTS_FILE = RUN_LOG_ROOT / "cumulative_results.csv"
RUN_METRICS_GLOB = "*RunMetrics*.csv"
MAX_WARNING_EXAMPLES = 10

DEDUP_COLUMNS = [
    "_DatasetKey",
    "Model",
    "StratificationType",
    "Fold_Seed",
    "Init_Seed",
    "Fold",
]

REQUIRED_COLUMNS = [
    "Dataset",
    "Model",
    "StratificationType",
    "Fold_Seed",
    "Init_Seed",
    "Fold",
    "Test_Accuracy",
]

PREFERRED_COLUMN_ORDER = [
    "Dataset",
    "StratificationType",
    "Fold_Seed",
    "Fold",
    "Model",
    "Init_Seed",
    "Stopped_Epoch",
    "Best_Epoch",
    "Best_Val_Loss",
    "Best_Val_Accuracy",
    "Test_Loss",
    "Test_Accuracy",
    "Stopped_Early",
    "Early_Stopping_Patience",
    "Validation_Interval",
    "Early_Stopping_Window",
    "SourceFile",
]


# Put folders or direct *RunMetrics*.csv files in this directory, then run this script.
# Nested folders are supported.
INPUT_PATHS = [NEW_RUNS_DIR]


def display_path(path):
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def collect_runmetrics_files(paths):
    files = []
    for path in paths:
        path = Path(path)
        if path.is_file() and path.name.endswith(".csv") and "RunMetrics" in path.name:
            files.append(path)
        elif path.is_dir():
            files.extend(path.rglob(RUN_METRICS_GLOB))

    return sorted(
        {
            path.resolve()
            for path in files
            if path.resolve() != CUMULATIVE_RESULTS_FILE.resolve()
        }
    )


def read_runmetrics_file(path):
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        print(f"Skipping {display_path(path)}: missing columns {missing}")
        return None

    frame = frame.copy()
    frame["SourceFile"] = display_path(path)
    return frame


def add_dedup_key(frame):
    frame = frame.copy()
    frame["_DatasetKey"] = frame["Dataset"].map(canonical_dataset_key)
    for column in ["Fold_Seed", "Init_Seed", "Fold"]:
        try:
            frame[column] = pd.to_numeric(frame[column])
        except (TypeError, ValueError):
            pass
    return frame


def report_conflicting_duplicates(frame):
    duplicate_summary = (
        frame.groupby(DEDUP_COLUMNS, as_index=False, dropna=False)
        .agg(
            RowCount=("Test_Accuracy", "size"),
            AccuracyCount=("Test_Accuracy", "nunique"),
            SourceFileCount=("SourceFile", "nunique"),
        )
    )
    conflicts = duplicate_summary[
        (duplicate_summary["RowCount"] > 1)
        & (duplicate_summary["AccuracyCount"] > 1)
    ]
    if conflicts.empty:
        return

    print(
        "WARNING: Found duplicated seeded rows with different Test_Accuracy values. "
        "Keeping the first occurrence. Examples:"
    )
    print(conflicts.head(MAX_WARNING_EXAMPLES).to_string(index=False))


def deduplicate_runmetrics(frame):
    frame = add_dedup_key(frame)
    report_conflicting_duplicates(frame)

    before = len(frame)
    frame = frame.drop_duplicates(subset=DEDUP_COLUMNS, keep="first").copy()
    removed = before - len(frame)
    if removed:
        print(f"Removed {removed} duplicated seeded run rows.")

    return frame.drop(columns=["_DatasetKey"])


def order_columns(frame):
    preferred = [column for column in PREFERRED_COLUMN_ORDER if column in frame.columns]
    extra = [
        column for column in frame.columns
        if column not in preferred and column != "_DatasetKey"
    ]
    return frame[preferred + extra]


def load_existing_cumulative_results():
    if not CUMULATIVE_RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"{display_path(CUMULATIVE_RESULTS_FILE)} does not exist yet. "
            "Create it from the existing RunMetrics files first."
        )
    return pd.read_csv(CUMULATIVE_RESULTS_FILE)


def add_new_runs_to_cumulative_results():
    NEW_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    files = collect_runmetrics_files(INPUT_PATHS)

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
