from itertools import product
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_METRICS_FILE = PROJECT_ROOT / "src/logs/runs/0522-1916_RunMetrics_CORA-CITE-CHAM-ACT-TX.csv"
GROUP_COLS = ["Dataset", "StratificationType", "Model"]
RUN_COLS = ["Fold_Seed", "Fold", "Init_Seed"]
MAX_EXAMPLES = 20


df = pd.read_csv(RUN_METRICS_FILE)
df = df[GROUP_COLS + RUN_COLS].drop_duplicates()

models = sorted(df["Model"].unique())
fold_seeds = sorted(map(int, df["Fold_Seed"].unique()))
folds = sorted(map(int, df["Fold"].unique()))
init_seeds = sorted(map(int, df["Init_Seed"].unique()))

expected_runs = set(product(fold_seeds, folds, init_seeds))
expected_count = len(expected_runs)
print(f"Expected per Dataset x StratificationType x Model: {expected_count} runs")
print(f"Fold seeds={fold_seeds}, folds={folds}, init seeds={init_seeds}\n")

missing_model_groups = []
incomplete_groups = []

dataset_stratifier_pairs = (
    df[["Dataset", "StratificationType"]]
    .drop_duplicates()
    .sort_values(["Dataset", "StratificationType"])
    .itertuples(index=False, name=None)
)

for dataset, stratifier in dataset_stratifier_pairs:
    for model in models:
        group = df[
            (df["Dataset"] == dataset)
            & (df["StratificationType"] == stratifier)
            & (df["Model"] == model)
        ]

        if group.empty:
            missing_model_groups.append((dataset, stratifier, model))
            continue

        observed_runs = set(map(tuple, group[RUN_COLS].astype(int).to_numpy()))
        missing_runs = sorted(expected_runs - observed_runs)
        if missing_runs:
            incomplete_groups.append((dataset, stratifier, model, len(observed_runs), missing_runs))

if missing_model_groups:
    print("Missing models inside existing Dataset x StratificationType groups:")
    for dataset, stratifier, model in missing_model_groups:
        print(f"  {dataset} | {stratifier} | {model}: 0/{expected_count}")
else:
    print("No missing models inside existing Dataset x StratificationType groups.")

print()
if incomplete_groups:
    print("Incomplete existing groups:")
    for dataset, stratifier, model, observed_count, missing_runs in incomplete_groups:
        print(f"\n{dataset} | {stratifier} | {model}: {observed_count}/{expected_count}")
        for fold_seed, fold, init_seed in missing_runs[:MAX_EXAMPLES]:
            print(f"  missing Fold_Seed={fold_seed}, Fold={fold}, Init_Seed={init_seed}")
        if len(missing_runs) > MAX_EXAMPLES:
            print(f"  ... {len(missing_runs) - MAX_EXAMPLES} more")
else:
    print("No incomplete existing groups.")

print("\nObserved counts:")
print(
    df.groupby(GROUP_COLS)
    .size()
    .rename("Runs")
    .reset_index()
    .pivot_table(
        index=["Dataset", "StratificationType"],
        columns="Model",
        values="Runs",
        fill_value=0,
    )
)
