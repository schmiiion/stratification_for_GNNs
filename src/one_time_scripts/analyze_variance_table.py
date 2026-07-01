from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.dataset_reference_metrics import dataset_metric_summary

RUN_METRICS_FILE = PROJECT_ROOT / "src/logs/runs/0522-1916_RunMetrics_CORA-CITE-CHAM-ACT-TX.csv"

DATASET_ORDER = ["Cora", "CiteSeer", "chameleon", "Actor", "Texas"]
MODEL_ORDER = ["GCN", "GAT", "SAGE", "GPRGNN", "H2GCN", "MLP"]
STRATIFIER_ORDER = [
    "RandomKFold",
    "LabelStratifiedKFold",
    "WDES_Degree",
    "WDES_NeighHet",
    "WDES_PageRank",
    "WDES_EigCentrality",
    "WDES_Clustering",
]
STRATIFIER_LABELS = {
    "RandomKFold": "Random",
    "LabelStratifiedKFold": "Label",
    "WDES_Degree": "WDES\nDegree",
    "WDES_NeighHet": "WDES\nNeigh. Het.",
    "WDES_PageRank": "WDES\nPageRank",
    "WDES_EigCentrality": "WDES\nEigenvector",
    "WDES_Clustering": "WDES\nClustering",
}


PROPERTY_ORDER = {
    "Degree": 0,
    "NeighHet": 1,
    "PageRank": 2,
    "EigCentrality": 3,
    "Clustering": 4,
}

PROPERTY_LABELS = {
    "Degree": "Degree",
    "NeighHet": "Neigh. Het.",
    "PageRank": "PageRank",
    "EigCentrality": "Eigenvector",
    "Clustering": "Clustering",
}


def stratifier_sort_key(stratifier):
    if stratifier in STRATIFIER_ORDER:
        return (0, STRATIFIER_ORDER.index(stratifier), 0, stratifier)
    if stratifier.startswith("StratifiedKFoldDynamic_"):
        parts = stratifier.split("_")
        property_name = parts[1] if len(parts) > 1 else ""
        bin_part = parts[2] if len(parts) > 2 and parts[2].startswith("b") else "b0"
        try:
            selected_bins = int(bin_part[1:])
        except ValueError:
            selected_bins = 0
        return (1, PROPERTY_ORDER.get(property_name, 99), selected_bins, stratifier)
    return (2, 99, 0, stratifier)


def stratifier_label(stratifier):
    if stratifier in STRATIFIER_LABELS:
        return STRATIFIER_LABELS[stratifier]
    if stratifier.startswith("StratifiedKFoldDynamic_"):
        parts = stratifier.split("_")
        property_name = parts[1] if len(parts) > 1 else ""
        bin_part = parts[2] if len(parts) > 2 and parts[2].startswith("b") else ""
        label = PROPERTY_LABELS.get(property_name, property_name)
        return f"SKF dyn.\n{label}\n{bin_part}" if bin_part else f"SKF dyn.\n{label}"
    return stratifier


df = pd.read_csv(RUN_METRICS_FILE)
# df = df[df["Model"] != "MLP"].copy()
df["AccuracyPercent"] = pd.to_numeric(df["Test_Accuracy"], errors="raise")
if df["AccuracyPercent"].max() <= 1.0:
    df["AccuracyPercent"] *= 100.0

fold_seed_stats = (
    df.groupby(["Dataset", "Model", "StratificationType", "Fold_Seed", "Init_Seed"], as_index=False)
    .agg(
        Mean=("AccuracyPercent", "mean"),
        Std=("AccuracyPercent", "std"),
        N=("AccuracyPercent", "size"),
    )
)

stats = (
    fold_seed_stats.groupby(["Dataset", "Model", "StratificationType"], as_index=False)
    .agg(
        Mean=("Mean", "mean"),
        Std=("Std", "mean"),
        N=("Std", "size"),
    )
)
stratifiers = sorted(stats["StratificationType"].unique(), key=stratifier_sort_key)

datasets = [dataset for dataset in DATASET_ORDER if dataset in stats["Dataset"].unique()]
datasets += [
    dataset
    for dataset in sorted(stats["Dataset"].unique())
    if dataset not in datasets
]

for dataset in datasets:
    dataset_stats = stats[stats["Dataset"] == dataset]
    models = [model for model in MODEL_ORDER if model in dataset_stats["Model"].unique()]
    table_rows = []
    bold_cells = set()
    for row_idx, model in enumerate(models, start=1):
        model_stats = dataset_stats[dataset_stats["Model"] == model]
        row = [model]

        available_stds = model_stats[model_stats["StratificationType"].isin(stratifiers)]["Std"]
        best_std = available_stds.min() if not available_stds.empty else None

        for col_idx, stratifier in enumerate(stratifiers, start=1):
            values = model_stats[model_stats["StratificationType"] == stratifier]
            if values.empty:
                row.append("n/a")
                continue

            mean = values.iloc[0]["Mean"]
            std = values.iloc[0]["Std"]
            row.append(f"Mean {mean:.2f}\nFold Std {std:.2f}")
            if best_std is not None and abs(std - best_std) < 1e-12:
                bold_cells.add((row_idx, col_idx))

        table_rows.append(row)

    n_values = sorted(map(int, dataset_stats["N"].unique()))
    n_text = f"n={n_values[0]}" if len(n_values) == 1 else f"n={min(n_values)}-{max(n_values)}"
    fig_width = max(15, 2.35 * (len(stratifiers) + 1))
    fig_height = max(6, 0.95 * len(models) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    fig.suptitle(
        f"{dataset}: Accuracy Dispersion | {dataset_metric_summary(dataset)} | {n_text}",
        fontsize=18,
        fontweight="bold",
        y=0.96,
    )
    fig.text(
        0.5,
        0.89,
        "Each cell reports mean test accuracy and the mean std across the 5 folds. "
        "Bold marks the lowest mean fold-std within each model row.",
        ha="center",
        va="center",
        fontsize=11,
    )

    table = ax.table(
        cellText=table_rows,
        colLabels=["Model", *[stratifier_label(stratifier) for stratifier in stratifiers]],
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.02, 0.05, 0.96, 0.76],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.0)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.visible_edges = "horizontal"
        cell.set_edgecolor("#333333")
        cell.set_linewidth(0.6)

        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")
            cell.set_linewidth(1.0)
        elif col_idx == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#fafafa")
        elif (row_idx, col_idx) in bold_cells:
            cell.set_text_props(weight="bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#fbfbfb")

    print(f"Rendering {dataset}: {len(dataset_stats)} model/stratifier summaries from {RUN_METRICS_FILE}")
    plt.show()
