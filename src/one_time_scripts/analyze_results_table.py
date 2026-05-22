from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_METRICS_FILES = [
    PROJECT_ROOT / "src/logs/0521-1525_RunMetrics_CORA-CHAM-SQUIR-ACT-TX.csv"
]

DATASET_DISPLAY_NAMES = {
    "Actor": "Actor",
    "Cora": "Cora",
    "Texas": "Texas",
    "chameleon": "Chameleon",
    "squirrel": "Squirrel",
}

DATASET_ORDER = ["Cora", "chameleon", "squirrel", "Actor", "Texas"]
MODEL_ORDER = ["GCN", "GAT", "SAGE", "GPRGNN", "H2GCN"]
RANDOM_STRATIFIER = "RandomKFold"
LABEL_STRATIFIER = "LabelStratifiedKFold"

# Zhu et al. (2020), Table 5. Values are mean accuracies in percent. SPLIT is 48-32-20
# We use H2GCN-2 for H2GCN, and GraphSAGE for the local SAGE model.
ZHU_TABLE_5 = {
    "H2GCN": {
        "Texas": 82.16,
        "Actor": 35.62,
        "squirrel": 37.90,
        "chameleon": 59.39,
        "Cora": 87.81,
    },

    # "GAT": {
    #     "Cora": 82.68,
    #     "chameleon": 54.69,
    #     "Texas": 58.38,
    #     "Actor": 26.28,
    #     "squirrel": 30.62,
    # },
}

# Chien et al. (2021),
CHIEN_TABLE_2 = {
    "GPRGNN": {
        "Cora": 88.65,   #Table 7. Pei split (48/32/20)
        "chameleon": 67.48, #Table 2.
        "Actor": 39.30,
        "squirrel": 49.93,
        "Texas": 92.92,
    },
    "GCN":{
        "Cora": 86.87, #Table 7. Pei split (48/32/20)
        "chameleon": 60.96, #Table 2.
        "Actor": 30.59,
        "squirrel": 45.66,
        "Texas": 75.16,
    },
    "SAGE": {
        "Cora": 86.58, #Table 7. Pei split (48/32/20)
        "chameleon": 62.15,
        "Actor": 36.37,
        "squirrel": 41.26,
        "Texas": 79.03,
    },
    "GAT": {
        "Cora": 87.52, #Table 7. Pei split (48/32/20)
        "chameleon": 63.9,
        "Texas": 78.87,
        "Actor": 35.98,
        "squirrel": 42.72,
    },
}

PUBLICATION_RESULTS = {
    **ZHU_TABLE_5,
    **CHIEN_TABLE_2,
}


def main() -> None:
    for path in RUN_METRICS_FILES:
        if not path.exists():
            raise FileNotFoundError(f"RunMetrics file not found: {path}")

    df = pd.concat([pd.read_csv(path) for path in RUN_METRICS_FILES], ignore_index=True)

    required_columns = {"Dataset", "Model", "StratificationType", "Test_Accuracy"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = df.dropna(subset=["Dataset", "Model", "StratificationType", "Test_Accuracy"]).copy()
    df["AccuracyPercent"] = pd.to_numeric(df["Test_Accuracy"], errors="raise")
    if df["AccuracyPercent"].max() <= 1.0:
        df["AccuracyPercent"] *= 100.0

    random_medians = (
        df[df["StratificationType"] == RANDOM_STRATIFIER]
        .groupby(["Dataset", "Model"], as_index=False)["AccuracyPercent"]
        .median()
        .rename(columns={"AccuracyPercent": "RandomMedian"})
    )
    label_medians = (
        df[df["StratificationType"] == LABEL_STRATIFIER]
        .groupby(["Dataset", "Model"], as_index=False)["AccuracyPercent"]
        .median()
        .rename(columns={"AccuracyPercent": "LabelMedian"})
    )

    summary = df[["Dataset", "Model"]].drop_duplicates()
    summary = summary.merge(random_medians, on=["Dataset", "Model"], how="left")
    summary = summary.merge(label_medians, on=["Dataset", "Model"], how="left")
    summary["Paper"] = summary.apply(
        lambda row: PUBLICATION_RESULTS.get(row["Model"], {}).get(row["Dataset"]),
        axis=1,
    )

    observed_datasets = list(dict.fromkeys(summary["Dataset"].tolist()))
    dataset_order = [dataset for dataset in DATASET_ORDER if dataset in observed_datasets]
    dataset_order.extend(dataset for dataset in observed_datasets if dataset not in dataset_order)

    observed_models = list(dict.fromkeys(summary["Model"].tolist()))
    model_order = [model for model in MODEL_ORDER if model in observed_models]
    model_order.extend(model for model in observed_models if model not in model_order)

    best_values = {}
    for dataset, dataset_summary in summary.groupby("Dataset"):
        values = dataset_summary[["RandomMedian", "LabelMedian", "Paper"]].to_numpy().ravel()
        values = pd.Series(values).dropna()
        if not values.empty:
            best_values[dataset] = values.max()

    rows = []
    for model in model_order:
        row = {"Model": model}
        for dataset in dataset_order:
            match = summary[(summary["Dataset"] == dataset) & (summary["Model"] == model)]
            if match.empty:
                row[DATASET_DISPLAY_NAMES.get(dataset, dataset)] = ""
                continue

            values = []
            best_value = best_values.get(dataset)
            for column in ["RandomMedian", "LabelMedian", "Paper"]:
                value = match.iloc[0][column]
                formatted_value = "n/a" if pd.isna(value) else f"{value:.2f}"
                if not pd.isna(value) and best_value is not None and abs(value - best_value) < 1e-9:
                    formatted_value = rf"$\bf{{{formatted_value}}}$"
                values.append(formatted_value)
            row[DATASET_DISPLAY_NAMES.get(dataset, dataset)] = " | ".join(values)
        rows.append(row)

    table_df = pd.DataFrame(
        rows,
        columns=["Model", *[DATASET_DISPLAY_NAMES.get(dataset, dataset) for dataset in dataset_order]],
    )

    seed_columns = ["Fold_Seed", "Init_Seed"]
    if set(seed_columns).issubset(df.columns):
        num_fold_seeds = df["Fold_Seed"].nunique()
        num_init_seeds = df["Init_Seed"].nunique()
        num_seed_runs = df[seed_columns].drop_duplicates().shape[0]
        sample_description = f"Empirical medians are computed from {num_seed_runs} seed runs."
        if "Fold" in df.columns:
            group_sizes = df.groupby(["Dataset", "Model", "StratificationType"]).size()
            if group_sizes.nunique() == 1:
                sample_description = (
                    f"Each empirical median with sample siz en = {group_sizes.iloc[0]} "
                    # f"from {num_seed_runs} seed runs "
                    # f"({num_fold_seeds} fold seeds x {num_init_seeds} init seeds)."
                )
    else:
        sample_description = "Medians are computed over all available fold-level test accuracies."

    cell_text = table_df.values.tolist()
    column_labels = table_df.columns.tolist()

    fig_width = max(16, 2.8 * len(column_labels))
    fig_height = max(6, 1.0 * len(table_df) + 2.3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    fig.suptitle(
        "Experiment Accuracy Summary",
        fontsize=18,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.905,
        "Each cell reports: median for random strat. | median for label-based strat. | "
        "reference paper acc. \nAll values are %-test acc\n"
        f"{sample_description}",
        ha="center",
        va="center",
        fontsize=11,
    )

    table = ax.table(
        cellText=cell_text,
        colLabels=column_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.02, 0.04, 0.96, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.15)

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
        elif row_idx % 2 == 0:
            cell.set_facecolor("#fbfbfb")

    print(f"Read {len(df)} runs from:")
    for path in RUN_METRICS_FILES:
        print(f"  {path}")
    print("Rendering table. Cell order: random median | label median | paper reference.")
    plt.show()


if __name__ == "__main__":
    main()
