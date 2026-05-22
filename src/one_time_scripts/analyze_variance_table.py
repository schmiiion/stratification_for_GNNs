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
    df = df[df["StratificationType"].isin([RANDOM_STRATIFIER, LABEL_STRATIFIER])]
    df["AccuracyPercent"] = pd.to_numeric(df["Test_Accuracy"], errors="raise")
    if df["AccuracyPercent"].max() <= 1.0:
        df["AccuracyPercent"] *= 100.0

    random_stats = (
        df[df["StratificationType"] == RANDOM_STRATIFIER]
        .groupby(["Dataset", "Model"], as_index=False)["AccuracyPercent"]
        .agg(RandomMean="mean", RandomVariance="var")
    )
    label_stats = (
        df[df["StratificationType"] == LABEL_STRATIFIER]
        .groupby(["Dataset", "Model"], as_index=False)["AccuracyPercent"]
        .agg(LabelMean="mean", LabelVariance="var")
    )

    summary = df[["Dataset", "Model"]].drop_duplicates()
    summary = summary.merge(random_stats, on=["Dataset", "Model"], how="left")
    summary = summary.merge(label_stats, on=["Dataset", "Model"], how="left")

    observed_datasets = list(dict.fromkeys(summary["Dataset"].tolist()))
    dataset_order = [dataset for dataset in DATASET_ORDER if dataset in observed_datasets]
    dataset_order.extend(dataset for dataset in observed_datasets if dataset not in dataset_order)

    observed_models = list(dict.fromkeys(summary["Model"].tolist()))
    model_order = [model for model in MODEL_ORDER if model in observed_models]
    model_order.extend(model for model in observed_models if model not in model_order)

    rows = []
    for model in model_order:
        row = {"Model": model}
        for dataset in dataset_order:
            match = summary[(summary["Dataset"] == dataset) & (summary["Model"] == model)]
            if match.empty:
                row[DATASET_DISPLAY_NAMES.get(dataset, dataset)] = ""
                continue

            values = match.iloc[0]
            random_text = "n/a"
            label_text = "n/a"
            if not pd.isna(values["RandomMean"]) and not pd.isna(values["RandomVariance"]):
                random_text = f"{values['RandomMean']:.2f} ({values['RandomVariance']:.2f})"
            if not pd.isna(values["LabelMean"]) and not pd.isna(values["LabelVariance"]):
                label_text = f"{values['LabelMean']:.2f} ({values['LabelVariance']:.2f})"
            row[DATASET_DISPLAY_NAMES.get(dataset, dataset)] = f"{random_text} | {label_text}"
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
        sample_description = f"Means and variances are computed from {num_seed_runs} seed runs."
        if "Fold" in df.columns:
            group_sizes = df.groupby(["Dataset", "Model", "StratificationType"]).size()
            if group_sizes.nunique() == 1:
                sample_description = (
                    f"Each mean and sample variance uses n = {group_sizes.iloc[0]} "
                    f"fold-level test accuracies from {num_seed_runs} seed runs "
                    f"({num_fold_seeds} fold seeds x {num_init_seeds} init seeds)."
                )
    else:
        sample_description = "Means and variances are computed over all available test accuracies."

    cell_text = table_df.values.tolist()
    column_labels = table_df.columns.tolist()

    fig_width = max(16, 2.8 * len(column_labels))
    fig_height = max(6, 1.0 * len(table_df) + 2.3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    fig.suptitle(
        "Experiment Accuracy Variance Summary",
        fontsize=18,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.905,
        "Each cell reports: random mean (variance) | label-based mean (variance). "
        "\nAll values are %-test acc; variance is the sample variance on %-test acc.\n"
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
    table.set_fontsize(10)
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

    print(f"Read {len(df)} random/label-stratified runs from:")
    for path in RUN_METRICS_FILES:
        print(f"  {path}")
    print("Rendering table. Cell order: random mean (variance) | label mean (variance).")
    plt.show()


if __name__ == "__main__":
    main()
