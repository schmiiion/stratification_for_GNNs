from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stratify.run_wdes_hyperparam_search import (
    EPS,
    WDES_HYPERPARAM_GRID,
    add_stratified_kfold_comparison,
    run_search,
)
from utils.dataset_reference_metrics import dataset_metric_summary


DATASET_NAME = "actor"
NUM_SPLIT_RUNS = 20

# Set this to False when you really want to rerun the WDES hyperparameter search.
USE_EXISTING_RESULTS = True
# RESULTS_CSV_PATH = SRC_ROOT / "logs/wdes_hyperparams/0523-2032_WdesHyperparamSearch_chameleon_filtered.csv"
RESULTS_CSV_PATH = SRC_ROOT / "logs/wdes_hyperparams/0623-0914_WdesHyperparamSearch_Cora.csv"
OUTPUT_DIR = SRC_ROOT / "outputs/stratification_diagnostics"

PROPERTY_LABELS = {
    "Degree": "Degree",
    "Neighborhood Heterogeneity": "Neigh. het.",
    "PageRank": "PageRank",
    "Eigenvector Centrality": "Eigenvector",
    "Clustering Coefficient": "Clustering",
}


def resolve_results_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_existing_results(path):
    path = resolve_results_path(path)
    if not path.exists():
        raise FileNotFoundError(f"WDES hyperparameter results CSV not found: {path}")

    results = pd.read_csv(path)
    required_columns = {
        "Dataset",
        "TargetProperty",
        "StratSeed",
        "HyperparamName",
        "Population",
        "Generations",
        "RandomTargetEmd",
        "LabelTargetEmd",
        "WdesTargetEmd",
    }
    missing_columns = required_columns - set(results.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing_columns)}")

    dataset_names = sorted(results["Dataset"].dropna().unique())
    dataset_name = dataset_names[0] if len(dataset_names) == 1 else DATASET_NAME
    num_split_runs = int(results["StratSeed"].nunique())

    hyperparam_grid = []
    for _, row in results.drop_duplicates("HyperparamName").iterrows():
        hyperparam_grid.append({
            "Name": row["HyperparamName"],
            "wdes_n_pop": int(row["Population"]),
            "wdes_n_gen": int(row["Generations"]),
        })

    return results, path, dataset_name, num_split_runs, hyperparam_grid


def make_matrices(results, hyperparam_name):
    rows = ["RandomKFold", "LabelStratifiedKFold"]
    properties = list(results["TargetProperty"].drop_duplicates())
    absolute_values = pd.DataFrame(index=rows, columns=properties, dtype=float)
    ratio_values = pd.DataFrame(index=rows, columns=properties, dtype=float)
    absolute_annotations = pd.DataFrame(index=rows, columns=properties, dtype=str)
    ratio_annotations = pd.DataFrame(index=rows, columns=properties, dtype=str)

    subset = results[results["HyperparamName"] == hyperparam_name]
    for property_name in properties:
        property_rows = subset[subset["TargetProperty"] == property_name]
        random_emd = property_rows["RandomTargetEmd"].mean()
        label_emd = property_rows["LabelTargetEmd"].mean()
        stratified_kfold_emd = property_rows["StratifiedKFoldTargetEmd"].mean()
        wdes_emd = property_rows["WdesTargetEmd"].mean()

        absolute_values.loc["RandomKFold", property_name] = random_emd
        absolute_values.loc["LabelStratifiedKFold", property_name] = label_emd
        absolute_annotations.loc["RandomKFold", property_name] = (
            f"base {random_emd:.2e}\nGA {wdes_emd:.2e}\nSKF {stratified_kfold_emd:.2e}"
        )
        absolute_annotations.loc["LabelStratifiedKFold", property_name] = (
            f"base {label_emd:.2e}\nGA {wdes_emd:.2e}\nSKF {stratified_kfold_emd:.2e}"
        )

        random_ratio = random_emd / max(wdes_emd, EPS)
        label_ratio = label_emd / max(wdes_emd, EPS)
        random_stratified_ratio = random_emd / max(stratified_kfold_emd, EPS)
        label_stratified_ratio = label_emd / max(stratified_kfold_emd, EPS)
        ratio_values.loc["RandomKFold", property_name] = random_ratio
        ratio_values.loc["LabelStratifiedKFold", property_name] = label_ratio
        ratio_annotations.loc["RandomKFold", property_name] = (
            f"GA {random_ratio:.2f}x\nSKF {random_stratified_ratio:.2f}x"
        )
        ratio_annotations.loc["LabelStratifiedKFold", property_name] = (
            f"GA {label_ratio:.2f}x\nSKF {label_stratified_ratio:.2f}x"
        )

    renamed_columns = [PROPERTY_LABELS.get(column, column) for column in properties]
    for frame in [absolute_values, ratio_values, absolute_annotations, ratio_annotations]:
        frame.columns = renamed_columns

    return absolute_values, ratio_values, absolute_annotations, ratio_annotations


def plot_results(results, dataset_name, num_split_runs, hyperparam_grid):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d-%H%M")
    output_base = OUTPUT_DIR / f"{timestamp}_WDES_HyperparamStudy_{dataset_name}_n{num_split_runs}"

    fig, axes = plt.subplots(
        nrows=len(hyperparam_grid),
        ncols=2,
        figsize=(22, 5.2 * len(hyperparam_grid)),
        dpi=220,
        constrained_layout=True,
        squeeze=False,
    )

    fig.suptitle(
        f"{dataset_name}: WDES stratification study | {dataset_metric_summary(dataset_name)} | "
        f"n={num_split_runs} split runs per method",
        fontsize=20,
        fontweight="bold",
    )

    for row_idx, hyperparams in enumerate(hyperparam_grid):
        hyperparam_name = hyperparams["Name"]
        absolute_values, ratio_values, absolute_annotations, ratio_annotations = make_matrices(
            results,
            hyperparam_name,
        )

        sns.heatmap(
            absolute_values,
            ax=axes[row_idx, 0],
            annot=absolute_annotations,
            fmt="",
            cmap="Blues",
            cbar=row_idx == 0,
            cbar_kws={"label": "Baseline mean EMD"} if row_idx == 0 else None,
            linewidths=0.5,
            linecolor="white",
        )
        axes[row_idx, 0].set_title(
            f"Absolute EMD | pop={hyperparams['wdes_n_pop']}, gen={hyperparams['wdes_n_gen']}",
            fontsize=13,
        )
        axes[row_idx, 0].set_xlabel("WDES target property")
        axes[row_idx, 0].set_ylabel("Baseline")

        sns.heatmap(
            ratio_values,
            ax=axes[row_idx, 1],
            annot=ratio_annotations,
            fmt="",
            cmap="RdYlGn",
            center=1.0,
            vmin=min(1.0, float(ratio_values.min().min())),
            cbar=row_idx == 0,
            cbar_kws={"label": "Baseline mean EMD / WDES mean EMD"} if row_idx == 0 else None,
            linewidths=0.5,
            linecolor="white",
        )
        axes[row_idx, 1].set_title(
            f"Reduction ratio, color=GA | pop={hyperparams['wdes_n_pop']}, gen={hyperparams['wdes_n_gen']}",
            fontsize=13,
        )
        axes[row_idx, 1].set_xlabel("WDES target property")
        axes[row_idx, 1].set_ylabel("Baseline")

    fig.savefig(f"{output_base}.png", dpi=300)
    fig.savefig(f"{output_base}.pdf")
    plt.show()

    print(f"Saved plot to: {output_base}.png")
    print(f"Saved vector plot to: {output_base}.pdf")


def main() -> None:
    if USE_EXISTING_RESULTS:
        results, output_path, dataset_name, num_split_runs, hyperparam_grid = load_existing_results(
            RESULTS_CSV_PATH
        )
        print(f"Loaded raw WDES hyperparameter search from: {output_path}")
    else:
        results, output_path = run_search(DATASET_NAME, NUM_SPLIT_RUNS)
        dataset_name = DATASET_NAME
        num_split_runs = NUM_SPLIT_RUNS
        hyperparam_grid = WDES_HYPERPARAM_GRID
        print(f"Saved raw WDES hyperparameter search to: {output_path}")

    results = add_stratified_kfold_comparison(results, dataset_name)
    plot_results(results, dataset_name, num_split_runs, hyperparam_grid)


if __name__ == "__main__":
    main()
