from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import pandas as pd
import seaborn as sns


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from factories.dataset_factory import DatasetFactory
from stratify.baseclass import BaseNodeStratifier
from stratify.label_based_stratifier import LabelStratifiedKFold
from stratify.random_stratifier import RandomKFold
from stratify.run_wdes_hyperparam_search import (
    EPS,
    run_baseline,
    run_property_stratified_kfold,
    target_column_for_property,
)
from stratify.wdes_stratifier import WDESKFold
from utils.experiment_utils import as_list
from utils.dataset_reference_metrics import dataset_metric_summary


DATASET_NAME = "chameleon"
NUM_SPLIT_RUNS = 25
BINNING_GRID = [50, 100, 150, 200, 250, 300, 350 , 400]

USE_EXISTING_RESULTS = False
RESULTS_CSV_PATH = SRC_ROOT / "logs/sklearn_skf_hyperparams/0623-1054_StratifiedKFoldBinningSearch_Cora.csv"

CONFIG_PATH = SRC_ROOT / "conf/config.yaml"
OUTPUT_DIR = SRC_ROOT / "outputs/stratification_diagnostics"
LOG_DIR = SRC_ROOT / "logs/sklearn_skf_hyperparams"
SAVE_FIGURE = True

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


def run_binning_search(dataset_name, num_split_runs, binning_grid=None, output_path=None):
    cfg = OmegaConf.load(CONFIG_PATH)
    properties = as_list(cfg.get("properties", ["Degree"]))
    binning_grid = binning_grid or BINNING_GRID

    timestamp = datetime.now().strftime("%m%d-%H%M")
    output_path = Path(output_path) if output_path else LOG_DIR / (
        f"{timestamp}_StratifiedKFoldBinningSearch_{dataset_name}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    run_cfg.log_fold_statistics = False
    run_cfg.plot_fold_statistics = False
    run_cfg.fold_stat_properties = list(BaseNodeStratifier.PROPERTY_NAMES)
    run_cfg.sampling_method = "sklearn"

    _, _, _, data = DatasetFactory.get_dataset(
        name=dataset_name,
        root_dir=str(SRC_ROOT / "data"),
    )
    props = WDESKFold(
        cfg=run_cfg,
        dataset_name=dataset_name,
        seed=0,
        n_splits=run_cfg.num_folds,
        property_name=properties[0],
    )._compute_node_properties(data)

    emd_columns = list(BaseNodeStratifier.PROPERTY_COLUMN_NAMES.values())
    header = [
        "Dataset",
        "TargetProperty",
        "StratSeed",
        "RequestedBins",
        "StratifiedKFoldSeconds",
        "RandomTargetEmd",
        "LabelTargetEmd",
        "StratifiedKFoldTargetEmd",
        "RandomToStratifiedKFoldRatio",
        "LabelToStratifiedKFoldRatio",
        *emd_columns,
    ]

    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for strat_seed in range(num_split_runs):
            print(f"Preparing baselines for seed {strat_seed + 1}/{num_split_runs}")
            random_summary = run_baseline(RandomKFold, run_cfg, dataset_name, strat_seed, data)
            label_summary = run_baseline(LabelStratifiedKFold, run_cfg, dataset_name, strat_seed, data)

            for requested_bins in binning_grid:
                run_cfg.skf_num_bins = [requested_bins]
                for property_name in properties:
                    print(
                        f"Running bins={requested_bins} | {property_name} | "
                        f"seed {strat_seed + 1}/{num_split_runs}"
                    )
                    stratifier = run_property_stratified_kfold(
                        run_cfg,
                        dataset_name,
                        strat_seed,
                        data,
                        property_name,
                        props=props,
                    )

                    target_column = target_column_for_property(property_name)
                    emd_summary = stratifier.last_fold_emd_summary
                    random_target_emd = random_summary[target_column]
                    label_target_emd = label_summary[target_column]
                    stratified_target_emd = emd_summary[target_column]

                    writer.writerow([
                        dataset_name,
                        property_name,
                        strat_seed,
                        requested_bins,
                        stratifier.optimization_seconds,
                        random_target_emd,
                        label_target_emd,
                        stratified_target_emd,
                        random_target_emd / max(stratified_target_emd, EPS),
                        label_target_emd / max(stratified_target_emd, EPS),
                        *[emd_summary[column_name] for column_name in emd_columns],
                    ])
                    file.flush()

    return pd.read_csv(output_path), output_path


def load_existing_results(path):
    path = resolve_results_path(path)
    if not path.exists():
        raise FileNotFoundError(f"StratifiedKFold binning results CSV not found: {path}")

    results = pd.read_csv(path)
    required_columns = {
        "Dataset",
        "TargetProperty",
        "StratSeed",
        "RequestedBins",
        "RandomTargetEmd",
        "LabelTargetEmd",
        "StratifiedKFoldTargetEmd",
    }
    missing_columns = required_columns - set(results.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing_columns)}")

    dataset_names = sorted(results["Dataset"].dropna().unique())
    dataset_name = dataset_names[0] if len(dataset_names) == 1 else DATASET_NAME
    num_split_runs = int(results["StratSeed"].nunique())
    binning_grid = sorted(map(int, results["RequestedBins"].drop_duplicates()))
    return results, path, dataset_name, num_split_runs, binning_grid


def make_matrices(results, requested_bins):
    rows = ["RandomKFold", "LabelStratifiedKFold"]
    properties = list(results["TargetProperty"].drop_duplicates())
    absolute_values = pd.DataFrame(index=rows, columns=properties, dtype=float)
    ratio_values = pd.DataFrame(index=rows, columns=properties, dtype=float)
    absolute_annotations = pd.DataFrame(index=rows, columns=properties, dtype=str)
    ratio_annotations = pd.DataFrame(index=rows, columns=properties, dtype=str)

    subset = results[results["RequestedBins"].astype(int) == int(requested_bins)]
    for property_name in properties:
        property_rows = subset[subset["TargetProperty"] == property_name]
        random_emd = property_rows["RandomTargetEmd"].mean()
        label_emd = property_rows["LabelTargetEmd"].mean()
        stratified_emd = property_rows["StratifiedKFoldTargetEmd"].mean()

        absolute_values.loc["RandomKFold", property_name] = random_emd
        absolute_values.loc["LabelStratifiedKFold", property_name] = label_emd
        absolute_annotations.loc["RandomKFold", property_name] = (
            f"base {random_emd:.2e}\nSKF {stratified_emd:.2e}"
        )
        absolute_annotations.loc["LabelStratifiedKFold", property_name] = (
            f"base {label_emd:.2e}\nSKF {stratified_emd:.2e}"
        )

        random_ratio = random_emd / max(stratified_emd, EPS)
        label_ratio = label_emd / max(stratified_emd, EPS)
        ratio_values.loc["RandomKFold", property_name] = random_ratio
        ratio_values.loc["LabelStratifiedKFold", property_name] = label_ratio
        ratio_annotations.loc["RandomKFold", property_name] = f"{random_ratio:.2f}x"
        ratio_annotations.loc["LabelStratifiedKFold", property_name] = f"{label_ratio:.2f}x"

    renamed_columns = [PROPERTY_LABELS.get(column, column) for column in properties]
    for frame in [absolute_values, ratio_values, absolute_annotations, ratio_annotations]:
        frame.columns = renamed_columns

    return absolute_values, ratio_values, absolute_annotations, ratio_annotations


def plot_results(results, dataset_name, num_split_runs, binning_grid):
    fig, axes = plt.subplots(
        nrows=len(binning_grid),
        ncols=2,
        figsize=(22, 5.2 * len(binning_grid)),
        dpi=220,
        constrained_layout=True,
        squeeze=False,
    )

    fig.suptitle(
        f"{dataset_name}: StratifiedKFold binning study | {dataset_metric_summary(dataset_name)} | "
        f"n={num_split_runs} split runs per method",
        fontsize=20,
        fontweight="bold",
    )

    for row_idx, requested_bins in enumerate(binning_grid):
        absolute_values, ratio_values, absolute_annotations, ratio_annotations = make_matrices(
            results,
            requested_bins,
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
            f"Absolute EMD | requested bins={requested_bins}",
            fontsize=13,
        )
        axes[row_idx, 0].set_xlabel("Stratified property")
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
            cbar_kws={"label": "Baseline mean EMD / StratifiedKFold mean EMD"} if row_idx == 0 else None,
            linewidths=0.5,
            linecolor="white",
        )
        axes[row_idx, 1].set_title(
            f"Reduction ratio | requested bins={requested_bins}",
            fontsize=13,
        )
        axes[row_idx, 1].set_xlabel("Stratified property")
        axes[row_idx, 1].set_ylabel("Baseline")

    if SAVE_FIGURE:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%m%d-%H%M")
        output_base = OUTPUT_DIR / (
            f"{timestamp}_StratifiedKFoldBinningStudy_{dataset_name}_n{num_split_runs}"
        )
        fig.savefig(f"{output_base}.png", dpi=300)
        fig.savefig(f"{output_base}.pdf")
        print(f"Saved plot to: {output_base}.png")
        print(f"Saved vector plot to: {output_base}.pdf")

    plt.show()


def main():
    if USE_EXISTING_RESULTS:
        results, output_path, dataset_name, num_split_runs, binning_grid = load_existing_results(
            RESULTS_CSV_PATH
        )
        print(f"Loaded StratifiedKFold binning search from: {output_path}")
    else:
        results, output_path = run_binning_search(
            DATASET_NAME,
            NUM_SPLIT_RUNS,
            BINNING_GRID,
        )
        dataset_name = DATASET_NAME
        num_split_runs = NUM_SPLIT_RUNS
        binning_grid = BINNING_GRID
        print(f"Saved StratifiedKFold binning search to: {output_path}")

    plot_results(results, dataset_name, num_split_runs, binning_grid)


if __name__ == "__main__":
    main()
