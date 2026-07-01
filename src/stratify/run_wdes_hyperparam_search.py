from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import sys

from omegaconf import OmegaConf
import numpy as np
import pandas as pd


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from factories.dataset_factory import DatasetFactory
from stratify.baseclass import BaseNodeStratifier
from stratify.label_based_stratifier import LabelStratifiedKFold
from stratify.random_stratifier import RandomKFold
from stratify.wdes_stratifier import WDESKFold
from utils.experiment_utils import as_list


CONFIG_PATH = SRC_ROOT / "conf/config.yaml"
OUTPUT_DIR = SRC_ROOT / "logs/wdes_hyperparams"
EPS = 1e-12

WDES_HYPERPARAM_GRID = [
    # {"Name": "pop50_gen20", "wdes_n_pop": 50, "wdes_n_gen": 20},
    # {"Name": "pop100_gen50", "wdes_n_pop": 100, "wdes_n_gen": 50},
    # {"Name": "pop200_gen50", "wdes_n_pop": 200, "wdes_n_gen": 50},
    # {"Name": "pop100_gen100", "wdes_n_pop": 100, "wdes_n_gen": 100},
    {"Name": "pop100_gen200", "wdes_n_pop": 100, "wdes_n_gen": 500},
]


def target_column_for_property(property_name):
    normalized_name = str(property_name).replace("_", "").replace(" ", "").lower()
    canonical_name = BaseNodeStratifier.PROPERTY_ALIASES.get(normalized_name, property_name)
    return BaseNodeStratifier.PROPERTY_COLUMN_NAMES[canonical_name]


def run_baseline(stratifier_class, cfg, dataset_name, seed, data):
    stratifier = stratifier_class(
        cfg=cfg,
        dataset_name=dataset_name,
        seed=seed,
        n_splits=cfg.num_folds,
    )
    stratifier.get_folds(data)
    return stratifier.last_fold_emd_summary


def run_property_stratified_kfold(cfg, dataset_name, seed, data, property_name, props=None):
    cfg.sampling_method = "sklearn"
    stratifier = WDESKFold(
        cfg=cfg,
        dataset_name=dataset_name,
        seed=seed,
        n_splits=cfg.num_folds,
        property_name=property_name,
    )

    if props is None:
        stratifier.get_folds(data)
        return stratifier

    stratifier.property_values = props[property_name]
    stratifier.num_nodes = data.num_nodes
    stratifier.target_fold_counts = np.array(
        [len(bucket) for bucket in np.array_split(np.arange(stratifier.num_nodes), stratifier.n_splits)],
        dtype=np.int64,
    )
    fold_buckets = stratifier._sklearn_buckets()
    folds = stratifier._masks_from_fold_buckets(fold_buckets, stratifier.num_nodes)
    stratifier.last_fold_emd_summary = stratifier.compute_fold_bucket_emd_summary(data, folds, props)
    return stratifier


def add_stratified_kfold_comparison(results, dataset_name=None):
    needed_columns = {
        "StratifiedKFoldTargetEmd",
        "RandomToStratifiedKFoldRatio",
        "LabelToStratifiedKFoldRatio",
        "StratifiedKFoldToWdesRatio",
    }
    if needed_columns.issubset(results.columns):
        return results

    cfg = OmegaConf.load(CONFIG_PATH)
    run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    run_cfg.log_fold_statistics = False
    run_cfg.plot_fold_statistics = False
    run_cfg.fold_stat_properties = list(BaseNodeStratifier.PROPERTY_NAMES)
    run_cfg.sampling_method = "sklearn"

    dataset_names = list(results["Dataset"].dropna().unique())
    if dataset_name is None:
        if len(dataset_names) != 1:
            raise ValueError("Pass dataset_name when enriching results with multiple datasets.")
        dataset_name = dataset_names[0]

    _, _, _, data = DatasetFactory.get_dataset(
        name=dataset_name,
        root_dir=str(SRC_ROOT / "data"),
    )
    props = WDESKFold(
        cfg=run_cfg,
        dataset_name=dataset_name,
        seed=0,
        n_splits=run_cfg.num_folds,
        property_name=results["TargetProperty"].iloc[0],
    )._compute_node_properties(data)

    stratified_target_emd = {}
    stratified_seconds = {}
    keys = results[["TargetProperty", "StratSeed"]].drop_duplicates()
    for _, row in keys.iterrows():
        property_name = row["TargetProperty"]
        strat_seed = int(row["StratSeed"])
        stratifier = run_property_stratified_kfold(
            run_cfg,
            dataset_name,
            strat_seed,
            data,
            property_name,
            props=props,
        )
        target_column = target_column_for_property(property_name)
        key = (property_name, strat_seed)
        stratified_target_emd[key] = stratifier.last_fold_emd_summary[target_column]
        stratified_seconds[key] = stratifier.optimization_seconds

    results = results.copy()
    results["StratifiedKFoldTargetEmd"] = results.apply(
        lambda row: stratified_target_emd[(row["TargetProperty"], int(row["StratSeed"]))],
        axis=1,
    )
    results["StratifiedKFoldSeconds"] = results.apply(
        lambda row: stratified_seconds[(row["TargetProperty"], int(row["StratSeed"]))],
        axis=1,
    )
    results["RandomToStratifiedKFoldRatio"] = (
        results["RandomTargetEmd"] / results["StratifiedKFoldTargetEmd"].clip(lower=EPS)
    )
    results["LabelToStratifiedKFoldRatio"] = (
        results["LabelTargetEmd"] / results["StratifiedKFoldTargetEmd"].clip(lower=EPS)
    )
    results["StratifiedKFoldToWdesRatio"] = (
        results["StratifiedKFoldTargetEmd"] / results["WdesTargetEmd"].clip(lower=EPS)
    )
    return results


def run_search(dataset_name, num_split_runs, hyperparam_grid=None, output_path=None):
    cfg = OmegaConf.load(CONFIG_PATH)
    properties = as_list(cfg.get("properties", ["Degree"]))
    hyperparam_grid = hyperparam_grid or WDES_HYPERPARAM_GRID

    timestamp = datetime.now().strftime("%m%d-%H%M")
    output_path = Path(output_path) if output_path else OUTPUT_DIR / (
        f"{timestamp}_WdesHyperparamSearch_{dataset_name}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    run_cfg.log_fold_statistics = False
    run_cfg.plot_fold_statistics = False
    run_cfg.fold_stat_properties = list(BaseNodeStratifier.PROPERTY_NAMES)
    run_cfg.sampling_method = "ga"

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
        "HyperparamName",
        "Population",
        "Generations",
        "EvolutionSeconds",
        "StratifiedKFoldSeconds",
        "RandomTargetEmd",
        "LabelTargetEmd",
        "StratifiedKFoldTargetEmd",
        "WdesTargetEmd",
        "RandomToWdesRatio",
        "LabelToWdesRatio",
        "RandomToStratifiedKFoldRatio",
        "LabelToStratifiedKFoldRatio",
        "StratifiedKFoldToWdesRatio",
        *emd_columns,
    ]

    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for strat_seed in range(num_split_runs):
            print(f"Preparing baselines for seed {strat_seed + 1}/{num_split_runs}")
            random_summary = run_baseline(RandomKFold, run_cfg, dataset_name, strat_seed, data)
            label_summary = run_baseline(LabelStratifiedKFold, run_cfg, dataset_name, strat_seed, data)
            stratified_summaries = {}
            stratified_seconds = {}

            run_cfg.sampling_method = "sklearn"
            for property_name in properties:
                stratifier = run_property_stratified_kfold(
                    run_cfg,
                    dataset_name,
                    strat_seed,
                    data,
                    property_name,
                    props=props,
                )
                stratified_summaries[property_name] = stratifier.last_fold_emd_summary
                stratified_seconds[property_name] = stratifier.optimization_seconds

            for hyperparams in hyperparam_grid:
                for property_name in properties:
                    run_cfg.sampling_method = "ga"
                    run_cfg.wdes_n_pop = hyperparams["wdes_n_pop"]
                    run_cfg.wdes_n_gen = hyperparams["wdes_n_gen"]

                    print(
                        f"Running {hyperparams['Name']} | {property_name} | "
                        f"seed {strat_seed + 1}/{num_split_runs}"
                    )
                    stratifier = WDESKFold(
                        cfg=run_cfg,
                        dataset_name=dataset_name,
                        seed=strat_seed,
                        n_splits=run_cfg.num_folds,
                        property_name=property_name,
                    )
                    stratifier.get_folds(data)

                    target_column = target_column_for_property(stratifier.property_name)
                    emd_summary = stratifier.last_fold_emd_summary
                    random_target_emd = random_summary[target_column]
                    label_target_emd = label_summary[target_column]
                    stratified_target_emd = stratified_summaries[property_name][target_column]
                    wdes_target_emd = emd_summary[target_column]

                    writer.writerow([
                        dataset_name,
                        stratifier.property_name,
                        strat_seed,
                        hyperparams["Name"],
                        hyperparams["wdes_n_pop"],
                        hyperparams["wdes_n_gen"],
                        stratifier.optimization_seconds,
                        stratified_seconds[property_name],
                        random_target_emd,
                        label_target_emd,
                        stratified_target_emd,
                        wdes_target_emd,
                        random_target_emd / max(wdes_target_emd, EPS),
                        label_target_emd / max(wdes_target_emd, EPS),
                        random_target_emd / max(stratified_target_emd, EPS),
                        label_target_emd / max(stratified_target_emd, EPS),
                        stratified_target_emd / max(wdes_target_emd, EPS),
                        *[emd_summary[column_name] for column_name in emd_columns],
                    ])
                    file.flush()

    return pd.read_csv(output_path), output_path


def main() -> None:
    dataset_name = "chameleon"
    num_split_runs = 5
    results, output_path = run_search(dataset_name, num_split_runs)
    print(f"\nSaved WDES hyperparameter search to: {output_path}")
    print(results.groupby("HyperparamName")["RandomToWdesRatio"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
