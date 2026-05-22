from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import sys

from omegaconf import OmegaConf
import pandas as pd


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from factories.dataset_factory import DatasetFactory
from factories.stratifier_factory import get_stratifiers
from stratify.baseclass import BaseNodeStratifier
from utils.experiment_utils import as_list


NUM_DIAGNOSTIC_RUNS = 100
CONFIG_PATH = SRC_ROOT / "conf/config.yaml"
OUTPUT_DIR = SRC_ROOT / "logs"


def main() -> None:
    cfg = OmegaConf.load(CONFIG_PATH)
    datasets = as_list(cfg.get("datasets", []))
    stratification_types = as_list(cfg.get("stratification_types", []))
    if not datasets:
        raise ValueError("No datasets configured. Add at least one dataset to config.yaml.")
    if not stratification_types:
        raise ValueError("No stratification_types configured.")

    run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    run_cfg.log_fold_statistics = False
    run_cfg.plot_fold_statistics = False

    timestamp = datetime.now().strftime("%m%d-%H%M")
    output_path = OUTPUT_DIR / f"{timestamp}_StratificationDiagnostics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    emd_columns = list(BaseNodeStratifier.PROPERTY_COLUMN_NAMES.values())
    header = [
        "Dataset",
        "StratificationMethod",
        "StratSeed",
        "EvolutionStart",
        "EvolutionStop",
        "EvolutionSeconds",
        *emd_columns,
    ]

    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for dataset_id in datasets:
            print(f"\nDATASET: {dataset_id}")
            _, _, _, data = DatasetFactory.get_dataset(name=dataset_id)

            for strat_seed in range(NUM_DIAGNOSTIC_RUNS):
                stratifiers = get_stratifiers(
                    cfg=run_cfg,
                    dataset_name=dataset_id,
                    seed=strat_seed,
                )
                for stratifier in stratifiers:
                    print(
                        f"Running {stratifier.stratification_method} | "
                        f"seed {strat_seed + 1}/{NUM_DIAGNOSTIC_RUNS}"
                    )
                    stratifier.get_folds(data)
                    emd_summary = stratifier.last_fold_emd_summary

                    writer.writerow([
                        dataset_id,
                        stratifier.stratification_method,
                        strat_seed,
                        getattr(stratifier, "optimization_start_time", ""),
                        getattr(stratifier, "optimization_stop_time", ""),
                        getattr(stratifier, "optimization_seconds", ""),
                        *[emd_summary[column_name] for column_name in emd_columns],
                    ])
                    file.flush()

    print(f"\nSaved stratification diagnostics to: {output_path}")

    results = pd.read_csv(output_path)
    print("\nMean +/- std of fold-bucket EMD to the full dataset distribution:")
    for stratification_method, group in results.groupby("StratificationMethod"):
        print(f"\n{stratification_method}")
        for column_name in emd_columns:
            values = pd.to_numeric(group[column_name], errors="coerce").dropna()
            if values.empty:
                continue
            print(f"  {column_name}: {values.mean():.6f} +/- {values.std(ddof=1):.6f}")


if __name__ == "__main__":
    main()
