# Homophily-Aware Stratification for GNN Evaluation

This repository contains experiments for studying whether graph-aware stratification can reduce the dispersion of node-classification test accuracies across cross-validation folds.

The code is built around PyTorch Geometric datasets, a shared 60/20/20 rotating split setup, and several stratification strategies:

- random K-fold buckets
- label-stratified K-fold buckets
- scalar node-property stratification with dynamic `StratifiedKFold` bin selection
- optional WDES / genetic optimization for scalar node properties
- propagated neighborhood-label distribution and neighborhood-count clustering

## Setup

Create the conda environment:

```bash
conda env create -f environment.yaml
conda activate graph_stratification
```

If your machine has OpenMP conflicts when using PyTorch and scikit-learn together, run experiments with:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python src/main.py
```

## Configuration

The main configuration lives in `src/conf/config.yaml`.

Important fields:

- `datasets`: datasets to run.
- `model_names`: model architectures.
- `stratification_types`: `random`, `label`, and/or `property`.
- `properties`: node properties used when `property` stratification is enabled.
- `sampling_method`: `sklearn` or `ga`.
- `skf_num_bins`: candidate bin counts for scalar sklearn stratification.
- `plot_gap_statistic_curve` and `plot_propagated_label_clusters`: optional diagnostics.

Run the main experiment with:

```bash
python src/main.py
```

The checked-in default config is intentionally small (`Cora`, `GCN`, one fold seed, one init seed) so a fresh clone can be smoke-tested before enabling the larger experiment lists.

## Data

Downloaded or manually placed datasets are intentionally not tracked by Git. The default data root is `src/data/`.

PyTorch Geometric datasets such as Cora, CiteSeer, PubMed, Amazon Computers, Amazon Photo, Coauthor CS/Physics, and WikiCS are downloaded automatically by PyG.

The custom NPZ datasets must be placed manually when used:

```text
src/data/Actor/actor.npz
src/data/chameleon_clean/chameleon_filtered.npz
src/data/squirrel_filtered/squirrel_filtered.npz
src/data/texas/texas.npz
src/data/wisconsin/wisconsin.npz
src/data/roman_empire/roman_empire.npz
src/data/cornell/cornell.npz
src/data/amazon_ratings/amazon_ratings.npz
src/data/crocodile/crocodile.npz
```

Synthetic Cora is downloaded from the H2GCN authors' public archive when `syn-cora` is requested.

## Outputs

Runtime logs, plots, generated CSV files, and local dataset caches are ignored. Main experiment outputs are written under:

```text
src/logs/runs/
```

Plot-only diagnostics are written under:

```text
src/logs/plots/
```

Generated LaTeX result tables are written under:

```text
src/outputs/latex_tables/
```

## Result Utilities

Useful one-off scripts:

```bash
python src/one_time_scripts/build_cumulative_results.py
python src/one_time_scripts/add_new_runs_to_cumulative_results.py
python src/one_time_scripts/plot_stddev_table_from_cumulativecsv.py
python src/one_time_scripts/latex_table_from_cumulativecsv.py
```

The cumulative result file expected by the plotting/table scripts is:

```text
src/logs/runs/cumulative_results.csv
```
