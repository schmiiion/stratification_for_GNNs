from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.dataset_reference_metrics import (
    canonical_dataset_key,
    dataset_graph_size_text,
    dataset_homophily_text,
    dataset_li_text,
)
from utils.experiment_utils import as_list


CUMULATIVE_RESULTS_FILE = SRC_ROOT / "logs/runs/cumulative_results.csv"
CONFIG_FILE = SRC_ROOT / "conf/config.yaml"

# Leave empty to plot every dataset found in the cumulative results file.
DATASETS_TO_PLOT = []
INCLUDE_SYN_CORA = False
DEDUPLICATE_SEEDED_RUNS = True
EXPECTED_NUM_FOLDS = 5
MAX_SANITY_EXAMPLES = 8
MODEL_ORDER = ["GCN", "GAT", "SAGE", "GPRGNN", "H2GCN", "MLP"]

PROPERTY_ALIASES = {
    "degree": "Degree",
    "neighborhoodheterogeneity": "Neighborhood Heterogeneity",
    "neighborhoodhomophily": "Neighborhood Heterogeneity",
    "neighhet": "Neighborhood Heterogeneity",
    "pagerank": "PageRank",
    "eigenvectorcentrality": "Eigenvector Centrality",
    "eigencentrality": "Eigenvector Centrality",
    "clusteringcoefficient": "Clustering Coefficient",
    "clustercoeff": "Clustering Coefficient",
    "clustering": "Clustering Coefficient",
    "propagatedlabelcluster": "Propagated Label Cluster",
    "propagatedlabelclusters": "Propagated Label Cluster",
    "propagatedlabeldistribution": "Propagated Label Cluster",
    "labelpropagationcluster": "Propagated Label Cluster",
    "propagatedlabel": "Propagated Label Cluster",
    "proplabelcluster": "Propagated Label Cluster",
    "neighborhoodcount": "Neighborhood Count",
    "neighborhoodcounts": "Neighborhood Count",
    "neighborhoodlabelcount": "Neighborhood Count",
    "neighborhoodlabelcounts": "Neighborhood Count",
    "neighcount": "Neighborhood Count",
    "neighcounts": "Neighborhood Count",
}
PROPERTY_TO_METHOD = {
    "Degree": "Degree",
    "Neighborhood Heterogeneity": "Neighborhood Heterogeneity",
    "PageRank": "PageRank",
    "Eigenvector Centrality": "Eigenvector Centrality",
    "Clustering Coefficient": "Clustering Coefficient",
    "Propagated Label Cluster": "Neighborhood Distribution",
    "Neighborhood Count": "Neighborhood Count",
}
METHOD_LABELS = {
    "Random": "Random",
    "Label": "Label",
    "Degree": "Degree",
    "Neighborhood Heterogeneity": "Neigh.\nHom.",
    "PageRank": "PageRank",
    "Eigenvector Centrality": "Eigen.\nCentrality",
    "Clustering Coefficient": "Clustering\nCoeff.",
    "Neighborhood Distribution": "Neigh.\nDistribution",
    "Neighborhood Count": "Neigh.\nCount",
}


def normalized_key(value):
    return str(value).replace(" ", "").replace("_", "").replace("-", "").lower()


def canonical_property_name(property_name):
    key = normalized_key(property_name)
    if key not in PROPERTY_ALIASES:
        return None
    return PROPERTY_ALIASES[key]


def load_config():
    if not CONFIG_FILE.exists():
        return OmegaConf.create({})
    return OmegaConf.load(CONFIG_FILE)


def configured_stratification_keys(cfg):
    return {
        str(stratification_type).replace("-", "_").lower()
        for stratification_type in as_list(cfg.get("stratification_types", []))
    }


def property_stratification_is_enabled(cfg):
    property_keys = {"property", "property_stratified", "wdes"}
    return bool(configured_stratification_keys(cfg) & property_keys)


def configured_sampling_method(cfg):
    key = normalized_key(cfg.get("sampling_method", "sklearn"))
    if key in {"ga", "wdes"}:
        return "ga"
    return "sklearn"


def configured_method_order(cfg):
    methods = []
    stratification_keys = configured_stratification_keys(cfg)

    if "random" in stratification_keys:
        methods.append("Random")
    if "label" in stratification_keys:
        methods.append("Label")

    if property_stratification_is_enabled(cfg):
        for property_name in as_list(cfg.get("properties", [])):
            canonical_name = canonical_property_name(property_name)
            if canonical_name is None:
                continue
            method = PROPERTY_TO_METHOD[canonical_name]
            if method not in methods:
                methods.append(method)

    return methods


CONFIG = load_config()
METHOD_ORDER = configured_method_order(CONFIG)
ACTIVE_SAMPLING_METHOD = configured_sampling_method(CONFIG)
PROPERTY_STRATIFICATION_ENABLED = property_stratification_is_enabled(CONFIG)


DATASET_DISPLAY_NAMES = {
    "actor": "Actor",
    "amazon-computers": "Computers",
    "amazon-photo": "Photo",
    "amazon-ratings": "amazon_ratings",
    "chameleon": "chameleon",
    "citeseer": "CiteSeer",
    "coauthor-cs": "CoauthorCS",
    "coauthor-physics": "CoauthorPhysics",
    "cora": "Cora",
    "cornell": "cornell",
    "crocodile": "crocodile",
    "photo": "Photo",
    "pubmed": "PubMed",
    "roman-empire": "roman_empire",
    "squirrel": "squirrel",
    "texas": "Texas",
    "wikics": "WikiCS",
    "wisconsin": "wisconsin",
}


def display_dataset_name(dataset_name):
    key = canonical_dataset_key(dataset_name)
    if key.startswith("syn-cora"):
        return str(dataset_name)
    return DATASET_DISPLAY_NAMES.get(key, str(dataset_name))


def method_from_stratifier(stratifier):
    stratifier = str(stratifier)
    if stratifier == "RandomKFold":
        return "Random"
    if stratifier == "LabelStratifiedKFold":
        return "Label"
    if "NeighCount" in stratifier or "NeighborhoodCount" in stratifier:
        return "Neighborhood Count"
    if "PropLabelCluster" in stratifier:
        return "Neighborhood Distribution"
    if stratifier == "WDES_NeighHet" or stratifier.startswith("Sklearn_NeighHet"):
        return "Neighborhood Heterogeneity"
    if stratifier.startswith("StratifiedKFoldDynamic_NeighHet"):
        return "Neighborhood Heterogeneity"
    if "PageRank" in stratifier:
        return "PageRank"
    if "EigCentrality" in stratifier or "Eigenvector" in stratifier:
        return "Eigenvector Centrality"
    if "Clustering" in stratifier:
        return "Clustering Coefficient"
    if "Degree" in stratifier:
        return "Degree"
    return None


def sampling_method_from_stratifier(stratifier):
    stratifier = str(stratifier)
    if stratifier.startswith("WDES_"):
        return "ga"
    if (
        stratifier.startswith("Sklearn")
        or stratifier.startswith("StratifiedKFoldDynamic")
        or stratifier.startswith("StratifiedKFoldCategorical")
    ):
        return "sklearn"
    return None


def row_matches_config(row):
    method = row["Method"]
    if method not in METHOD_ORDER:
        return False

    stratification_keys = configured_stratification_keys(CONFIG)
    if method == "Random":
        return "random" in stratification_keys
    if method == "Label":
        return "label" in stratification_keys

    if not PROPERTY_STRATIFICATION_ENABLED:
        return False
    raw_sampling_method = sampling_method_from_stratifier(row["StratificationType"])
    return raw_sampling_method == ACTIVE_SAMPLING_METHOD


def load_cumulative_results():
    if not CUMULATIVE_RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"{CUMULATIVE_RESULTS_FILE} does not exist. "
            "Run src/one_time_scripts/build_cumulative_results.py first."
        )

    df = pd.read_csv(CUMULATIVE_RESULTS_FILE)
    required = {
        "Dataset",
        "Model",
        "StratificationType",
        "Fold_Seed",
        "Init_Seed",
        "Fold",
        "Test_Accuracy",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{CUMULATIVE_RESULTS_FILE} is missing required columns: {sorted(missing)}"
        )

    df["DatasetDisplay"] = df["Dataset"].map(display_dataset_name)
    df["DatasetKey"] = df["Dataset"].map(canonical_dataset_key)
    df["Method"] = df["StratificationType"].map(method_from_stratifier)
    df = df[df["Method"].notna()].copy()
    df = df[df.apply(row_matches_config, axis=1)].copy()
    if df.empty:
        raise ValueError("No rows matched the requested stratification methods.")

    df["AccuracyPercent"] = pd.to_numeric(df["Test_Accuracy"], errors="raise")
    if df["AccuracyPercent"].max() <= 1.0:
        df["AccuracyPercent"] *= 100.0

    if DEDUPLICATE_SEEDED_RUNS:
        dedup_keys = [
            "DatasetKey",
            "Model",
            "StratificationType",
            "Fold_Seed",
            "Init_Seed",
            "Fold",
        ]
        duplicate_groups = (
            df.groupby(dedup_keys, as_index=False, dropna=False)
            .agg(
                RowCount=("AccuracyPercent", "size"),
                AccuracyCount=("AccuracyPercent", "nunique"),
            )
        )
        conflicting_duplicates = duplicate_groups[
            (duplicate_groups["RowCount"] > 1)
            & (duplicate_groups["AccuracyCount"] > 1)
        ]
        if not conflicting_duplicates.empty:
            print(
                "WARNING: Found duplicated seeded rows with different accuracies. "
                "Keeping the first occurrence. Examples:"
            )
            print(conflicting_duplicates.head(MAX_SANITY_EXAMPLES).to_string(index=False))

        before = len(df)
        df = df.drop_duplicates(subset=dedup_keys, keep="first").copy()
        removed = before - len(df)
        if removed:
            print(f"Removed {removed} duplicated seeded run rows before aggregation.")

    return df


def summarize_fold_dispersion(df):
    exact_estimates = (
        df.groupby(
            [
                "DatasetKey",
                "DatasetDisplay",
                "Model",
                "Method",
                "StratificationType",
                "Fold_Seed",
                "Init_Seed",
            ],
            as_index=False,
        )
        .agg(
            FoldStd=("AccuracyPercent", "std"),
            FoldCount=("AccuracyPercent", "size"),
            UniqueFolds=("Fold", "nunique"),
        )
    )

    invalid_fold_groups = exact_estimates[
        (exact_estimates["FoldCount"] != EXPECTED_NUM_FOLDS)
        | (exact_estimates["UniqueFolds"] != EXPECTED_NUM_FOLDS)
    ]
    if not invalid_fold_groups.empty:
        print(
            f"WARNING: Dropping {len(invalid_fold_groups)} seeded groups that do not "
            f"contain exactly {EXPECTED_NUM_FOLDS} unique folds. Examples:"
        )
        print(invalid_fold_groups.head(MAX_SANITY_EXAMPLES).to_string(index=False))

    exact_estimates = exact_estimates[
        (exact_estimates["FoldCount"] == EXPECTED_NUM_FOLDS)
        & (exact_estimates["UniqueFolds"] == EXPECTED_NUM_FOLDS)
    ].copy()

    seeded_estimates = (
        exact_estimates.groupby(
            ["DatasetKey", "DatasetDisplay", "Model", "Method", "Fold_Seed", "Init_Seed"],
            as_index=False,
        )
        .agg(
            SeedPairFoldStd=("FoldStd", "mean"),
            RawStratificationVariants=("StratificationType", "nunique"),
        )
    )

    multi_variant_groups = seeded_estimates[seeded_estimates["RawStratificationVariants"] > 1]
    if not multi_variant_groups.empty:
        print(
            "INFO: Some displayed methods contain multiple raw StratificationType variants "
            "for the same Fold_Seed/Init_Seed. Their fold stds are averaged before counting n. "
            "Examples:"
        )
        print(multi_variant_groups.head(MAX_SANITY_EXAMPLES).to_string(index=False))

    return (
        seeded_estimates.groupby(
            ["DatasetKey", "DatasetDisplay", "Model", "Method"],
            as_index=False,
        )
        .agg(
            MeanFoldStd=("SeedPairFoldStd", "mean"),
            NumEstimates=("SeedPairFoldStd", "size"),
            NumFoldSeeds=("Fold_Seed", "nunique"),
            NumInitSeeds=("Init_Seed", "nunique"),
        )
    )


def dataset_sort_key(dataset_name):
    key = canonical_dataset_key(dataset_name)
    preferred = [
        "cora",
        "citeseer",
        "pubmed",
        "amazon-computers",
        "amazon-photo",
        "chameleon",
        "squirrel",
        "wisconsin",
        "texas",
        "actor",
        "roman-empire",
        "cornell",
        "amazon-ratings",
        "coauthor-cs",
        "coauthor-physics",
        "wikics",
    ]
    if key in preferred:
        return (0, preferred.index(key), str(dataset_name))
    return (1, str(dataset_name).lower(), str(dataset_name))


def displayed_methods(dataset_stats):
    return METHOD_ORDER


def plot_dataset_table(dataset_name, stats):
    dataset_stats = stats[stats["DatasetDisplay"] == dataset_name]
    models = [model for model in MODEL_ORDER if model in set(dataset_stats["Model"])]
    methods = displayed_methods(dataset_stats)
    if not models:
        return

    table_rows = []
    bold_cells = set()
    for row_idx, model in enumerate(models, start=1):
        model_stats = dataset_stats[dataset_stats["Model"] == model]
        best_std = model_stats["MeanFoldStd"].min()
        row = [model]

        for col_idx, method in enumerate(methods, start=1):
            values = model_stats[model_stats["Method"] == method]
            if values.empty:
                row.append("n/a")
                continue

            fold_std = float(values.iloc[0]["MeanFoldStd"])
            num_estimates = int(values.iloc[0]["NumEstimates"])
            row.append(f"Fold Std {fold_std:.2f}\nn={num_estimates}")
            if abs(fold_std - best_std) < 1e-12:
                bold_cells.add((row_idx, col_idx))

        table_rows.append(row)

    n_values = sorted(dataset_stats["NumEstimates"].dropna().astype(int).unique())
    if len(n_values) == 1:
        n_text = f"n={n_values[0]}"
    else:
        n_text = f"n={n_values[0]}-{n_values[-1]}"

    title_line = (
        f"{dataset_name}: Aggregated Accuracy Dispersion | "
        f"{dataset_homophily_text(dataset_name)} | {n_text}"
    )
    metadata_line = f"{dataset_graph_size_text(dataset_name)} | {dataset_li_text(dataset_name)}"

    fig, ax = plt.subplots(figsize=(13.5, max(6.5, 0.85 * len(models) + 2.8)))
    ax.axis("off")
    fig.text(0.5, 0.965, title_line, ha="center", va="top", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.915, metadata_line, ha="center", va="top", fontsize=12.5, fontweight="bold")

    table = ax.table(
        cellText=table_rows,
        colLabels=["Model", *[METHOD_LABELS[method] for method in methods]],
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.04, 0.08, 0.92, 0.74],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.85)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.visible_edges = "horizontal"
        cell.set_edgecolor("#333333")
        cell.set_linewidth(0.6)
        cell.get_text().set_wrap(True)
        cell.get_text().set_linespacing(1.05)

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

    print(f"Rendering {dataset_name}: {len(dataset_stats)} aggregated cells")
    plt.show()


def main():
    df = load_cumulative_results()
    stats = summarize_fold_dispersion(df)

    datasets = [display_dataset_name(dataset) for dataset in DATASETS_TO_PLOT]
    if not datasets:
        datasets = sorted(stats["DatasetDisplay"].unique(), key=dataset_sort_key)
    if not INCLUDE_SYN_CORA:
        datasets = [
            dataset for dataset in datasets
            if not canonical_dataset_key(dataset).startswith("syn-cora")
        ]

    print(f"Loaded {len(df)} relevant rows from {CUMULATIVE_RESULTS_FILE}.")
    print("Included methods:", ", ".join(METHOD_ORDER))
    print("Datasets:", ", ".join(datasets))

    for dataset_name in datasets:
        if dataset_name not in set(stats["DatasetDisplay"]):
            print(f"Skipping {dataset_name}: no matching rows.")
            continue
        plot_dataset_table(dataset_name, stats)


if __name__ == "__main__":
    main()
