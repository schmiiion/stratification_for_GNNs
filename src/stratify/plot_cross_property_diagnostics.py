from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.dataset_reference_metrics import dataset_metric_summary

LOG_ROOT = SRC_ROOT / "logs"
WDES_LOG_DIR = LOG_ROOT / "wdes_hyperparams"
SKF_LOG_DIR = LOG_ROOT / "sklearn_skf_hyperparams"
OUTPUT_DIR = SRC_ROOT / "outputs/stratification_diagnostics"
EPS = 1e-12

# Main switch for this script.
DATASET_NAME = "chameleon"

# Use "RandomKFold" or "LabelStratifiedKFold".
BASELINE_METHOD = "RandomKFold"

# The GA result used in the left panel.
GA_GENERATIONS = 500

# Leave these as None to combine all available CSVs in the matching log folder.
WDES_RESULTS_CSV_PATHS = None
SKF_RESULTS_CSV_PATHS = None

SAVE_FIGURE = False

PROPERTY_TO_COLUMN = {
    "Degree": "DegreeEmd",
    "Neighborhood Heterogeneity": "NeighHetEmd",
    "PageRank": "PageRankEmd",
    "Eigenvector Centrality": "EigCentralityEmd",
    "Clustering Coefficient": "ClusteringEmd",
}

PROPERTY_LABELS = {
    "Degree": "Degree",
    "Neighborhood Heterogeneity": "Neigh. het.",
    "PageRank": "PageRank",
    "Eigenvector Centrality": "Eigenvector",
    "Clustering Coefficient": "Clustering",
}

BASELINE_TO_TARGET_COLUMN = {
    "RandomKFold": "RandomTargetEmd",
    "LabelStratifiedKFold": "LabelTargetEmd",
}


def resolve_paths(paths, folder, pattern):
    if paths is not None:
        resolved = []
        for path in paths:
            path = Path(path).expanduser()
            resolved.append(path if path.is_absolute() else SRC_ROOT.parent / path)
        return resolved

    candidates = [
        *folder.glob(pattern),
        *LOG_ROOT.glob(pattern),  # backwards compatibility with older flat logs
    ]
    return sorted(set(candidates))


def load_many_csvs(paths, required_columns, kind):
    if not paths:
        raise FileNotFoundError(f"No {kind} CSVs found.")

    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"{kind} CSV not found: {path}")
        frame = pd.read_csv(path)
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["Dataset"] == DATASET_NAME].copy()
    if combined.empty:
        raise ValueError(f"No {kind} rows found for dataset '{DATASET_NAME}'.")

    print(f"Loaded {len(paths)} {kind} CSV(s), {len(combined)} rows for {DATASET_NAME}.")
    return combined


def mean_baseline_emd(results):
    baseline_column = BASELINE_TO_TARGET_COLUMN[BASELINE_METHOD]
    baseline_emd = {}

    for property_name, emd_column in PROPERTY_TO_COLUMN.items():
        property_rows = results[results["TargetProperty"] == property_name]
        if property_rows.empty:
            raise ValueError(f"Missing rows for target property '{property_name}'.")
        baseline_emd[emd_column] = pd.to_numeric(
            property_rows[baseline_column],
            errors="coerce",
        ).mean()

    return pd.Series(baseline_emd)


def select_wdes_generation_500(results):
    available_generations = sorted(
        pd.to_numeric(results["Generations"], errors="coerce").dropna().unique()
    )
    results = results[pd.to_numeric(results["Generations"], errors="coerce") == GA_GENERATIONS].copy()
    if results.empty:
        raise ValueError(
            f"No WDES rows with Generations={GA_GENERATIONS}. "
            f"Available: {available_generations}"
        )

    selected_rows = []
    selected_settings = {}
    for property_name, property_rows in results.groupby("TargetProperty", sort=False):
        setting_scores = (
            property_rows
            .groupby(["HyperparamName", "Population", "Generations"], as_index=False)
            .agg(TargetEmd=("WdesTargetEmd", "mean"))
            .sort_values("TargetEmd", kind="mergesort")
        )
        best_setting = setting_scores.iloc[0]
        mask = (
            (property_rows["HyperparamName"] == best_setting["HyperparamName"])
            & (property_rows["Population"] == best_setting["Population"])
            & (property_rows["Generations"] == best_setting["Generations"])
        )
        selected_rows.append(property_rows[mask])
        selected_settings[property_name] = (
            f"{best_setting['HyperparamName']} "
            f"(pop={int(best_setting['Population'])}, gen={int(best_setting['Generations'])})"
        )

    return pd.concat(selected_rows, ignore_index=True), selected_settings


def select_best_skf_bins(results):
    selected_rows = []
    best_bins = {}

    for property_name, property_rows in results.groupby("TargetProperty", sort=False):
        bin_scores = (
            property_rows
            .groupby("RequestedBins", as_index=False)
            .agg(TargetEmd=("StratifiedKFoldTargetEmd", "mean"))
            .sort_values("TargetEmd", kind="mergesort")
        )
        best_bins[property_name] = int(bin_scores.iloc[0]["RequestedBins"])
        selected_rows.append(property_rows[property_rows["RequestedBins"] == best_bins[property_name]])

    return pd.concat(selected_rows, ignore_index=True), best_bins


def make_ratio_matrix(results, row_suffixes=None):
    emd_columns = list(PROPERTY_TO_COLUMN.values())
    baseline_emd = mean_baseline_emd(results)

    rows = []
    row_labels = []
    for property_name, property_rows in results.groupby("TargetProperty", sort=False):
        achieved_emd = property_rows[emd_columns].apply(pd.to_numeric, errors="coerce").mean()
        ratio = baseline_emd / achieved_emd.clip(lower=EPS)

        row = {}
        for measured_property, column_name in PROPERTY_TO_COLUMN.items():
            row[PROPERTY_LABELS[measured_property]] = ratio[column_name]

        off_target_values = [
            ratio[column_name]
            for measured_property, column_name in PROPERTY_TO_COLUMN.items()
            if measured_property != property_name
        ]
        row["Mean Other"] = sum(off_target_values) / len(off_target_values)
        rows.append(row)

        label = PROPERTY_LABELS.get(property_name, property_name)
        if row_suffixes and property_name in row_suffixes:
            label = f"{label}\n{row_suffixes[property_name]}"
        row_labels.append(label)

    return pd.DataFrame(rows, index=row_labels)


def load_wdes_matrix():
    required = {
        "Dataset",
        "TargetProperty",
        "StratSeed",
        "HyperparamName",
        "Population",
        "Generations",
        BASELINE_TO_TARGET_COLUMN[BASELINE_METHOD],
        "WdesTargetEmd",
        *PROPERTY_TO_COLUMN.values(),
    }
    paths = resolve_paths(
        WDES_RESULTS_CSV_PATHS,
        WDES_LOG_DIR,
        "*_WdesHyperparamSearch_*.csv",
    )
    results = load_many_csvs(paths, required, "WDES hyperparameter")
    selected_results, selected_settings = select_wdes_generation_500(results)

    n_runs = selected_results[["TargetProperty", "StratSeed"]].drop_duplicates()["StratSeed"].nunique()
    print("\nWDES settings used:")
    for property_name, setting in selected_settings.items():
        print(f"  {property_name}: {setting}")

    return make_ratio_matrix(selected_results), n_runs


def load_skf_matrix():
    required = {
        "Dataset",
        "TargetProperty",
        "StratSeed",
        "RequestedBins",
        BASELINE_TO_TARGET_COLUMN[BASELINE_METHOD],
        "StratifiedKFoldTargetEmd",
        *PROPERTY_TO_COLUMN.values(),
    }
    paths = resolve_paths(
        SKF_RESULTS_CSV_PATHS,
        SKF_LOG_DIR,
        "*_StratifiedKFoldBinningSearch_*.csv",
    )
    results = load_many_csvs(paths, required, "StratifiedKFold binning")
    selected_results, best_bins = select_best_skf_bins(results)
    row_suffixes = {
        property_name: f"bins={num_bins}"
        for property_name, num_bins in best_bins.items()
    }

    n_runs = selected_results[["TargetProperty", "StratSeed"]].drop_duplicates()["StratSeed"].nunique()
    print("\nBest SKF bins used:")
    for property_name, num_bins in best_bins.items():
        print(f"  {property_name}: {num_bins}")

    return make_ratio_matrix(selected_results, row_suffixes), n_runs


def plot_matrices(wdes_matrix, skf_matrix, wdes_runs, skf_runs):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(26, 8),
        dpi=220,
        constrained_layout=True,
    )

    max_distance_from_one = max(
        abs(float(wdes_matrix.min().min()) - 1.0),
        abs(float(wdes_matrix.max().max()) - 1.0),
        abs(float(skf_matrix.min().min()) - 1.0),
        abs(float(skf_matrix.max().max()) - 1.0),
        0.1,
    )
    vmin = max(0.0, 1.0 - max_distance_from_one)
    vmax = 1.0 + max_distance_from_one

    for ax, matrix, title in [
        (
            axes[0],
            wdes_matrix,
            f"WDES GA, generations={GA_GENERATIONS} | n={wdes_runs}",
        ),
        (
            axes[1],
            skf_matrix,
            f"StratifiedKFold, best bins per property | n={skf_runs}",
        ),
    ]:
        annotations = matrix.map(lambda value: f"{value:.2f}x")
        sns.heatmap(
            matrix,
            ax=ax,
            annot=annotations,
            fmt="",
            cmap="RdYlGn",
            center=1.0,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.6,
            linecolor="white",
            cbar_kws={"label": f"EMD reduction ratio vs {BASELINE_METHOD}"},
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Measured fold-bucket EMD property")
        ax.set_ylabel("Optimized stratification property")

    fig.suptitle(
        f"{DATASET_NAME}: cross-property stratification effect | "
        f"{dataset_metric_summary(DATASET_NAME)} | baseline={BASELINE_METHOD} | "
        f"value = baseline EMD / achieved EMD",
        fontsize=17,
        fontweight="bold",
    )

    if SAVE_FIGURE:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_base = OUTPUT_DIR / (
            f"CrossPropertyDiagnostics_{DATASET_NAME}_{BASELINE_METHOD}_GA{GA_GENERATIONS}_BestSKF"
        )
        fig.savefig(f"{output_base}.png", dpi=300)
        fig.savefig(f"{output_base}.pdf")
        print(f"Saved plot to: {output_base}.png")
        print(f"Saved vector plot to: {output_base}.pdf")

    plt.show()


def main():
    if BASELINE_METHOD not in BASELINE_TO_TARGET_COLUMN:
        raise ValueError("BASELINE_METHOD must be 'RandomKFold' or 'LabelStratifiedKFold'.")

    wdes_matrix, wdes_runs = load_wdes_matrix()
    skf_matrix, skf_runs = load_skf_matrix()
    plot_matrices(wdes_matrix, skf_matrix, wdes_runs, skf_runs)


if __name__ == "__main__":
    main()
