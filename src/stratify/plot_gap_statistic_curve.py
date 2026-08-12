from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from factories.dataset_factory import DatasetFactory
from stratify.baseclass import BaseNodeStratifier
from stratify.propagated_label_distribution import (
    compute_gap_statistic_selection_curve,
    compute_neighborhood_label_counts,
    compute_propagated_label_distribution,
)
from utils.dataset_reference_metrics import dataset_metric_summary
from utils.experiment_utils import as_list, dataset_suffix


# Main switches for standalone usage.
DATASET_NAME = "Cora"
STRAT_SEED = 0
SAVE_FIGURE = False

CONFIG_PATH = SRC_ROOT / "conf/config.yaml"
OUTPUT_ROOT = SRC_ROOT / "logs/runs"

CLUSTERED_PLOT_CONFIGS = {
    "Propagated Label Cluster": {
        "label": "propagated-label",
        "title": "propagated-label",
        "prefix": "propagated_label",
        "compute_vectors": compute_propagated_label_distribution,
        "gap_filename_prefix": "GapStatistic",
        "cluster_filename_prefix": "PropagatedLabelPCA_TSNEClusters",
    },
    "Neighborhood Count": {
        "label": "neighborhood-count",
        "title": "neighborhood-count",
        "prefix": "neighborhood_count",
        "compute_vectors": compute_neighborhood_label_counts,
        "gap_filename_prefix": "GapStatistic_NeighborhoodCount",
        "cluster_filename_prefix": "NeighborhoodCountPCA_TSNEClusters",
    },
}


def cfg_get(cfg, key, default):
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def load_dataset(dataset_name):
    return DatasetFactory.get_dataset(name=dataset_name)


def canonical_clustered_property_name(property_name):
    canonical_name = BaseNodeStratifier.canonical_property_name(property_name)
    if canonical_name not in CLUSTERED_PLOT_CONFIGS:
        available = ", ".join(CLUSTERED_PLOT_CONFIGS)
        raise ValueError(f"Property '{property_name}' has no cluster plot support. Available: {available}")
    return canonical_name


def clustered_property_names_from_cfg(cfg):
    property_names = []
    for property_name in as_list(cfg_get(cfg, "properties", ["Propagated Label Cluster"])):
        try:
            canonical_name = BaseNodeStratifier.canonical_property_name(property_name)
        except ValueError:
            continue
        if canonical_name in CLUSTERED_PLOT_CONFIGS and canonical_name not in property_names:
            property_names.append(canonical_name)
    return property_names


def clustered_property_name_from_cfg(cfg):
    property_names = clustered_property_names_from_cfg(cfg)
    if property_names:
        return property_names[0]
    return "Propagated Label Cluster"


def cluster_plot_config(property_name):
    return CLUSTERED_PLOT_CONFIGS[canonical_clustered_property_name(property_name)]


def compute_clustered_property_vectors(cfg, data, property_name):
    canonical_name = canonical_clustered_property_name(property_name)
    spec = cluster_plot_config(canonical_name)
    prefix = spec["prefix"]
    vector_kwargs = {
        "data": data,
        "num_hops": cfg_get(cfg, f"{prefix}_num_hops", 3),
        "decay": cfg_get(cfg, f"{prefix}_decay", 0.5),
    }
    if canonical_name == "Neighborhood Count":
        vector_kwargs["log_scale"] = cfg_get(cfg, "neighborhood_count_log_scale", False)
    return spec["compute_vectors"](**vector_kwargs)


def compute_curve_from_data(cfg, data, strat_seed, dataset_name="dataset", property_name=None):
    property_name = clustered_property_name_from_cfg(cfg) if property_name is None else property_name
    canonical_name = canonical_clustered_property_name(property_name)
    spec = cluster_plot_config(canonical_name)
    prefix = spec["prefix"]
    distributions = compute_clustered_property_vectors(cfg, data, canonical_name)
    return compute_gap_statistic_selection_curve(
        distributions=distributions,
        seed=strat_seed,
        reference_runs=cfg_get(cfg, f"{prefix}_gap_reference_runs", 5),
        min_cluster_size=cfg_get(cfg, f"{prefix}_min_cluster_size", 25),
        min_cluster_fraction=cfg_get(cfg, f"{prefix}_min_cluster_fraction", 0.005),
        num_folds=cfg_get(cfg, "num_folds", 5),
        min_nodes_per_fold=cfg_get(cfg, f"{prefix}_min_nodes_per_fold", 5),
        min_k=cfg_get(cfg, f"{prefix}_gap_min_k", 2),
        max_k=cfg_get(cfg, f"{prefix}_gap_max_k", 50),
        extra_after_selected=cfg_get(
            cfg,
            f"{prefix}_gap_plot_extra_k",
            cfg_get(cfg, "propagated_label_gap_plot_extra_k", 5),
        ),
        show_progress=cfg_get(cfg, f"{prefix}_gap_progress", True),
        progress_label=f"Gap curve {spec['label']} {dataset_name} seed={strat_seed}",
    )


def mark_selected_k(ax, selected_k):
    ax.axvline(selected_k, color="black", linestyle="--", linewidth=1.2, alpha=0.75)
    ax.text(
        selected_k,
        0.98,
        f" selected k={selected_k}",
        transform=ax.get_xaxis_transform(),
        va="top",
        ha="left",
        fontsize=10,
        fontweight="bold",
    )


def plot_gap_statistic_curve(
    cfg,
    dataset_name,
    data,
    strat_seed=STRAT_SEED,
    save_figure=SAVE_FIGURE,
    output_dir=None,
    show=True,
    property_name=None,
):
    property_name = clustered_property_name_from_cfg(cfg) if property_name is None else property_name
    canonical_name = canonical_clustered_property_name(property_name)
    spec = cluster_plot_config(canonical_name)
    prefix = spec["prefix"]
    curve, selected_k = compute_curve_from_data(
        cfg,
        data,
        strat_seed,
        dataset_name,
        canonical_name,
    )
    if not curve:
        raise ValueError("Cannot plot gap statistic curve because no k values were evaluated.")

    k_values = np.array([row["k"] for row in curve], dtype=int)
    gaps = np.array([row["gap"] for row in curve], dtype=float)
    reference_se = np.array([row["reference_se"] for row in curve], dtype=float)
    penalized_gaps = gaps - reference_se

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=180, constrained_layout=True)

    axes[0].errorbar(
        k_values,
        gaps,
        yerr=reference_se,
        color="#f59e0b",
        marker="o",
        markersize=3.5,
        linewidth=1.7,
        capsize=3,
    )
    axes[0].set_title("Gap statistic with standard-error bars", fontweight="bold")
    axes[0].set_xlabel("k clusters")
    axes[0].set_ylabel("Gap(k) = E_ref[log(W_k)] - log(W_k)")
    mark_selected_k(axes[0], selected_k)

    axes[1].plot(
        k_values,
        gaps,
        color="#2563eb",
        marker="o",
        markersize=3.5,
        linewidth=1.7,
        label="Gap(k)",
    )
    axes[1].plot(
        k_values,
        penalized_gaps,
        color="#dc2626",
        marker="o",
        markersize=3.5,
        linewidth=1.7,
        label="Gap(k) - s'_k",
    )
    axes[1].set_title("Decision rule penalty curve", fontweight="bold")
    axes[1].set_xlabel("k clusters")
    axes[1].set_ylabel("Gap value")
    axes[1].legend(frameon=False)
    mark_selected_k(axes[1], selected_k)

    if selected_k < int(k_values[-1]):
        next_index = int(np.where(k_values == selected_k + 1)[0][0])
        axes[1].scatter(
            [selected_k + 1],
            [penalized_gaps[next_index]],
            color="#dc2626",
            s=70,
            zorder=5,
        )
        criterion_text = (
            f"Rule: Gap({selected_k}) >= Gap({selected_k + 1}) - s'_{selected_k + 1}"
        )
    else:
        criterion_text = "Rule reached the configured maximum k."

    for ax in axes:
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"{dataset_name}: {spec['title']} gap statistic | "
        f"{dataset_metric_summary(dataset_name, data)}\n"
        f"hops={cfg_get(cfg, f'{prefix}_num_hops', 3)} | "
        f"decay={cfg_get(cfg, f'{prefix}_decay', 0.5)} | "
        f"reference runs={cfg_get(cfg, f'{prefix}_gap_reference_runs', 5)} | "
        f"plotted k={int(k_values[0])}-{int(k_values[-1])} | "
        f"selected k={selected_k} | {criterion_text}",
        fontsize=14,
        fontweight="bold",
    )

    if save_figure:
        if output_dir is None:
            output_dir = OUTPUT_ROOT / dataset_suffix([dataset_name])
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{spec['gap_filename_prefix']}_{dataset_name}_k{selected_k}.png"
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved figure to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, selected_k, curve


def main():
    cfg = OmegaConf.load(CONFIG_PATH)
    _, _, _, data = load_dataset(DATASET_NAME)
    plot_gap_statistic_curve(
        cfg=cfg,
        dataset_name=DATASET_NAME,
        data=data,
        strat_seed=STRAT_SEED,
        save_figure=SAVE_FIGURE,
        show=True,
    )


if __name__ == "__main__":
    main()
