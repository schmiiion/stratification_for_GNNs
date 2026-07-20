from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from factories.dataset_factory import DatasetFactory
from stratify.propagated_label_distribution import (
    compute_gap_statistic_curve,
    compute_propagated_label_distribution,
    select_gap_statistic_cluster_count_from_curve,
)
from utils.dataset_reference_metrics import dataset_metric_summary


# Main switches for standalone usage.
DATASET_NAME = "Cora"
STRAT_SEED = 0
SAVE_FIGURE = False

CONFIG_PATH = SRC_ROOT / "conf/config.yaml"
OUTPUT_DIR = SRC_ROOT / "outputs/stratification_diagnostics"


def cfg_get(cfg, key, default):
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def load_dataset(dataset_name):
    return DatasetFactory.get_dataset(name=dataset_name)


def compute_curve_from_data(cfg, data, strat_seed):
    distributions = compute_propagated_label_distribution(
        data=data,
        num_hops=cfg_get(cfg, "propagated_label_num_hops", 3),
        decay=cfg_get(cfg, "propagated_label_decay", 0.5),
    )
    return compute_gap_statistic_curve(
        distributions=distributions,
        seed=strat_seed,
        reference_runs=cfg_get(cfg, "propagated_label_gap_reference_runs", 5),
        min_cluster_size=cfg_get(cfg, "propagated_label_min_cluster_size", 25),
        min_cluster_fraction=cfg_get(cfg, "propagated_label_min_cluster_fraction", 0.005),
        num_folds=cfg_get(cfg, "num_folds", 5),
        min_nodes_per_fold=cfg_get(cfg, "propagated_label_min_nodes_per_fold", 5),
        min_k=cfg_get(cfg, "propagated_label_gap_min_k", 2),
        max_k=cfg_get(cfg, "propagated_label_gap_max_k", 50),
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
    show=True,
):
    curve = compute_curve_from_data(cfg, data, strat_seed)
    if not curve:
        raise ValueError("Cannot plot gap statistic curve because no k values were evaluated.")

    selected_k = select_gap_statistic_cluster_count_from_curve(curve)
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
        f"{dataset_name}: propagated-label gap statistic | "
        f"{dataset_metric_summary(dataset_name, data)}\n"
        f"hops={cfg_get(cfg, 'propagated_label_num_hops', 3)} | "
        f"decay={cfg_get(cfg, 'propagated_label_decay', 0.5)} | "
        f"reference runs={cfg_get(cfg, 'propagated_label_gap_reference_runs', 5)} | "
        f"k range={int(k_values[0])}-{int(k_values[-1])} | "
        f"selected k={selected_k} | {criterion_text}",
        fontsize=14,
        fontweight="bold",
    )

    if save_figure:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"GapStatistic_{dataset_name}_k{selected_k}.png"
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved figure to: {output_path}")

    if show:
        plt.show()

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
