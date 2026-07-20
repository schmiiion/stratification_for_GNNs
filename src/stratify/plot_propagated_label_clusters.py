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
    cluster_label_distributions,
    compute_effective_min_cluster_size,
    compute_propagated_label_distribution,
    select_gap_statistic_cluster_counts,
)
from utils.dataset_reference_metrics import dataset_metric_summary
from utils.experiment_utils import as_list, dataset_suffix


# Main switches for this diagnostic plot.
DATASET_NAME = "Cora"
STRAT_SEED = 0
MAX_TSNE_NODES = 5000
SAVE_FIGURE = False

CONFIG_PATH = SRC_ROOT / "conf/config.yaml"
OUTPUT_ROOT = SRC_ROOT / "logs/runs"


def cfg_get(cfg, key, default):
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def load_dataset(dataset_name):
    canonical_name = str(dataset_name).replace("_", "-").lower()
    if canonical_name.startswith("syn-cora-h"):
        return DatasetFactory.get_dataset(name=dataset_name)
    return DatasetFactory.get_dataset(name=dataset_name)


def select_cluster_count(cfg, distributions, strat_seed):
    selection_method = str(
        cfg_get(cfg, "propagated_label_cluster_selection", "fixed")
    ).replace("-", "_").lower()

    if selection_method in {"gap", "gap_topk", "gap_statistic"}:
        selected = select_gap_statistic_cluster_counts(
            distributions=distributions,
            seed=strat_seed,
            reference_runs=cfg_get(cfg, "propagated_label_gap_reference_runs", 5),
            min_cluster_size=cfg_get(cfg, "propagated_label_min_cluster_size", 25),
            min_cluster_fraction=cfg_get(cfg, "propagated_label_min_cluster_fraction", 0.005),
            num_folds=cfg_get(cfg, "num_folds", 5),
            min_nodes_per_fold=cfg_get(cfg, "propagated_label_min_nodes_per_fold", 5),
            min_k=cfg_get(cfg, "propagated_label_gap_min_k", 2),
            max_k=cfg_get(cfg, "propagated_label_gap_max_k", 50),
            show_progress=cfg_get(cfg, "propagated_label_gap_progress", True),
            progress_label="Gap statistic for propagated-label t-SNE",
        )
        return int(selected[0])

    configured = cfg_get(cfg, "propagated_label_num_clusters", 50)
    return int(as_list(configured)[0])


def sample_nodes(num_nodes, max_tsne_nodes, strat_seed):
    if max_tsne_nodes is None or num_nodes <= int(max_tsne_nodes):
        return np.arange(num_nodes)

    rng = np.random.default_rng(strat_seed)
    return np.sort(rng.choice(num_nodes, size=int(max_tsne_nodes), replace=False))


def fit_tsne(distributions, strat_seed):
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError(
            "This diagnostic requires scikit-learn. Run it with the project "
            "environment, e.g. graph_stratification."
        ) from exc

    perplexity = min(30, max(1, (len(distributions) - 1) // 3))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=strat_seed,
    ).fit_transform(distributions)


def scatter(ax, embedding, colors, title, cmap):
    points = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        s=7,
        alpha=0.78,
        linewidths=0,
        cmap=cmap,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return points


def plot_propagated_label_clusters(
    cfg,
    dataset_name,
    data,
    strat_seed=STRAT_SEED,
    max_tsne_nodes=MAX_TSNE_NODES,
    save_figure=SAVE_FIGURE,
    output_dir=None,
    show=True,
    selected_k=None,
):
    distributions = compute_propagated_label_distribution(
        data=data,
        num_hops=cfg_get(cfg, "propagated_label_num_hops", 3),
        decay=cfg_get(cfg, "propagated_label_decay", 0.5),
    )
    if selected_k is None:
        selected_k = select_cluster_count(cfg, distributions, strat_seed)
    selected_k = int(selected_k)

    effective_min_cluster_size = compute_effective_min_cluster_size(
        num_nodes=int(data.num_nodes),
        min_cluster_size=cfg_get(cfg, "propagated_label_min_cluster_size", 25),
        min_cluster_fraction=cfg_get(cfg, "propagated_label_min_cluster_fraction", 0.005),
        num_folds=cfg_get(cfg, "num_folds", 5),
        min_nodes_per_fold=cfg_get(cfg, "propagated_label_min_nodes_per_fold", 5),
    )
    cluster_ids = cluster_label_distributions(
        distributions=distributions,
        num_clusters=selected_k,
        seed=strat_seed,
        min_cluster_size=effective_min_cluster_size,
    )

    sample_idx = sample_nodes(len(distributions), max_tsne_nodes, strat_seed)
    sampled_distributions = distributions[sample_idx]
    print(f"Computing t-SNE for {dataset_name} on {len(sample_idx)} propagated-label vectors...")
    embedding = fit_tsne(sampled_distributions, strat_seed)
    label_colors = data.y.detach().cpu().numpy()[sample_idx]
    cluster_colors = cluster_ids[sample_idx]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=180, constrained_layout=True)
    label_points = scatter(
        axes[0],
        embedding,
        label_colors,
        "Before clustering\ncolor = node label",
        "tab20",
    )
    cluster_points = scatter(
        axes[1],
        embedding,
        cluster_colors,
        f"After KMeans clustering\ncolor = cluster id, selected k={selected_k}",
        "turbo",
    )
    fig.colorbar(label_points, ax=axes[0], shrink=0.78, label="Node label")
    fig.colorbar(cluster_points, ax=axes[1], shrink=0.78, label="Cluster id")

    plotted_text = (
        f"plotted nodes={len(sample_idx)}/{int(data.num_nodes)}"
        if len(sample_idx) < int(data.num_nodes)
        else f"nodes={int(data.num_nodes)}"
    )
    fig.suptitle(
        f"{dataset_name}: propagated-label t-SNE | "
        f"{dataset_metric_summary(dataset_name, data)}\n"
        f"hops={cfg_get(cfg, 'propagated_label_num_hops', 3)} | "
        f"decay={cfg_get(cfg, 'propagated_label_decay', 0.5)} | "
        f"selected k={selected_k} | seed={strat_seed} | {plotted_text}",
        fontsize=15,
        fontweight="bold",
    )

    if save_figure:
        if output_dir is None:
            output_dir = OUTPUT_ROOT / dataset_suffix([dataset_name])
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"PropagatedLabelClusters_{dataset_name}_k{selected_k}.png"
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved figure to: {output_path}")

    if show:
        plt.show()

    return fig, selected_k


def main():
    cfg = OmegaConf.load(CONFIG_PATH)
    _, _, _, data = load_dataset(DATASET_NAME)
    plot_propagated_label_clusters(
        cfg=cfg,
        dataset_name=DATASET_NAME,
        data=data,
        strat_seed=STRAT_SEED,
        max_tsne_nodes=MAX_TSNE_NODES,
        save_figure=SAVE_FIGURE,
        show=True,
    )


if __name__ == "__main__":
    main()
