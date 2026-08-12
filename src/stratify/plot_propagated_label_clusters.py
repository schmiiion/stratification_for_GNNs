from pathlib import Path
from datetime import datetime
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from omegaconf import OmegaConf


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from factories.dataset_factory import DatasetFactory
from stratify.propagated_label_distribution import (
    cluster_label_distributions,
    compute_effective_min_cluster_size,
    select_gap_statistic_cluster_counts,
)
from stratify.plot_gap_statistic_curve import (
    canonical_clustered_property_name,
    cluster_plot_config,
    clustered_property_name_from_cfg,
    clustered_property_names_from_cfg,
    compute_clustered_property_vectors,
    plot_gap_statistic_curve,
)
from utils.dataset_reference_metrics import dataset_metric_summary
from utils.experiment_utils import as_list, dataset_suffix


# Main switches for this diagnostic plot.
DATASET_NAME = "Cora"
STRAT_SEED = 0
MAX_TSNE_NODES = 5000
SAVE_FIGURE = True

CONFIG_PATH = SRC_ROOT / "conf/config.yaml"
PLOT_OUTPUT_ROOT = SRC_ROOT / "logs/plots"


def cfg_get(cfg, key, default):
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def create_plot_output_dir(dataset_names):
    timestamp = datetime.now().strftime("%m%d-%H%M")
    suffix = dataset_suffix(dataset_names)
    output_dir = PLOT_OUTPUT_ROOT / f"{timestamp}_{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG_PATH, output_dir / "config.yaml")
    return output_dir


def create_dataset_plot_output_dir(run_output_dir, dataset_name):
    safe_dataset_name = str(dataset_name).replace("/", "_").replace(" ", "_")
    output_dir = Path(run_output_dir) / safe_dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def configured_dataset_requests(cfg):
    for dataset_name in as_list(cfg_get(cfg, "datasets", [DATASET_NAME])):
        canonical_name = str(dataset_name).replace("_", "-").lower()
        if canonical_name != "syn-cora":
            yield str(dataset_name), {"name": str(dataset_name)}
            continue

        ratios = cfg_get(cfg, "syn_cora_ratios", None)
        if ratios is None:
            ratios = cfg_get(cfg, "syn-cora-ratio", None)
        if ratios is None:
            ratios = cfg_get(cfg, "syn_cora_ratio", [0.70])

        realizations = cfg_get(cfg, "syn_cora_realizations", None)
        if realizations is None:
            realizations = cfg_get(cfg, "syn-cora-realizations", [1])

        for ratio in as_list(ratios):
            for realization in as_list(realizations):
                ratio_value = float(ratio)
                realization_value = int(realization)
                run_dataset_name = f"syn-cora-h{ratio_value:.2f}-r{realization_value}"
                yield run_dataset_name, {
                    "name": "syn-cora",
                    "syn_cora_homophily": ratio_value,
                    "syn_cora_realization": realization_value,
                }


def select_cluster_count(cfg, distributions, strat_seed, property_name):
    canonical_name = canonical_clustered_property_name(property_name)
    spec = cluster_plot_config(canonical_name)
    prefix = spec["prefix"]
    selection_method = str(
        cfg_get(cfg, f"{prefix}_cluster_selection", "fixed")
    ).replace("-", "_").lower()

    if selection_method in {"gap", "gap_topk", "gap_statistic"}:
        selected = select_gap_statistic_cluster_counts(
            distributions=distributions,
            seed=strat_seed,
            reference_runs=cfg_get(cfg, f"{prefix}_gap_reference_runs", 5),
            min_cluster_size=cfg_get(cfg, f"{prefix}_min_cluster_size", 25),
            min_cluster_fraction=cfg_get(cfg, f"{prefix}_min_cluster_fraction", 0.005),
            num_folds=cfg_get(cfg, "num_folds", 5),
            min_nodes_per_fold=cfg_get(cfg, f"{prefix}_min_nodes_per_fold", 5),
            min_k=cfg_get(cfg, f"{prefix}_gap_min_k", 2),
            max_k=cfg_get(cfg, f"{prefix}_gap_max_k", 50),
            show_progress=cfg_get(cfg, f"{prefix}_gap_progress", True),
            progress_label=f"Gap statistic for {spec['label']} t-SNE",
        )
        return int(selected[0])

    configured = cfg_get(cfg, f"{prefix}_num_clusters", 50)
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
        init="random",
        learning_rate="auto",
        random_state=strat_seed,
        n_jobs=1,
    ).fit_transform(distributions)


def fit_pca(distributions):
    distributions = np.asarray(distributions, dtype=float)
    centered = distributions - distributions.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    component_order = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, component_order[:2]]
    embedding = centered @ components
    if embedding.shape[1] < 2:
        embedding = np.column_stack([embedding[:, 0], np.zeros(len(embedding))])
    return embedding


def categorical_scatter(ax, embedding, values, title, cmap_name):
    values = np.asarray(values, dtype=np.int64)
    categories = np.unique(values)
    encoded_values = np.searchsorted(categories, values)

    base_cmap = plt.get_cmap(cmap_name)
    listed_categorical_maps = {
        "tab10",
        "tab20",
        "tab20b",
        "tab20c",
        "Set1",
        "Set2",
        "Set3",
        "Accent",
        "Dark2",
        "Paired",
        "Pastel1",
        "Pastel2",
    }
    if (
        cmap_name in listed_categorical_maps
        and hasattr(base_cmap, "colors")
        and len(base_cmap.colors) >= len(categories)
    ):
        color_values = base_cmap.colors[:len(categories)]
    else:
        color_values = base_cmap(np.linspace(0.05, 0.95, max(1, len(categories))))

    cmap = ListedColormap(color_values)
    norm = BoundaryNorm(
        np.arange(len(categories) + 1) - 0.5,
        cmap.N,
    )
    points = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=encoded_values,
        s=7,
        alpha=0.78,
        linewidths=0,
        cmap=cmap,
        norm=norm,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return points, categories


def add_categorical_colorbar(fig, points, ax, categories, label):
    ticks = np.arange(len(categories))
    colorbar = fig.colorbar(
        points,
        ax=ax,
        shrink=0.78,
        ticks=ticks,
        boundaries=np.arange(len(categories) + 1) - 0.5,
    )
    colorbar.ax.set_yticklabels([str(int(category)) for category in categories])
    colorbar.set_label(label)
    if len(categories) > 20:
        colorbar.ax.tick_params(labelsize=6)
    return colorbar


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
    property_name=None,
):
    property_name = clustered_property_name_from_cfg(cfg) if property_name is None else property_name
    canonical_name = canonical_clustered_property_name(property_name)
    spec = cluster_plot_config(canonical_name)
    prefix = spec["prefix"]
    distributions = compute_clustered_property_vectors(cfg, data, canonical_name)
    if selected_k is None:
        selected_k = select_cluster_count(cfg, distributions, strat_seed, canonical_name)
    selected_k = int(selected_k)
    max_tsne_nodes = cfg_get(
        cfg,
        f"{prefix}_tsne_max_nodes",
        cfg_get(cfg, "propagated_label_tsne_max_nodes", max_tsne_nodes),
    )

    effective_min_cluster_size = compute_effective_min_cluster_size(
        num_nodes=int(data.num_nodes),
        min_cluster_size=cfg_get(cfg, f"{prefix}_min_cluster_size", 25),
        min_cluster_fraction=cfg_get(cfg, f"{prefix}_min_cluster_fraction", 0.005),
        num_folds=cfg_get(cfg, "num_folds", 5),
        min_nodes_per_fold=cfg_get(cfg, f"{prefix}_min_nodes_per_fold", 5),
    )
    cluster_ids = cluster_label_distributions(
        distributions=distributions,
        num_clusters=selected_k,
        seed=strat_seed,
        min_cluster_size=effective_min_cluster_size,
    )

    labels = data.y.detach().cpu().numpy()

    print(
        f"Computing PCA for {dataset_name} on {len(distributions)} {spec['label']} vectors...",
        flush=True,
    )
    pca_embedding = fit_pca(distributions)
    print(f"Finished PCA for {dataset_name}.", flush=True)

    tsne_idx = sample_nodes(len(distributions), max_tsne_nodes, strat_seed)
    tsne_distributions = distributions[tsne_idx]
    print(
        f"Computing t-SNE for {dataset_name} on {len(tsne_idx)} {spec['label']} vectors...",
        flush=True,
    )
    tsne_embedding = fit_tsne(tsne_distributions, strat_seed)
    print(f"Finished t-SNE for {dataset_name}.", flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=180, constrained_layout=True)
    pca_label_points, pca_label_categories = categorical_scatter(
        axes[0, 0],
        pca_embedding,
        labels,
        "PCA before clustering\ncolor = node label",
        "tab20",
    )
    pca_cluster_points, pca_cluster_categories = categorical_scatter(
        axes[0, 1],
        pca_embedding,
        cluster_ids,
        f"PCA after KMeans clustering\ncolor = cluster id, selected k={selected_k}",
        "tab20",
    )
    tsne_label_points, tsne_label_categories = categorical_scatter(
        axes[1, 0],
        tsne_embedding,
        labels[tsne_idx],
        "t-SNE before clustering\ncolor = node label",
        "tab20",
    )
    tsne_cluster_points, tsne_cluster_categories = categorical_scatter(
        axes[1, 1],
        tsne_embedding,
        cluster_ids[tsne_idx],
        f"t-SNE after KMeans clustering\ncolor = cluster id, selected k={selected_k}",
        "tab20",
    )
    add_categorical_colorbar(fig, pca_label_points, axes[0, 0], pca_label_categories, "Node label")
    add_categorical_colorbar(fig, pca_cluster_points, axes[0, 1], pca_cluster_categories, "Cluster id")
    add_categorical_colorbar(fig, tsne_label_points, axes[1, 0], tsne_label_categories, "Node label")
    add_categorical_colorbar(fig, tsne_cluster_points, axes[1, 1], tsne_cluster_categories, "Cluster id")

    tsne_text = (
        f"t-SNE nodes={len(tsne_idx)}/{int(data.num_nodes)}"
        if len(tsne_idx) < int(data.num_nodes)
        else f"t-SNE nodes={int(data.num_nodes)}"
    )
    fig.suptitle(
        f"{dataset_name}: {spec['title']} PCA and t-SNE | "
        f"{dataset_metric_summary(dataset_name, data)}\n"
        f"hops={cfg_get(cfg, f'{prefix}_num_hops', 3)} | "
        f"decay={cfg_get(cfg, f'{prefix}_decay', 0.5)} | "
        f"selected k={selected_k} | seed={strat_seed} | PCA nodes={int(data.num_nodes)} | {tsne_text}",
        fontsize=15,
        fontweight="bold",
    )

    if save_figure:
        if output_dir is None:
            output_dir = create_plot_output_dir([dataset_name])
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{spec['cluster_filename_prefix']}_{dataset_name}_k{selected_k}.png"
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved figure to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, selected_k


def main():
    cfg = OmegaConf.load(CONFIG_PATH)
    dataset_requests = list(configured_dataset_requests(cfg))
    if not dataset_requests:
        raise ValueError("No datasets configured in src/conf/config.yaml.")

    fold_seeds = as_list(cfg_get(cfg, "fold_seeds", [STRAT_SEED]))
    strat_seed = int(fold_seeds[0]) if fold_seeds else STRAT_SEED
    output_dir = create_plot_output_dir([dataset_name for dataset_name, _ in dataset_requests])

    clustered_properties = clustered_property_names_from_cfg(cfg)
    if not clustered_properties:
        raise ValueError("No configured properties support PCA/t-SNE clustering plots.")

    print(
        "Creating PCA/t-SNE cluster plots for configured datasets: "
        f"{', '.join(dataset_name for dataset_name, _ in dataset_requests)}"
    )
    print(f"Clustered properties: {', '.join(clustered_properties)}")
    print(f"Using stratification seed {strat_seed}.")
    print(f"Saving figures to: {output_dir}")

    for dataset_name, dataset_kwargs in dataset_requests:
        print(f"\n{'=' * 40}\nDATASET: {dataset_name}\n{'=' * 40}")
        try:
            _, _, _, data = DatasetFactory.get_dataset(**dataset_kwargs)
        except Exception as exc:
            print(f"Skipping {dataset_name}: {exc}", flush=True)
            continue

        dataset_output_dir = create_dataset_plot_output_dir(output_dir, dataset_name)
        print(f"Saving {dataset_name} plots to: {dataset_output_dir}")

        for property_name in clustered_properties:
            try:
                _, selected_k, _ = plot_gap_statistic_curve(
                    cfg=cfg,
                    dataset_name=dataset_name,
                    data=data,
                    strat_seed=strat_seed,
                    save_figure=SAVE_FIGURE,
                    output_dir=dataset_output_dir,
                    show=False,
                    property_name=property_name,
                )
            except Exception as exc:
                print(
                    f"Skipping {dataset_name} {property_name} plots because "
                    f"gap-statistic plotting failed: {exc}",
                    flush=True,
                )
                continue

            plot_propagated_label_clusters(
                cfg=cfg,
                dataset_name=dataset_name,
                data=data,
                strat_seed=strat_seed,
                max_tsne_nodes=cfg_get(cfg, "propagated_label_tsne_max_nodes", MAX_TSNE_NODES),
                save_figure=SAVE_FIGURE,
                output_dir=dataset_output_dir,
                show=False,
                selected_k=selected_k,
                property_name=property_name,
            )


if __name__ == "__main__":
    main()
