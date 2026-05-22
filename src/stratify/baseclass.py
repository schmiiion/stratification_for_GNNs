from abc import ABC, abstractmethod
import csv

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance
import seaborn as sns
import torch
from torch_geometric.utils import degree, to_networkx


class BaseNodeStratifier(ABC):
    """
    Base class for transductive node-classification splits.

    Concrete stratifiers only decide how nodes are assigned to equally sized
    buckets. The base class turns those buckets into rotating train/val/test
    masks and keeps the shared fold diagnostics in one place.
    """

    split_colors = {
        "Train": "tab:blue",
        "Val": "tab:green",
        "Test": "tab:orange",
    }

    PROPERTY_NAMES = (
        "Degree",
        "Neighborhood Heterogeneity",
        "PageRank",
        "Eigenvector Centrality",
        "Clustering Coefficient",
    )

    PROPERTY_COLUMN_NAMES = {
        "Degree": "DegreeEmd",
        "Neighborhood Heterogeneity": "NeighHetEmd",
        "PageRank": "PageRankEmd",
        "Eigenvector Centrality": "EigCentralityEmd",
        "Clustering Coefficient": "ClusteringEmd",
    }

    PROPERTY_METHOD_NAMES = {
        "Degree": "Degree",
        "Neighborhood Heterogeneity": "NeighHet",
        "PageRank": "PageRank",
        "Eigenvector Centrality": "EigCentrality",
        "Clustering Coefficient": "Clustering",
    }

    PROPERTY_ALIASES = {
        "degree": "Degree",
        "neighborhoodheterogeneity": "Neighborhood Heterogeneity",
        "neighborhoodhomophily": "Neighborhood Heterogeneity",
        "neighhet": "Neighborhood Heterogeneity",
        "pagerank": "PageRank",
        "eigenvectorcentrality": "Eigenvector Centrality",
        "eigencentrality": "Eigenvector Centrality",
        "eigenveccent": "Eigenvector Centrality",
        "clusteringcoefficient": "Clustering Coefficient",
        "clustercoeff": "Clustering Coefficient",
        "clustering": "Clustering Coefficient",
    }

    def __init__(self, cfg, dataset_name, seed, n_splits=5):
        if n_splits != 5:
            raise ValueError("n_splits is fixed to 5 in our setting")

        self.cfg = cfg
        self.dataset_name = dataset_name
        self.n_splits = n_splits
        self.seed = seed
        self.stratification_method = self.__class__.__name__

    @abstractmethod
    def get_folds(self, data):
        """Return a list of dicts with train_mask, val_mask, and test_mask."""

    def _masks_from_fold_buckets(self, fold_buckets, num_nodes):
        """
        Rotate through fold buckets to create train/val/test masks.

        With n_splits=5, each bucket is 20% of the data. For fold i, bucket i is
        test, bucket i+1 is validation, and the remaining three buckets are
        training. This gives the requested 60/20/20 split while preserving
        K-fold coverage of every node as test exactly once.
        """
        fold_buckets = [np.asarray(bucket, dtype=np.int64) for bucket in fold_buckets]
        self._validate_fold_buckets(fold_buckets, num_nodes)

        folds = []
        for test_bucket in range(self.n_splits):
            val_bucket = (test_bucket + 1) % self.n_splits
            train_buckets = [
                bucket_id
                for bucket_id in range(self.n_splits)
                if bucket_id not in {test_bucket, val_bucket}
            ]

            train_idx = np.concatenate([fold_buckets[bucket_id] for bucket_id in train_buckets])
            val_idx = fold_buckets[val_bucket]
            test_idx = fold_buckets[test_bucket]

            folds.append({
                "train_mask": self._mask_from_indices(train_idx, num_nodes),
                "val_mask": self._mask_from_indices(val_idx, num_nodes),
                "test_mask": self._mask_from_indices(test_idx, num_nodes),
            })

        return folds

    @staticmethod
    def _mask_from_indices(indices, num_nodes):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[torch.as_tensor(indices, dtype=torch.long)] = True
        return mask

    def _validate_fold_buckets(self, fold_buckets, num_nodes):
        if len(fold_buckets) != self.n_splits:
            raise ValueError(f"Expected {self.n_splits} fold buckets, got {len(fold_buckets)}.")

        all_indices = np.concatenate(fold_buckets)
        if len(all_indices) != num_nodes:
            raise ValueError("Fold buckets must cover every node exactly once.")

        unique_indices = np.unique(all_indices)
        if len(unique_indices) != num_nodes:
            raise ValueError("Fold buckets contain duplicate node indices.")

        if unique_indices[0] != 0 or unique_indices[-1] != num_nodes - 1:
            raise ValueError("Fold buckets contain node indices outside the graph.")

    def _compute_node_properties(self, data):
        """
        Precompute node-level properties used for fold diagnostics.
        """
        num_nodes = data.num_nodes

        row, col = data.edge_index
        deg = degree(col, num_nodes).cpu().numpy()

        y = data.y
        same_class_match = (y[row] == y[col]).float()
        same_class_neighbors = torch.zeros(num_nodes, device=data.edge_index.device)
        same_class_neighbors.scatter_add_(0, col, same_class_match)

        deg_tensor = torch.tensor(deg, device=data.edge_index.device)
        neighborhood_homophily = (same_class_neighbors / deg_tensor.clamp(min=1)).cpu().numpy()

        graph = to_networkx(data, to_undirected=True)
        pagerank = np.array(list(nx.pagerank(graph).values()))

        try:
            eigen = np.array(list(nx.eigenvector_centrality(graph, max_iter=1000).values()))
        except nx.PowerIterationFailedConvergence:
            print("Eigenvector centrality failed to converge. Defaulting to degree centrality.")
            eigen = np.array(list(nx.degree_centrality(graph).values()))

        clustering = np.array(list(nx.clustering(graph).values()))

        return {
            "Degree": deg,
            "Neighborhood Heterogeneity": neighborhood_homophily,
            "PageRank": pagerank,
            "Eigenvector Centrality": eigen,
            "Clustering Coefficient": clustering,
        }

    def get_fold_stat_property_names(self):
        configured_properties = self._cfg_get("fold_stat_properties", list(self.PROPERTY_NAMES))
        if isinstance(configured_properties, str):
            configured_properties = [configured_properties]

        property_names = []
        for property_name in configured_properties:
            canonical_name = self._canonical_property_name(property_name)
            if canonical_name not in property_names:
                property_names.append(canonical_name)

        return property_names

    def compute_fold_bucket_emd_summary(self, data, folds, props=None):
        if props is None:
            props = self._compute_node_properties(data)
        fold_bucket_masks = [fold["test_mask"].cpu().numpy() for fold in folds]
        emd_summary = {column_name: "" for column_name in self.PROPERTY_COLUMN_NAMES.values()}

        for prop_name in self.get_fold_stat_property_names():
            prop_data = props[prop_name]
            emd_values = []

            for fold_bucket_mask in fold_bucket_masks:
                fold_bucket_data = prop_data[fold_bucket_mask]
                emd_values.append(wasserstein_distance(fold_bucket_data, prop_data))

            emd_summary[self.PROPERTY_COLUMN_NAMES[prop_name]] = float(np.mean(emd_values))

        return emd_summary

    def _canonical_property_name(self, property_name):
        normalized_name = str(property_name).replace("_", "").replace(" ", "").lower()
        if normalized_name in self.PROPERTY_ALIASES:
            return self.PROPERTY_ALIASES[normalized_name]
        if property_name in self.PROPERTY_NAMES:
            return property_name

        available = ", ".join(self.PROPERTY_NAMES)
        raise ValueError(f"Unknown fold-stat property '{property_name}'. Available: {available}")

    def _cfg_get(self, key, default):
        if hasattr(self.cfg, "get"):
            return self.cfg.get(key, default)
        return getattr(self.cfg, key, default)

    def _analyze_distributions(self, data, folds, dataset_name):
        props = self._compute_node_properties(data)
        prop_names = self.get_fold_stat_property_names()
        split_specs = [
            ("Train", "train_mask"),
            ("Val", "val_mask"),
            ("Test", "test_mask"),
        ]

        should_log = self._cfg_get("log_fold_statistics", True)
        should_plot = self._cfg_get("plot_fold_statistics", False)

        if should_plot:
            fig, axes = plt.subplots(
                len(folds),
                len(prop_names),
                figsize=(25, 4 * len(folds)),
                squeeze=False,
            )
            fig.suptitle(
                f"{dataset_name} - Fold Property PDFs - {self.stratification_method}",
                fontsize=20,
            )

        self.last_fold_emd_summary = self.compute_fold_bucket_emd_summary(data, folds, props)
        if should_log:
            row = [
                dataset_name,
                self.stratification_method,
                self.seed,
                *[
                    self.last_fold_emd_summary[column_name]
                    for column_name in self.PROPERTY_COLUMN_NAMES.values()
                ],
            ]

            with open(self.cfg.fold_stats_csv_filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(row)

        if should_plot:
            for fold_idx, fold in enumerate(folds):
                masks = {name: fold[mask_key].cpu().numpy() for name, mask_key in split_specs}

                for prop_idx, prop_name in enumerate(prop_names):
                    prop_data = props[prop_name]
                    ax = axes[fold_idx, prop_idx]

                    for split_name, _ in split_specs:
                        split_data = prop_data[masks[split_name]]
                        if len(split_data) == 0:
                            continue
                        sns.kdeplot(
                            split_data,
                            ax=ax,
                            label=split_name,
                            fill=True,
                            alpha=0.3,
                            color=self.split_colors[split_name],
                            warn_singular=False,
                        )

                    if fold_idx == 0:
                        ax.set_title(prop_name, fontsize=14)
                    if prop_idx == 0:
                        ax.set_ylabel(f"Fold {fold_idx + 1}\nDensity", fontsize=12)
                    else:
                        ax.set_ylabel("")

                    if fold_idx == 0 and prop_idx == 0:
                        ax.legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
