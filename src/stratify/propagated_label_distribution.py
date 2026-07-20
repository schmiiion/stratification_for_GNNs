import numpy as np
import torch
import torch.nn.functional as F


def compute_propagated_label_distribution(data, num_hops=3, decay=0.5):
    """
    Compute a decayed k-hop neighborhood label distribution for every node.

    The returned matrix has shape [num_nodes, num_classes]. Row v contains the
    label mass that reaches node v through 1..num_hops propagation steps. The
    node's own label at hop 0 is intentionally excluded.
    """
    num_hops = int(num_hops)
    decay = float(decay)
    if num_hops < 1:
        raise ValueError("propagated_label_num_hops must be at least 1.")
    if decay <= 0:
        raise ValueError("propagated_label_decay must be positive.")

    labels = data.y.detach().to(torch.long)
    _, labels = torch.unique(labels, sorted=True, return_inverse=True)
    num_nodes = int(data.num_nodes)
    num_classes = int(labels.max().item()) + 1

    edge_index = data.edge_index.detach().to(labels.device)
    source, target = edge_index[0], edge_index[1]

    current = F.one_hot(labels, num_classes=num_classes).float()
    accumulated = torch.zeros((num_nodes, num_classes), device=labels.device)
    deg = torch.bincount(target, minlength=num_nodes).float().clamp(min=1.0)

    for hop in range(num_hops):
        propagated = torch.zeros_like(current)
        propagated.index_add_(0, target, current[source])
        propagated = propagated / deg[:, None]

        weight = decay ** hop
        accumulated = accumulated + weight * propagated
        current = propagated

    row_sums = accumulated.sum(dim=1, keepdim=True)
    has_signal = row_sums.squeeze(1) > 0
    accumulated[has_signal] = accumulated[has_signal] / row_sums[has_signal]

    if not bool(has_signal.all()):
        global_prior = F.one_hot(labels, num_classes=num_classes).float().mean(dim=0)
        accumulated[~has_signal] = global_prior

    return accumulated.cpu().numpy()


def _compact_cluster_ids(cluster_ids):
    _, compact_ids = np.unique(cluster_ids, return_inverse=True)
    return compact_ids.astype(np.int64)


def compute_effective_min_cluster_size(
    num_nodes,
    min_cluster_size=25,
    min_cluster_fraction=0.005,
    num_folds=5,
    min_nodes_per_fold=5,
):
    num_nodes = int(num_nodes)
    if num_nodes <= 0:
        return 1

    min_size = max(
        1,
        int(np.ceil(float(min_cluster_size))),
        int(np.ceil(float(min_cluster_fraction) * num_nodes)),
        int(num_folds) * int(min_nodes_per_fold),
    )
    return min(min_size, num_nodes)


def _fit_kmeans(distributions, num_clusters, seed, max_iter=100, n_init=5):
    rng = np.random.default_rng(int(seed))
    num_nodes = distributions.shape[0]
    best_labels = None
    best_centers = None
    best_inertia = float("inf")

    for _ in range(int(n_init)):
        center_indices = rng.choice(num_nodes, size=num_clusters, replace=False)
        centers = distributions[center_indices].copy()
        labels = np.full(num_nodes, -1, dtype=np.int64)

        for _ in range(int(max_iter)):
            distances = np.sum((distributions[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(distances, axis=1).astype(np.int64)
            if np.array_equal(labels, new_labels):
                break

            labels = new_labels
            for cluster_id in range(num_clusters):
                cluster_mask = labels == cluster_id
                if np.any(cluster_mask):
                    centers[cluster_id] = distributions[cluster_mask].mean(axis=0)
                else:
                    centers[cluster_id] = distributions[rng.integers(num_nodes)]

        distances = np.sum((distributions[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1).astype(np.int64)
        inertia = float(distances[np.arange(num_nodes), labels].sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    return best_labels, best_centers, best_inertia


def _merge_rare_clusters(cluster_ids, distributions, centers, min_cluster_size):
    if min_cluster_size <= 1:
        return _compact_cluster_ids(cluster_ids)

    cluster_ids = np.asarray(cluster_ids, dtype=np.int64).copy()
    counts = np.bincount(cluster_ids)
    stable_clusters = np.flatnonzero(counts >= min_cluster_size)
    if len(stable_clusters) == len(counts):
        return _compact_cluster_ids(cluster_ids)
    if len(stable_clusters) == 0:
        return np.zeros_like(cluster_ids, dtype=np.int64)

    rare_mask = ~np.isin(cluster_ids, stable_clusters)
    stable_centers = centers[stable_clusters]
    rare_distributions = distributions[rare_mask]
    distances = np.linalg.norm(
        rare_distributions[:, None, :] - stable_centers[None, :, :],
        axis=2,
    )
    cluster_ids[rare_mask] = stable_clusters[np.argmin(distances, axis=1)]
    return _compact_cluster_ids(cluster_ids)


def cluster_label_distributions(distributions, num_clusters=50, seed=0, min_cluster_size=5):
    """
    Cluster per-node label distributions into categorical neighborhood patterns.
    """
    distributions = np.asarray(distributions, dtype=float)
    if distributions.ndim != 2:
        raise ValueError("Expected propagated label distributions with shape [num_nodes, num_classes].")
    if not np.all(np.isfinite(distributions)):
        raise ValueError("Propagated label distributions contain non-finite values.")

    num_nodes = distributions.shape[0]
    if num_nodes == 0:
        return np.array([], dtype=np.int64)

    min_cluster_size = max(1, int(min_cluster_size))
    max_clusters = max(1, num_nodes // min_cluster_size)
    effective_clusters = min(int(num_clusters), max_clusters)
    if effective_clusters <= 1:
        return np.zeros(num_nodes, dtype=np.int64)

    cluster_ids, centers, _ = _fit_kmeans(
        distributions=distributions,
        num_clusters=effective_clusters,
        seed=seed,
    )
    return _merge_rare_clusters(
        cluster_ids=cluster_ids,
        distributions=distributions,
        centers=centers,
        min_cluster_size=min_cluster_size,
    )


def _prepare_gap_statistic_inputs(
    distributions,
    min_cluster_size,
    min_cluster_fraction=0.005,
    num_folds=5,
    min_nodes_per_fold=5,
    min_k=2,
    max_k=50,
):
    distributions = np.asarray(distributions, dtype=float)
    if distributions.ndim != 2:
        raise ValueError("Expected propagated label distributions with shape [num_nodes, num_classes].")

    num_nodes = distributions.shape[0]
    if num_nodes == 0:
        return distributions, []

    min_cluster_size = compute_effective_min_cluster_size(
        num_nodes=num_nodes,
        min_cluster_size=min_cluster_size,
        min_cluster_fraction=min_cluster_fraction,
        num_folds=num_folds,
        min_nodes_per_fold=min_nodes_per_fold,
    )
    max_clusters = max(1, num_nodes // min_cluster_size)
    if max_k is not None:
        max_clusters = min(max_clusters, max(1, int(max_k)))
    min_k = max(1, int(min_k))
    if max_clusters < min_k:
        return distributions, [max_clusters]

    return distributions, list(range(min_k, max_clusters + 1))


def _gap_statistic_score(distributions, cluster_count, rng, reference_runs, lower, upper):
    _, _, inertia = _fit_kmeans(
        distributions=distributions,
        num_clusters=cluster_count,
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    log_inertia = float(np.log(max(inertia, 1e-12)))
    reference_log_inertias = []
    for _ in range(reference_runs):
        reference = rng.uniform(lower, upper, size=distributions.shape)
        _, _, reference_inertia = _fit_kmeans(
            distributions=reference,
            num_clusters=cluster_count,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        reference_log_inertias.append(np.log(max(reference_inertia, 1e-12))) #store results from all reference runs

    reference_log_inertias = np.asarray(reference_log_inertias, dtype=float)
    reference_log_inertia_mean = float(np.mean(reference_log_inertias))
    gap = float(reference_log_inertia_mean - log_inertia) # Gap(k) = E_ref[log(W_k)] - log(W_k)
    reference_sd = float(np.std(reference_log_inertias, ddof=1))
    reference_se = reference_sd * np.sqrt(1.0 + 1.0 / reference_runs)
    return {
        "k": int(cluster_count),
        "log_inertia": log_inertia,
        "reference_log_inertia_mean": reference_log_inertia_mean,
        "gap": gap,
        "reference_sd": reference_sd,
        "reference_se": reference_se,
        "gap_minus_reference_se": gap - reference_se,
    }


def _make_progress_bar(total, enabled, label):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(total=total, desc=label, unit="k")


def compute_gap_statistic_curve(
    distributions,
    seed,
    reference_runs,
    min_cluster_size,
    min_cluster_fraction=0.005,
    num_folds=5,
    min_nodes_per_fold=5,
    min_k=2,
    max_k=50,
    show_progress=True,
    progress_label="Gap statistic curve",
):
    """
    Compute Gap(k) for every k in the configured range.

    The reference samples are drawn uniformly from the coordinate-wise bounding
    box of the observed propagated-label distributions.
    """
    distributions, cluster_counts = _prepare_gap_statistic_inputs(
        distributions=distributions,
        min_cluster_size=min_cluster_size,
        min_cluster_fraction=min_cluster_fraction,
        num_folds=num_folds,
        min_nodes_per_fold=min_nodes_per_fold,
        min_k=min_k,
        max_k=max_k,
    )
    if not cluster_counts:
        return []

    reference_runs = max(2, int(reference_runs))
    rng = np.random.default_rng(int(seed))
    lower = distributions.min(axis=0)
    upper = distributions.max(axis=0)
    progress_bar = _make_progress_bar(
        total=len(cluster_counts),
        enabled=show_progress,
        label=progress_label,
    )

    curve = []
    try:
        for cluster_count in cluster_counts:
            curve.append(
                _gap_statistic_score(
                    distributions=distributions,
                    cluster_count=cluster_count,
                    rng=rng,
                    reference_runs=reference_runs,
                    lower=lower,
                    upper=upper,
                )
            )
            if progress_bar is not None:
                progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return curve


def select_gap_statistic_cluster_count_from_curve(curve):
    if not curve:
        return None

    for current, following in zip(curve[:-1], curve[1:]):
        if current["gap"] >= following["gap"] - following["reference_se"]:
            return int(current["k"])

    return int(curve[-1]["k"])


def select_gap_statistic_cluster_counts(
    distributions,
    seed,
    reference_runs,
    min_cluster_size,
    min_cluster_fraction=0.005,
    num_folds=5,
    min_nodes_per_fold=5,
    min_k=2,
    max_k=50,
    show_progress=True,
    progress_label="Gap statistic selection",
):
    """
    Select k using the original gap-statistic stopping rule.

    Candidate values are all integers from min_k to the smaller of the
    configured cap and the largest k that still respects the minimum
    cluster-size heuristic. We choose the first k where
    Gap(k) >= Gap(k + 1) - s'_{k+1}, with
    s'_k = sd(log(W_ref(k))) * sqrt(1 + 1 / B).
    """
    distributions, cluster_counts = _prepare_gap_statistic_inputs(
        distributions=distributions,
        min_cluster_size=min_cluster_size,
        min_cluster_fraction=min_cluster_fraction,
        num_folds=num_folds,
        min_nodes_per_fold=min_nodes_per_fold,
        min_k=min_k,
        max_k=max_k,
    )
    if not cluster_counts:
        return []

    reference_runs = max(2, int(reference_runs))
    rng = np.random.default_rng(int(seed))
    lower = distributions.min(axis=0)
    upper = distributions.max(axis=0)
    progress_bar = _make_progress_bar(
        total=len(cluster_counts),
        enabled=show_progress,
        label=progress_label,
    )

    try:
        current = _gap_statistic_score(distributions, cluster_counts[0], rng, reference_runs, lower, upper)
        if progress_bar is not None:
            progress_bar.update(1)
        for next_k in cluster_counts[1:]:
            following = _gap_statistic_score(distributions, next_k, rng, reference_runs, lower, upper)
            if progress_bar is not None:
                progress_bar.update(1)
            if current["gap"] >= following["gap"] - following["reference_se"]:
                return [int(current["k"])]
            current = following
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return [int(current["k"])]


def compute_propagated_label_cluster_ids(
    data,
    num_hops=3,
    decay=0.5,
    num_clusters=50,
    seed=0,
    min_cluster_size=5,
):
    distributions = compute_propagated_label_distribution(
        data=data,
        num_hops=num_hops,
        decay=decay,
    )
    return cluster_label_distributions(
        distributions=distributions,
        num_clusters=num_clusters,
        seed=seed,
        min_cluster_size=min_cluster_size,
    )
