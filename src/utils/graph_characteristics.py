import numpy as np
import torch


def _as_numpy_labels(y):
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    _, inverse = np.unique(np.asarray(y), return_inverse=True) #from [10, 10, 30, 50, 30] to [10, 30, 50], [0, 0, 1, 2, 1]
    return inverse.astype(np.int64)


def _as_numpy_edge_index(edge_index):
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.detach().cpu().numpy()
    return np.asarray(edge_index, dtype=np.int64)


def compute_graph_size_summary(edge_index, num_nodes):
    edge_index = _as_numpy_edge_index(edge_index)
    num_nodes = int(num_nodes)
    if edge_index.size == 0 or num_nodes == 0:
        return {
            "num_nodes": num_nodes,
            "directed_edges": 0,
            "undirected_edges": 0,
            "average_degree": 0.0,
        }

    edge_pairs = edge_index.T
    unordered_edges = np.sort(edge_pairs, axis=1)
    unique_undirected_edges = np.unique(unordered_edges, axis=0)
    num_undirected_edges = int(unique_undirected_edges.shape[0])

    return {
        "num_nodes": num_nodes,
        "directed_edges": int(edge_pairs.shape[0]),
        "undirected_edges": num_undirected_edges,
        "average_degree": float(2 * num_undirected_edges / num_nodes),
    }


def _entropy(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities[probabilities > 0]
    if probabilities.size == 0:
        return 0.0
    return float(-np.sum(probabilities * np.log(probabilities)))


def _mutual_information(joint, source_marginal, target_marginal):
    expected = np.outer(source_marginal, target_marginal)
    mask = joint > 0
    if not np.any(mask):
        return 0.0
    return float(np.sum(joint[mask] * np.log(joint[mask] / expected[mask])))


def _normalized_mutual_information(joint, source_marginal):
    entropy = _entropy(source_marginal)
    if entropy <= 0:
        return 0.0

    target_marginal = joint.sum(axis=0)
    mutual_information = _mutual_information(joint, source_marginal, target_marginal)
    return float(mutual_information / entropy)


def compute_edge_label_informativeness(edge_index, y):
    """
    Compute Platonov et al.'s main LI variant.

    This samples an edge endpoint pair uniformly from the directed edge_index
    representation. For an undirected PyG graph with both directions present,
    this gives the paper's edge-weighted label informativeness.
    """
    labels = _as_numpy_labels(y)
    edge_index = _as_numpy_edge_index(edge_index)
    num_classes = int(labels.max()) + 1

    if edge_index.size == 0:
        return 0.0

    source_labels = labels[edge_index[0]]
    target_labels = labels[edge_index[1]]
    joint_counts = np.zeros((num_classes, num_classes), dtype=float) #
    np.add.at(joint_counts, (source_labels, target_labels), 1.0) #fill the CxC Joint Distribution (label-pair)

    joint = joint_counts / joint_counts.sum() #normalize
    source_marginal = joint.sum(axis=1) #distribution of source labels
    return _normalized_mutual_information(joint, source_marginal) #final equation from the definition


def compute_node_label_informativeness(edge_index, y, num_nodes):
    """
    Compute Appendix C.2's node-first LI variant.

    The sampling process is: choose a node uniformly, then choose one of its
    neighbors uniformly. Isolated nodes have no valid neighbor choice, so the
    implementation conditions on non-isolated nodes.
    """
    labels = _as_numpy_labels(y)
    edge_index = _as_numpy_edge_index(edge_index)
    num_classes = int(labels.max()) + 1

    if edge_index.size == 0:
        return 0.0

    source_nodes = edge_index[0]
    target_nodes = edge_index[1]
    degrees = np.bincount(source_nodes, minlength=num_nodes).astype(float)
    non_isolated = degrees > 0
    num_non_isolated = int(non_isolated.sum())
    if num_non_isolated == 0:
        return 0.0

    source_labels = labels[source_nodes]
    target_labels = labels[target_nodes]
    edge_weights = 1.0 / (num_non_isolated * degrees[source_nodes])

    joint = np.zeros((num_classes, num_classes), dtype=float)
    np.add.at(joint, (source_labels, target_labels), edge_weights)

    source_label_counts = np.bincount(
        labels[non_isolated],
        minlength=num_classes,
    ).astype(float)
    source_marginal = source_label_counts / source_label_counts.sum()

    return _normalized_mutual_information(joint, source_marginal)


def compute_label_informativeness(edge_index, y, num_nodes=None, variant="edge"):
    if variant == "edge":
        return compute_edge_label_informativeness(edge_index, y)
    if variant == "node":
        if num_nodes is None:
            num_nodes = int(y.numel()) if isinstance(y, torch.Tensor) else len(y)
        return compute_node_label_informativeness(edge_index, y, num_nodes)
    raise ValueError("variant must be either 'edge' or 'node'.")
