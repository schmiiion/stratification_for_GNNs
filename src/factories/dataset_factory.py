import torch
from torch_geometric.datasets import Planetoid, Amazon, WikipediaNetwork, Actor
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix

class DatasetFactory:
    """
    Factory class to load PyTorch Geometric datasets by name.
    """

    # Internal registry mapping the string name to the PyG Dataset Class
    _REGISTRY = {
        "Cora": Planetoid,
        "CiteSeer": Planetoid,
        "PubMed": Planetoid,
        "Computers": Amazon,
        "Photo": Amazon,
        "chameleon": WikipediaNetwork,
        "squirrel": WikipediaNetwork,
        "crocodile": WikipediaNetwork,
        "Actor": Actor
    }

    @classmethod
    def get_dataset(cls, name: str, root_dir: str = '/tmp/'):
        """
        Instantiates and returns the dataset and its properties.

        Args:
            name (str): The name of the dataset (e.g., "Cora", "chameleon").
            root_dir (str): The root directory to download/load the data.

        Returns:
            tuple: (dataset, input_dim, output_dim, data)
        """
        if name not in cls._REGISTRY:
            supported = ", ".join(cls._REGISTRY.keys())
            raise ValueError(f"Dataset '{name}' not recognized. Supported datasets: {supported}")

        DatasetClass = cls._REGISTRY[name]
        data_path = f"{root_dir.rstrip('/')}/{name}"

        # Handle the specific instantiation logic for Actor vs others
        if DatasetClass == Actor:
            dataset = DatasetClass(root=data_path)
        else:
            dataset = DatasetClass(root=data_path, name=name)

        # Extract dimensions and move data to the target device
        input_dim = dataset.num_features
        output_dim = dataset.num_classes
        data = dataset[0]

        print(f"Nodes: {data.num_nodes}")
        print(f"Edges: {data.num_edges}")
        print(f"Features: {input_dim}")
        print(f"Classes: {output_dim}")

        edge_index = data.edge_index
        edges = set(map(tuple, edge_index.t().tolist()))
        num_missing_reverse = sum((v, u) not in edges for u, v in edges)
        print("Missing reverse edges:", num_missing_reverse)

        return dataset, input_dim, output_dim, data

    @classmethod
    def precompute_h2gcn_hops(cls, edge_index, num_nodes):
        """
        Precompute H2GCN exact 1-hop and exact 2-hop normalized adjacency matrices.

        This mirrors H2GCN's TransformSPAdj.nhoodSplit(adj, 2) followed by
        independent symmetric normalization of hop 1 and hop 2.

        Returns:
            hop1_norm, hop2_norm : torch.sparse_coo_tensor
                Both are shape [num_nodes, num_nodes], symmetrically normalized.
        """

        # Build sparse adjacency over the FULL node set.
        adj = (
            to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
            .tocsr()
            .astype(np.float32)
        )

        # H2GCN assumes an unweighted binary adjacency for neighborhood construction.
        adj.data[:] = 1.0

        # The reference preprocessing calls adj_remove_eye() before nhoodSplit().
        adj.setdiag(0)
        adj.eliminate_zeros()

        adj_splits = cls._h2gcn_nhood_split(adj, nhood=2)
        hop1_norm = cls._scipy_to_torch_sparse(
            cls._normalize_sparse_adjacency(adj_splits[1])
        )
        hop2_norm = cls._scipy_to_torch_sparse(
            cls._normalize_sparse_adjacency(adj_splits[2])
        )

        return hop1_norm, hop2_norm

    @staticmethod
    def row_normalize_features(x: torch.Tensor):
        row_sum = x.sum(dim=1, keepdim=True)
        row_sum_inv = torch.zeros_like(row_sum)
        nonzero_rows = row_sum.squeeze(1) > 0
        row_sum_inv[nonzero_rows] = row_sum[nonzero_rows].reciprocal()
        return x * row_sum_inv

    @staticmethod
    def _h2gcn_nhood_split(adj: sp.csr_matrix, nhood: int):
        """
        Sparse equivalent of H2GCN's TransformSPAdj.nhoodSplit.

        The returned list contains adjacency shells:
        index 0 is I, index 1 is exact 1-hop, index 2 is exact 2-hop, etc.
        """
        assert adj.ndim == 2 and adj.shape[0] == adj.shape[1]

        mt = sp.eye(adj.shape[1], format="csr", dtype=adj.dtype)
        adj_with_eye = adj + sp.eye(adj.shape[0], format="csr", dtype=adj.dtype)
        mt_list = [mt]
        edge_sum = 0

        for _ in range(nhood):
            prev_mt = mt
            mt = mt @ adj_with_eye
            mt = (mt > 0).astype(adj.dtype).tocsr()

            new_edge_sum = mt.sum()
            if edge_sum == new_edge_sum:
                break

            edge_sum = new_edge_sum
            mt_list.append((mt - prev_mt).tocsr())

        while len(mt_list) <= nhood:
            mt_list.append(sp.csr_matrix(adj.shape, dtype=adj.dtype))

        return mt_list

    @staticmethod
    def _normalize_sparse_adjacency(adj: sp.csr_matrix):
        deg = adj.sum(axis=1).A1
        deg_inv_sqrt = np.zeros_like(deg, dtype=np.float32)
        nonzero_deg = deg > 0
        deg_inv_sqrt[nonzero_deg] = np.power(deg[nonzero_deg], -0.5)
        d_inv_sqrt = sp.diags(deg_inv_sqrt, format="csr")
        return (d_inv_sqrt @ adj @ d_inv_sqrt).tocoo()

    @staticmethod
    def _scipy_to_torch_sparse(adj: sp.spmatrix):
        adj = adj.tocoo()
        indices = torch.from_numpy(
            np.vstack((adj.row, adj.col)).astype(np.int64)
        )
        values = torch.from_numpy(adj.data.astype(np.float32))
        torch.sparse.check_sparse_tensor_invariants.disable()
        return torch.sparse_coo_tensor(
            indices,
            values,
            size=adj.shape,
            dtype=torch.float32
        ).coalesce()
