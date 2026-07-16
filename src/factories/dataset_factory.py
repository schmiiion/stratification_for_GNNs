import torch
from pathlib import Path
from types import SimpleNamespace
import hashlib
import re
import tarfile
from urllib.request import urlretrieve
from torch_geometric.datasets import (
    Actor,
    Amazon,
    Coauthor,
    Planetoid,
    WikiCS,
    WikipediaNetwork,
)
from torch_geometric.data import Data
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import (
    contains_self_loops,
    is_undirected,
    remove_self_loops,
    to_scipy_sparse_matrix,
    to_undirected,
)

from utils.graph_characteristics import (
    compute_edge_label_informativeness,
    compute_graph_size_summary,
    compute_node_label_informativeness,
)
from utils.dataset_reference_metrics import sanity_check_label_informativeness

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
        "CoauthorCS": Coauthor,
        "CoauthorPhysics": Coauthor,
        "WikiCS": WikiCS,
        "crocodile": WikipediaNetwork,
    }

    _REGISTRY_ALIASES = {
        "coauthor_cs": "CoauthorCS",
        "coauthorcs": "CoauthorCS",
        "coauthor_physics": "CoauthorPhysics",
        "coauthorphysics": "CoauthorPhysics",
        "wiki_cs": "WikiCS",
        "wikics": "WikiCS",
    }

    _HETEROPHILOUS_DATASET_FILES = {                        #these are from the platonov paper
        "chameleon": Path("chameleon_clean/chameleon_filtered.npz"),
        "squirrel": Path("squirrel_filtered/squirrel_filtered.npz"),
        "texas": Path("texas/texas.npz"),
        "roman_empire": Path("roman_empire/roman_empire.npz"),
        "wisconsin": Path("wisconsin/wisconsin.npz"),
        "actor": Path("Actor/actor.npz"),
        "cornell": Path("cornell/cornell.npz"),
        "amazon_ratings": Path("amazon_ratings/amazon_ratings.npz"),
    }

    _DEFAULT_HETEROPHILOUS_ROOT = (
        Path(__file__).resolve().parents[1] / "data"
    )

    _SYN_CORA_NAME = "syn-cora"
    _SYN_CORA_URL = "https://public-files.jiongzhu.net/syn-cora-npz.tar.gz"
    _SYN_CORA_ARCHIVE = "syn-cora-npz.tar.gz"
    _SYN_CORA_SHA256 = "7609527ece3dbc3eadb84350754404a37d5fc6b2dc3ff74f0e4fda3922fb28fa"
    _DEFAULT_SYNTHETIC_ROOT = Path(__file__).resolve().parents[1] / "data"

    @classmethod
    def get_dataset(
            cls,
            name: str,
            root_dir: str = '/tmp/',
            heterophilous_root_dir: str | Path | None = None,
            syn_cora_root_dir: str | Path | None = None,
            syn_cora_homophily: float | str | None = None,
            syn_cora_realization: int = 1):
        """
        Instantiates and returns the dataset and its properties.

        Args:
            name (str): The name of the dataset (e.g., "Cora", "chameleon").
            root_dir (str): The root directory to download/load the data.

        Returns:
            tuple: (dataset, input_dim, output_dim, data)
        """
        canonical_dataset_name = cls._canonical_dataset_name(name)
        syn_cora_request = cls._parse_syn_cora_name(canonical_dataset_name)
        if syn_cora_request is not None:
            name_homophily, name_realization = syn_cora_request
            synthetic_root = (
                root_dir
                if syn_cora_root_dir is None and root_dir != '/tmp/'
                else syn_cora_root_dir
            )
            return cls.load_syn_cora_dataset(
                homophily=(
                    0.70
                    if syn_cora_homophily is None and name_homophily is None
                    else name_homophily
                    if syn_cora_homophily is None
                    else syn_cora_homophily
                ),
                realization=name_realization if name_realization is not None else syn_cora_realization,
                root_dir=synthetic_root,
            )

        registry_name = cls._registry_dataset_name(name)
        heterophilous_name = cls._canonical_heterophilous_name(name)
        if registry_name not in cls._REGISTRY and heterophilous_name in cls._HETEROPHILOUS_DATASET_FILES:
            return cls.load_heterophilous_graph_dataset(
                name=heterophilous_name,
                root_dir=heterophilous_root_dir,
            )

        if registry_name not in cls._REGISTRY:
            supported = ", ".join(
                sorted([
                    *cls._REGISTRY.keys(),
                    *cls._REGISTRY_ALIASES.keys(),
                    *cls._HETEROPHILOUS_DATASET_FILES.keys(),
                    cls._SYN_CORA_NAME,
                ])
            )
            raise ValueError(f"Dataset '{name}' not recognized. Supported datasets: {supported}")

        DatasetClass = cls._REGISTRY[registry_name]
        data_path = f"{root_dir.rstrip('/')}/{registry_name}"

        if DatasetClass == Actor:
            dataset = DatasetClass(root=data_path)
        elif DatasetClass == Coauthor:
            coauthor_name = "CS" if registry_name == "CoauthorCS" else "Physics"
            dataset = DatasetClass(root=data_path, name=coauthor_name)
        elif DatasetClass == WikiCS:
            dataset = DatasetClass(root=data_path)
        else:
            dataset = DatasetClass(root=data_path, name=registry_name)

        # Extract dimensions and move data to the target device
        input_dim = dataset.num_features
        output_dim = dataset.num_classes
        data = dataset[0]
        data.edge_index = cls.standardize_paper_edge_index(
            data.edge_index,
            num_nodes=data.num_nodes,
        )

        print(f"Nodes: {data.num_nodes}")
        print(f"Edges: {data.num_edges}")
        print(f"Features: {input_dim}")
        print(f"Classes: {output_dim}")

        cls.validate_edge_index_topology(data.edge_index, data.num_nodes)
        cls.attach_graph_characteristics(data, name)

        return dataset, input_dim, output_dim, data

    @classmethod
    def load_syn_cora_dataset(
            cls,
            homophily: float | str,
            realization: int = 1,
            root_dir: str | Path | None = None):
        """
        Load Zhu et al.'s synthetic Cora NPZ dataset.

        The official archive is published by the H2GCN authors at:
        https://public-files.jiongzhu.net/syn-cora-npz.tar.gz
        """
        root = Path(root_dir) if root_dir is not None else cls._DEFAULT_SYNTHETIC_ROOT
        syn_cora_root = root / cls._SYN_CORA_NAME
        homophily_value = float(homophily)
        realization = int(realization)
        filename = f"h{homophily_value:.2f}-r{realization}.npz"
        data_path = cls._ensure_syn_cora_file(syn_cora_root, filename)

        with np.load(data_path, allow_pickle=True) as raw:
            adj = sp.csr_matrix(
                (raw["adj_data"], raw["adj_indices"], raw["adj_indptr"]),
                shape=tuple(raw["adj_shape"]),
            )
            features = sp.csr_matrix(
                (raw["attr_data"], raw["attr_indices"], raw["attr_indptr"]),
                shape=tuple(raw["attr_shape"]),
            )
            labels = raw["labels"]
            metadata = str(raw["metadata"]) if "metadata" in raw.files else ""

        # Match the official NPZ loader: symmetrize, binarize, and remove loops.
        adj = adj + adj.T
        adj = adj.tolil(copy=False)
        adj[adj > 1] = 1
        adj.setdiag(0)
        adj = adj.astype("float32").tocsr()
        adj.eliminate_zeros()

        edge_index = torch.from_numpy(np.vstack(adj.nonzero())).to(torch.long)
        edge_index = cls.standardize_paper_edge_index(
            edge_index,
            num_nodes=features.shape[0],
        )

        x = torch.from_numpy(features.toarray()).to(torch.float32)
        y = torch.from_numpy(labels).to(torch.long)
        data = Data(x=x, y=y, edge_index=edge_index, num_nodes=x.size(0))
        data.synthetic_homophily = homophily_value
        data.synthetic_realization = realization

        input_dim = data.num_features
        output_dim = int(y.max().item()) + 1
        dataset_name = f"{cls._SYN_CORA_NAME}-h{homophily_value:.2f}-r{realization}"
        dataset = SimpleNamespace(
            name=dataset_name,
            num_features=input_dim,
            num_classes=output_dim,
            source_path=str(data_path),
            metadata=metadata,
            homophily=homophily_value,
            realization=realization,
        )

        print(f"Nodes: {data.num_nodes}")
        print(f"Edges: {data.num_edges}")
        print(f"Features: {input_dim}")
        print(f"Classes: {output_dim}")
        print(f"Source: {data_path}")
        print(f"Synthetic homophily target: {homophily_value:.2f}")
        print(f"Synthetic realization: r{realization}")

        cls.validate_edge_index_topology(data.edge_index, data.num_nodes)
        cls.attach_graph_characteristics(data, cls._SYN_CORA_NAME)

        return dataset, input_dim, output_dim, data

    @classmethod
    def load_heterophilous_graph_dataset(
            cls,
            name: str,
            root_dir: str | Path | None = None):
        """
        Load .npz datasets from the heterophilous-graphs repository as PyG Data.

        The source files store undirected graphs with each edge listed once.
        PyG represents message-passing topology as directed COO edges, so we
        materialize both directions with to_undirected().
        """
        root = Path(root_dir) if root_dir is not None else cls._DEFAULT_HETEROPHILOUS_ROOT
        canonical_name = cls._canonical_heterophilous_name(name)
        relative_path = cls._HETEROPHILOUS_DATASET_FILES[canonical_name]
        data_path = cls._resolve_heterophilous_data_path(root, relative_path)
        if not data_path.exists():
            candidate_text = ", ".join(
                str(path) for path in cls._heterophilous_data_path_candidates(root, relative_path)
            )
            raise FileNotFoundError(
                f"Could not find heterophilous dataset '{name}'. Checked: {candidate_text}"
            )

        with np.load(data_path) as raw:
            x = torch.from_numpy(raw["node_features"]).to(torch.float32)
            y = torch.from_numpy(raw["node_labels"]).to(torch.long)
            edges = torch.from_numpy(raw["edges"]).to(torch.long)

        edge_index = edges.t().contiguous()
        edge_index = cls.standardize_paper_edge_index(
            edge_index,
            num_nodes=x.size(0),
        )

        data = Data(x=x, y=y, edge_index=edge_index, num_nodes=x.size(0))

        input_dim = data.num_features
        output_dim = int(y.max().item()) + 1
        dataset = SimpleNamespace(
            name=name,
            num_features=input_dim,
            num_classes=output_dim,
            source_path=str(data_path),
        )

        print(f"Nodes: {data.num_nodes}")
        print(f"Edges: {data.num_edges}")
        print(f"Features: {input_dim}")
        print(f"Classes: {output_dim}")
        print(f"Source: {data_path}")

        cls.validate_edge_index_topology(data.edge_index, data.num_nodes)
        cls.attach_graph_characteristics(data, canonical_name)

        return dataset, input_dim, output_dim, data

    @staticmethod
    def _canonical_heterophilous_name(name: str):
        return name.replace("-", "_").replace(" ", "_").lower()

    @staticmethod
    def _canonical_dataset_name(name: str):
        return name.replace("_", "-").lower()

    @classmethod
    def _registry_dataset_name(cls, name: str):
        key = name.replace("-", "_").replace(" ", "_").lower()
        return cls._REGISTRY_ALIASES.get(key, name)

    @staticmethod
    def _heterophilous_data_path_candidates(root: Path, relative_path: Path):
        return [
            root / relative_path,
            root / relative_path.name,
        ]

    @classmethod
    def _resolve_heterophilous_data_path(cls, root: Path, relative_path: Path):
        for candidate in cls._heterophilous_data_path_candidates(root, relative_path):
            if candidate.exists():
                return candidate
        return root / relative_path

    @classmethod
    def _parse_syn_cora_name(cls, canonical_name: str):
        match = re.fullmatch(
            rf"{cls._SYN_CORA_NAME}(?:-h(?P<homophily>\d(?:\.\d+)?))?(?:-r(?P<realization>\d+))?",
            canonical_name,
        )
        if match is None:
            return None

        homophily = match.group("homophily")
        realization = match.group("realization")
        return (
            None if homophily is None else float(homophily),
            None if realization is None else int(realization),
        )

    @classmethod
    def _ensure_syn_cora_file(cls, syn_cora_root: Path, filename: str):
        data_path = syn_cora_root / filename
        if data_path.exists():
            return data_path

        syn_cora_root.mkdir(parents=True, exist_ok=True)
        archive_path = syn_cora_root / cls._SYN_CORA_ARCHIVE
        if not archive_path.exists():
            print(f"Downloading official syn-cora NPZ archive from {cls._SYN_CORA_URL}")
            try:
                urlretrieve(cls._SYN_CORA_URL, archive_path)
            except Exception as exc:
                raise FileNotFoundError(
                    f"Could not find {data_path} and automatic download failed. "
                    f"Download {cls._SYN_CORA_URL} manually, verify SHA256 "
                    f"{cls._SYN_CORA_SHA256}, and extract it under {syn_cora_root.parent}."
                ) from exc

        cls._verify_sha256(archive_path, cls._SYN_CORA_SHA256)
        with tarfile.open(archive_path, mode="r:gz") as archive:
            archive.extractall(path=syn_cora_root.parent)

        if not data_path.exists():
            available = ", ".join(sorted(path.name for path in syn_cora_root.glob("h*.npz")))
            raise FileNotFoundError(
                f"syn-cora file {filename} not found after extraction. "
                f"Available files: {available}"
            )
        return data_path

    @staticmethod
    def _verify_sha256(path: Path, expected_sha256: str):
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Checksum mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
            )

    @staticmethod
    def standardize_paper_edge_index(edge_index, num_nodes):
        edge_index, _ = remove_self_loops(edge_index)
        return to_undirected(edge_index, num_nodes=num_nodes)

    @staticmethod
    def validate_edge_index_topology(edge_index, num_nodes, expect_undirected=True):
        """Check for self loops and directedness. If so, raise excaptions"""
        has_self_loops = contains_self_loops(edge_index)
        undirected = is_undirected(edge_index, num_nodes=num_nodes)
        num_missing_reverse = 0

        if not undirected:
            edges = set(map(tuple, edge_index.t().tolist()))
            num_missing_reverse = sum((v, u) not in edges for u, v in edges)

        print("Self-loops:", int(has_self_loops))
        print("Missing reverse edges:", num_missing_reverse)

        if has_self_loops:
            raise ValueError(
                "Loaded graph contains self-loops. Keep raw datasets loop-free; "
                "PyG layers or model-specific preprocessing should add loops when needed."
            )
        if expect_undirected and not undirected:
            raise ValueError(
                "Loaded graph is not represented as bidirectional/undirected edge_index."
            )

    @staticmethod
    def attach_graph_characteristics(data, dataset_name=None):
        data.graph_size_summary = compute_graph_size_summary(data.edge_index, data.num_nodes)
        data.num_undirected_edges = data.graph_size_summary["undirected_edges"]
        data.average_degree = data.graph_size_summary["average_degree"]
        data.label_informativeness_edge = compute_edge_label_informativeness(
            data.edge_index,
            data.y,
        )
        data.label_informativeness_node = compute_node_label_informativeness(
            data.edge_index,
            data.y,
            data.num_nodes,
        )

        print(f"Undirected edges: {data.num_undirected_edges}")
        print(f"Average degree: {data.average_degree:.2f}")
        print(f"LI_edge: {data.label_informativeness_edge:.4f}")
        print(f"LI_node: {data.label_informativeness_node:.4f}")
        if dataset_name is not None:
            sanity_check_label_informativeness(
                dataset_name,
                data.label_informativeness_edge,
                data.label_informativeness_node,
            )

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
