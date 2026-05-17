import torch
from torch_geometric.datasets import Planetoid, Amazon, WikipediaNetwork, Actor


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
    def get_dataset(cls, name: str, root_dir: str = '/tmp/', device: str = 'cpu'):
        """
        Instantiates and returns the dataset and its properties.

        Args:
            name (str): The name of the dataset (e.g., "Cora", "chameleon").
            root_dir (str): The root directory to download/load the data.
            device (str or torch.device): The device to move the data to.

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
        data = dataset[0].to(device)

        print(f"Nodes: {data.num_nodes}")
        print(f"Edges: {data.num_edges}")
        print(f"Features: {input_dim}")
        print(f"Classes: {output_dim}")

        return dataset, input_dim, output_dim, data