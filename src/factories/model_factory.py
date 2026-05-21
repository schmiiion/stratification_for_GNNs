import models


def get_models(cfg, input_dim, output_dim):
    """
    Returns a list of (name, model_instance) based on the config.
    """
    # Registry
    MODEL_REGISTRY = {
        "MLP": models.MLPNet,
        "GCN": models.GCNNet,
        "GAT": models.GATNet,
        "SAGE": models.SAGENet,
        "MixHop": models.MixHopNet,
        "GPRGNN": models.GPRGNN,
        "H2GCN": models.H2GCN_2
    }

    selected_models = []

    for name in cfg.model_names:
        if name not in MODEL_REGISTRY:
            print(f"Warning: Model '{name}' not found in registry. Skipping.")
            continue

        ModelClass = MODEL_REGISTRY[name]

        # Instantiate the model
        instance = ModelClass(
            in_channels=input_dim,
            hidden_channels=cfg.hidden_units,
            out_channels=output_dim
        )

        selected_models.append((name, instance))

    return selected_models