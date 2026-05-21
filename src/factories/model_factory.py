import models


MODEL_REGISTRY = {
    "MLP": models.MLPNet,
    "GCN": models.GCNNet,
    "GAT": models.GATNet,
    "SAGE": models.SAGENet,
    "MixHop": models.MixHopNet,
    "GPRGNN": models.GPRGNN,
    "H2GCN": models.H2GCN_2,
}


def get_models(cfg, input_dim, output_dim, model_name):
    """
    Instantiate one configured model by name.
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    ModelClass = MODEL_REGISTRY[model_name]
    return ModelClass(
        in_channels=input_dim,
        hidden_channels=cfg.hidden_units,
        out_channels=output_dim,
    )
