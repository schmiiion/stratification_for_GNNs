from stratify.label_based_stratifier import LabelStratifiedKFold
from stratify.random_stratifier import RandomKFold
from utils.experiment_utils import as_list


STRATIFIER_REGISTRY = {
    "label": LabelStratifiedKFold,
    "random": RandomKFold,
}


def get_stratifier_class(stratification_type):
    key = stratification_type.replace("-", "_").lower()
    if key not in STRATIFIER_REGISTRY:
        available = ", ".join(sorted(STRATIFIER_REGISTRY))
        raise ValueError(f"Unknown stratification type '{stratification_type}'. Available: {available}")
    return STRATIFIER_REGISTRY[key]


def get_stratifiers(cfg, dataset_name, seed):
    stratification_types = as_list(cfg.get("stratification_types", ["label"]))

    return [
        get_stratifier_class(stratification_type)(
            cfg=cfg,
            dataset_name=dataset_name,
            n_splits=cfg.num_folds,
            seed=seed,
        )
        for stratification_type in stratification_types
    ]
