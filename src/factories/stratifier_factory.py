from stratify.label_based_stratifier import LabelStratifiedKFold
from stratify.random_stratifier import RandomKFold
from stratify.wdes_stratifier import WDESKFold
from utils.experiment_utils import as_list


STRATIFIER_REGISTRY = {
    "label": LabelStratifiedKFold,
    "random": RandomKFold,
    "wdes": WDESKFold,
}


def cfg_get(cfg, key, default):
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def get_stratifier_class(stratification_type):
    key = stratification_type.replace("-", "_").lower()
    if key not in STRATIFIER_REGISTRY:
        available = ", ".join(sorted(STRATIFIER_REGISTRY))
        raise ValueError(f"Unknown stratification type '{stratification_type}'. Available: {available}")
    return STRATIFIER_REGISTRY[key]


def get_wdes_properties(cfg):
    configured_properties = cfg_get(cfg, "wdes_properties", ["Degree"])
    return as_list(configured_properties)


def get_stratifiers(cfg, dataset_name, seed):
    stratification_types = as_list(cfg_get(cfg, "stratification_types", ["label"]))
    stratifiers = []

    for stratification_type in stratification_types:
        key = stratification_type.replace("-", "_").lower()
        stratifier_class = get_stratifier_class(stratification_type)

        if key == "wdes":
            for property_name in get_wdes_properties(cfg):
                stratifiers.append(
                    stratifier_class(
                        cfg=cfg,
                        dataset_name=dataset_name,
                        n_splits=cfg.num_folds,
                        seed=seed,
                        property_name=property_name,
                    )
                )
        else:
            stratifiers.append(
                stratifier_class(
                    cfg=cfg,
                    dataset_name=dataset_name,
                    n_splits=cfg.num_folds,
                    seed=seed,
                )
            )

    return stratifiers
