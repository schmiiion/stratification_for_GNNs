from stratify.label_based_stratifier import LabelStratifiedKFold
from stratify.propagated_label_distribution import (
    compute_propagated_label_distribution,
    select_gap_statistic_cluster_counts,
)
from stratify.random_stratifier import RandomKFold
from stratify.wdes_stratifier import WDESKFold
from utils.experiment_utils import as_list


STRATIFIER_REGISTRY = {
    "label": LabelStratifiedKFold,
    "property": WDESKFold,
    "property_stratified": WDESKFold,
    "random": RandomKFold,
    "wdes": WDESKFold,
}

PROPERTY_BASED_STRATIFIER_KEYS = {"property", "property_stratified", "wdes"}


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


def get_properties(cfg):
    configured_properties = cfg_get(cfg, "properties", ["Degree"])
    return as_list(configured_properties)


def get_property_variants(cfg, property_name, seed, data=None):
    canonical_name = WDESKFold.canonical_property_name(property_name)
    if canonical_name != "Propagated Label Cluster":
        return [(canonical_name, {})]

    selection_method = str(
        cfg_get(cfg, "propagated_label_cluster_selection", "fixed")
    ).replace("-", "_").lower()
    if selection_method in {"gap", "gap_topk", "gap_statistic"} and data is not None:
        distributions = compute_propagated_label_distribution(
            data=data,
            num_hops=cfg_get(cfg, "propagated_label_num_hops", 3),
            decay=cfg_get(cfg, "propagated_label_decay", 0.5),
        )
        cluster_counts = select_gap_statistic_cluster_counts(
            distributions=distributions,
            candidate_clusters=as_list(cfg_get(cfg, "propagated_label_cluster_candidates", [50])),
            top_k=cfg_get(cfg, "propagated_label_gap_top_k", 3),
            seed=seed,
            reference_runs=cfg_get(cfg, "propagated_label_gap_reference_runs", 5),
            min_cluster_size=cfg_get(cfg, "num_folds", 5),
        )
        print(f"Gap statistic selected propagated-label cluster counts: {cluster_counts}")
    else:
        cluster_counts = as_list(cfg_get(cfg, "propagated_label_num_clusters", [50]))

    return [
        (canonical_name, {"propagated_label_num_clusters": int(cluster_count)})
        for cluster_count in cluster_counts
    ]


def get_stratifiers(cfg, dataset_name, seed, data=None):
    stratification_types = as_list(cfg_get(cfg, "stratification_types", ["label"]))
    stratifiers = []

    for stratification_type in stratification_types:
        key = stratification_type.replace("-", "_").lower()
        stratifier_class = get_stratifier_class(stratification_type)

        if key in PROPERTY_BASED_STRATIFIER_KEYS:
            for property_name in get_properties(cfg):
                for canonical_property_name, property_options in get_property_variants(
                    cfg=cfg,
                    property_name=property_name,
                    seed=seed,
                    data=data,
                ):
                    stratifiers.append(
                        stratifier_class(
                            cfg=cfg,
                            dataset_name=dataset_name,
                            n_splits=cfg.num_folds,
                            seed=seed,
                            property_name=canonical_property_name,
                            property_options=property_options,
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
