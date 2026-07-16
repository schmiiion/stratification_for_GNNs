import contextlib
from functools import lru_cache
import io
from pathlib import Path
import re
import warnings

from utils.graph_characteristics import compute_graph_size_summary


SRC_ROOT = Path(__file__).resolve().parents[1]
LI_REFERENCE_TOLERANCE = 0.02

# Zhu et al. (2020), Table 5, "Hom. ratio h".
ZHU_HOMOPHILY_RATIO = {
    "texas": 0.11,
    "wisconsin": 0.21,
    "actor": 0.22,
    "squirrel": 0.22,
    "squirrel-filtered": 0.22,
    "chameleon": 0.23,
    "chameleon-filtered": 0.23,
    "cornell": 0.30,
    "cora-full": 0.57,
    "citeseer": 0.74,
    "pubmed": 0.80,
    "cora": 0.81,
}

# Platonov et al., Table 5. Values are copied from the paper excerpt.
PLATONOV_DATASET_CHARACTERISTICS = {
    "cora": {"n": 2708, "edges": 5278, "classes": 7, "h_edge": 0.81, "h_node": 0.83, "h_class": 0.77, "h_adj": 0.77, "li_edge": 0.59, "li_node": 0.61},
    "citeseer": {"n": 3327, "edges": 4552, "classes": 6, "h_edge": 0.74, "h_node": 0.72, "h_class": 0.63, "h_adj": 0.67, "li_edge": 0.45, "li_node": 0.45},
    "pubmed": {"n": 19717, "edges": 44324, "classes": 3, "h_edge": 0.80, "h_node": 0.79, "h_class": 0.66, "h_adj": 0.69, "li_edge": 0.41, "li_node": 0.40},
    "coauthor-cs": {"n": 18333, "edges": 81894, "classes": 15, "h_edge": 0.81, "h_node": 0.83, "h_class": 0.75, "h_adj": 0.78, "li_edge": 0.65, "li_node": 0.68},
    "coauthor-physics": {"n": 34493, "edges": 247962, "classes": 5, "h_edge": 0.93, "h_node": 0.92, "h_class": 0.85, "h_adj": 0.87, "li_edge": 0.72, "li_node": 0.76},
    "amazon-computers": {"n": 13752, "edges": 245861, "classes": 10, "h_edge": 0.78, "h_node": 0.80, "h_class": 0.70, "h_adj": 0.68, "li_edge": 0.53, "li_node": 0.62},
    "amazon-photo": {"n": 7650, "edges": 119081, "classes": 8, "h_edge": 0.83, "h_node": 0.85, "h_class": 0.77, "h_adj": 0.79, "li_edge": 0.67, "li_node": 0.72},
    "lastfm-asia": {"n": 7624, "edges": 27806, "classes": 18, "h_edge": 0.87, "h_node": 0.83, "h_class": 0.77, "h_adj": 0.86, "li_edge": 0.74, "li_node": 0.68},
    "facebook": {"n": 22470, "edges": 170823, "classes": 4, "h_edge": 0.89, "h_node": 0.88, "h_class": 0.82, "h_adj": 0.82, "li_edge": 0.62, "li_node": 0.74},
    "github": {"n": 37700, "edges": 289003, "classes": 2, "h_edge": 0.85, "h_node": 0.80, "h_class": 0.38, "h_adj": 0.38, "li_edge": 0.13, "li_node": 0.15},
    "twitter-hate": {"n": 2700, "edges": 11934, "classes": 2, "h_edge": 0.78, "h_node": 0.67, "h_class": 0.50, "h_adj": 0.55, "li_edge": 0.23, "li_node": 0.51},
    "ogbn-arxiv": {"n": 169343, "edges": 1157799, "classes": 40, "h_edge": 0.65, "h_node": 0.64, "h_class": 0.42, "h_adj": 0.59, "li_edge": 0.45, "li_node": 0.53},
    "ogbn-products": {"n": 2449029, "edges": 61859012, "classes": 47, "h_edge": 0.81, "h_node": 0.83, "h_class": 0.46, "h_adj": 0.79, "li_edge": 0.68, "li_node": 0.72},
    "actor": {"n": 7600, "edges": 26659, "classes": 5, "h_edge": 0.22, "h_node": 0.22, "h_class": 0.01, "h_adj": 0.00, "li_edge": 0.00, "li_node": 0.00},
    "flickr": {"n": 89250, "edges": 449878, "classes": 7, "h_edge": 0.32, "h_node": 0.32, "h_class": 0.07, "h_adj": 0.09, "li_edge": 0.01, "li_node": 0.01},
    "deezer-europe": {"n": 28281, "edges": 92752, "classes": 2, "h_edge": 0.53, "h_node": 0.53, "h_class": 0.03, "h_adj": 0.03, "li_edge": 0.00, "li_node": 0.00},
    "twitch-de": {"n": 9498, "edges": 153138, "classes": 2, "h_edge": 0.63, "h_node": 0.60, "h_class": 0.14, "h_adj": 0.14, "li_edge": 0.02, "li_node": 0.03},
    "twitch-pt": {"n": 1912, "edges": 31299, "classes": 2, "h_edge": 0.57, "h_node": 0.59, "h_class": 0.12, "h_adj": 0.11, "li_edge": 0.01, "li_node": 0.02},
    "twitch-gamers": {"n": 168114, "edges": 6797557, "classes": 2, "h_edge": 0.55, "h_node": 0.56, "h_class": 0.09, "h_adj": 0.09, "li_edge": 0.01, "li_node": 0.02},
    "genius": {"n": 421961, "edges": 922868, "classes": 2, "h_edge": 0.59, "h_node": 0.51, "h_class": 0.02, "h_adj": -0.05, "li_edge": 0.00, "li_node": 0.17},
    "arxiv-year": {"n": 169343, "edges": 1157799, "classes": 5, "h_edge": 0.22, "h_node": 0.29, "h_class": 0.07, "h_adj": 0.01, "li_edge": 0.04, "li_node": 0.12},
    "snap-patents": {"n": 2923922, "edges": 13972547, "classes": 5, "h_edge": 0.22, "h_node": 0.21, "h_class": 0.04, "h_adj": 0.00, "li_edge": 0.02, "li_node": 0.00},
    "wiki": {"n": 1770981, "edges": 242605360, "classes": 5, "h_edge": 0.38, "h_node": 0.28, "h_class": 0.17, "h_adj": 0.15, "li_edge": 0.06, "li_node": 0.04},
    "roman-empire": {"n": 22662, "edges": 32927, "classes": 18, "h_edge": 0.05, "h_node": 0.05, "h_class": 0.02, "h_adj": -0.05, "li_edge": 0.11, "li_node": 0.11},
    "amazon-ratings": {"n": 24492, "edges": 93050, "classes": 5, "h_edge": 0.38, "h_node": 0.38, "h_class": 0.13, "h_adj": 0.14, "li_edge": 0.04, "li_node": 0.04},
    "minesweeper": {"n": 10000, "edges": 39402, "classes": 2, "h_edge": 0.68, "h_node": 0.68, "h_class": 0.01, "h_adj": 0.01, "li_edge": 0.00, "li_node": 0.00},
    "workers": {"n": 11758, "edges": 519000, "classes": 2, "h_edge": 0.59, "h_node": 0.63, "h_class": 0.18, "h_adj": 0.09, "li_edge": 0.01, "li_node": 0.02},
    "questions": {"n": 48921, "edges": 153540, "classes": 2, "h_edge": 0.84, "h_node": 0.90, "h_class": 0.08, "h_adj": 0.02, "li_edge": 0.00, "li_node": 0.01},
}

DATASET_ALIASES = {
    "computers": "amazon-computers",
    "photo": "amazon-photo",
    "roman_empire": "roman-empire",
    "roman empire": "roman-empire",
    "coauthor_cs": "coauthor-cs",
    "coauthorcs": "coauthor-cs",
    "coauthor_physics": "coauthor-physics",
    "coauthorphysics": "coauthor-physics",
    "amazon_computers": "amazon-computers",
    "amazon_photo": "amazon-photo",
    "amazon_ratings": "amazon-ratings",
}


def canonical_dataset_key(dataset_name):
    key = str(dataset_name).strip().replace("_", "-").lower()
    return DATASET_ALIASES.get(key, key)


def parse_syn_cora_name(dataset_name):
    key = canonical_dataset_key(dataset_name)
    match = re.fullmatch(
        r"syn-cora(?:-h(?P<homophily>\d(?:\.\d+)?))?(?:-r(?P<realization>\d+))?",
        key,
    )
    if match is None:
        return None

    homophily = match.group("homophily")
    realization = match.group("realization")
    return (
        None if homophily is None else float(homophily),
        None if realization is None else int(realization),
    )


def _format_metric(value):
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def zhu_homophily_ratio(dataset_name):
    syn_cora = parse_syn_cora_name(dataset_name)
    if syn_cora is not None:
        return syn_cora[0]
    return ZHU_HOMOPHILY_RATIO.get(canonical_dataset_key(dataset_name))


def platonov_dataset_characteristics(dataset_name):
    return PLATONOV_DATASET_CHARACTERISTICS.get(canonical_dataset_key(dataset_name))


def _dataset_kwargs(dataset_name):
    syn_cora = parse_syn_cora_name(dataset_name)
    if syn_cora is None:
        return {"name": dataset_name}

    homophily, realization = syn_cora
    return {
        "name": "syn-cora",
        "syn_cora_homophily": 0.70 if homophily is None else homophily,
        "syn_cora_realization": 1 if realization is None else realization,
    }


def sanity_check_label_informativeness(
    dataset_name,
    li_edge,
    li_node,
    tolerance=LI_REFERENCE_TOLERANCE,
    warn=True,
):
    reference = platonov_dataset_characteristics(dataset_name)
    if reference is None:
        return []

    checks = [
        ("LI_edge", li_edge, reference["li_edge"]),
        ("LI_node", li_node, reference["li_node"]),
    ]
    messages = []
    for metric_name, computed, expected in checks:
        if computed is None:
            continue
        difference = abs(float(computed) - float(expected))
        if difference > tolerance:
            messages.append(
                f"{metric_name} for {dataset_name}: computed={float(computed):.4f}, "
                f"Platonov Table 5={float(expected):.4f}, diff={difference:.4f}"
            )

    if warn and messages:
        warnings.warn(
            "LI sanity check exceeded tolerance "
            f"({tolerance:.3f}): " + "; ".join(messages),
            RuntimeWarning,
            stacklevel=2,
        )
    return messages


@lru_cache(maxsize=None)
def computed_dataset_characteristics(dataset_name):
    from factories.dataset_factory import DatasetFactory

    with contextlib.redirect_stdout(io.StringIO()):
        _, _, _, data = DatasetFactory.get_dataset(
            root_dir=str(SRC_ROOT / "data"),
            **_dataset_kwargs(dataset_name),
        )

    li_edge = getattr(data, "label_informativeness_edge", None)
    li_node = getattr(data, "label_informativeness_node", None)
    graph_summary = getattr(data, "graph_size_summary", None)
    if graph_summary is None:
        graph_summary = compute_graph_size_summary(data.edge_index, data.num_nodes)

    sanity_check_label_informativeness(dataset_name, li_edge, li_node)
    return {
        "li_edge": li_edge,
        "li_node": li_node,
        **graph_summary,
    }


@lru_cache(maxsize=None)
def dataset_label_informativeness(dataset_name):
    characteristics = computed_dataset_characteristics(dataset_name)
    return characteristics["li_edge"], characteristics["li_node"]


def dataset_graph_size_summary(dataset_name, data=None):
    if data is not None:
        graph_summary = getattr(data, "graph_size_summary", None)
        if graph_summary is None:
            graph_summary = compute_graph_size_summary(data.edge_index, data.num_nodes)
        return graph_summary

    reference = platonov_dataset_characteristics(dataset_name)
    if reference is not None:
        num_nodes = int(reference["n"])
        num_edges = int(reference["edges"])
        return {
            "num_nodes": num_nodes,
            "directed_edges": 2 * num_edges,
            "undirected_edges": num_edges,
            "average_degree": float(2 * num_edges / num_nodes),
        }

    characteristics = computed_dataset_characteristics(dataset_name)
    return {
        "num_nodes": characteristics["num_nodes"],
        "directed_edges": characteristics["directed_edges"],
        "undirected_edges": characteristics["undirected_edges"],
        "average_degree": characteristics["average_degree"],
    }


def dataset_graph_size_text(dataset_name, data=None):
    graph_summary = dataset_graph_size_summary(dataset_name, data)
    return (
        f"nodes={int(graph_summary['num_nodes'])} | "
        f"edges={int(graph_summary['undirected_edges'])} | "
        f"avg_deg={float(graph_summary['average_degree']):.2f}"
    )


def dataset_homophily_text(dataset_name):
    syn_cora = parse_syn_cora_name(dataset_name)
    homophily = zhu_homophily_ratio(dataset_name)
    reference = platonov_dataset_characteristics(dataset_name)

    if syn_cora is not None:
        return f"target h={_format_metric(homophily)}"
    elif homophily is None and reference is not None:
        return f"h_edge={_format_metric(reference['h_edge'])}"
    else:
        return f"Zhu h={_format_metric(homophily)}"


def dataset_li_text(dataset_name, data=None):
    if data is None:
        characteristics = computed_dataset_characteristics(dataset_name)
        li_node = characteristics["li_node"]
    else:
        li_node = getattr(data, "label_informativeness_node", None)
        li_edge = getattr(data, "label_informativeness_edge", None)
        sanity_check_label_informativeness(dataset_name, li_edge, li_node)

    return f"LI={_format_metric(li_node)}"


def dataset_metric_summary(dataset_name, data=None):
    homophily_text = dataset_homophily_text(dataset_name)

    return (
        f"{homophily_text} | "
        f"{dataset_graph_size_text(dataset_name, data)} | "
        f"{dataset_li_text(dataset_name, data)}"
    )
