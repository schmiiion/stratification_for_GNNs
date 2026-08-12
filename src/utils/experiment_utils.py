import os


DATASET_ABBREVIATIONS = {
    "actor": "ACT",
    "amazon_ratings": "AMZRAT",
    "amazon-ratings": "AMZRAT",
    "chameleon": "CHAM",
    "chameleon_filtered": "CHAMF",
    "chameleon-filtered": "CHAMF",
    "citeseer": "CITE",
    "coauthor_cs": "COCS",
    "coauthor-cs": "COCS",
    "coauthor_physics": "COPHY",
    "coauthor-physics": "COPHY",
    "computers": "COMP",
    "cornell": "CORN",
    "cora": "CORA",
    "crocodile": "CROC",
    "photo": "PHOTO",
    "pubmed": "PUB",
    "roman_empire": "ROM",
    "roman-empire": "ROM",
    "squirrel": "SQUIR",
    "squirrel_filtered": "SQUIRF",
    "squirrel-filtered": "SQUIRF",
    "syn-cora": "SYN-CORA",
    "syn_cora": "SYN-CORA",
    "texas": "TX",
    "wikics": "WIKICS",
    "wiki_cs": "WIKICS",
    "wiki-cs": "WIKICS",
    "wisconsin": "WIS",
}


def as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def dataset_abbreviation(dataset_name):
    key = dataset_name.replace("-", "_").lower()
    if key in DATASET_ABBREVIATIONS:
        return DATASET_ABBREVIATIONS[key]
    return "".join(ch for ch in dataset_name.upper() if ch.isalnum())[:6]


def dataset_suffix(datasets):
    return "-".join(dataset_abbreviation(dataset) for dataset in datasets)


def build_log_path(filepath, ts, datasets, metric_name):
    directory, filename = os.path.split(filepath)
    _, ext = os.path.splitext(filename)
    suffix = dataset_suffix(datasets)
    new_filename = f"{ts}_{metric_name}_{suffix}{ext or '.csv'}"
    return os.path.join(directory, suffix, new_filename)
