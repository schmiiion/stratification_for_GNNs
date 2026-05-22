import os


DATASET_ABBREVIATIONS = {
    "actor": "ACT",
    "chameleon": "CHAM",
    "chameleon_filtered": "CHAMF",
    "chameleon-filtered": "CHAMF",
    "citeseer": "CITE",
    "computers": "COMP",
    "cora": "CORA",
    "crocodile": "CROC",
    "photo": "PHOTO",
    "pubmed": "PUB",
    "roman_empire": "ROM",
    "roman-empire": "ROM",
    "squirrel": "SQUIR",
    "squirrel_filtered": "SQUIRF",
    "squirrel-filtered": "SQUIRF",
    "texas": "TX",
    "wisconsin": "WIS",
}


def as_list(value):
    if isinstance(value, str):
        return [value]
    return list(value)


def dataset_abbreviation(dataset_name):
    key = dataset_name.replace("-", "_").lower()
    if key in DATASET_ABBREVIATIONS:
        return DATASET_ABBREVIATIONS[key]
    return "".join(ch for ch in dataset_name.upper() if ch.isalnum())[:6]


def build_log_path(filepath, ts, datasets, metric_name):
    directory, filename = os.path.split(filepath)
    _, ext = os.path.splitext(filename)
    dataset_suffix = "-".join(dataset_abbreviation(dataset) for dataset in datasets)
    new_filename = f"{ts}_{metric_name}_{dataset_suffix}{ext or '.csv'}"
    return os.path.join(directory, new_filename)
