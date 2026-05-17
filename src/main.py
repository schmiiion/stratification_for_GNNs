import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from factories.dataset_factory import DatasetFactory
from factories.model_factory import get_models
import numpy as np
import hydra
from omegaconf import DictConfig
import csv
import os
from datetime import datetime

from stratify.baseclass_and_naive import LabelStratifiedKFold

def add_timestamp_to_path(filepath, ts):
    directory, filename = os.path.split(filepath)
    new_filename = f"{ts}_{filename}"
    return os.path.join(directory, new_filename)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg.run_csv_filename = add_timestamp_to_path(cfg.run_csv_filename, timestamp)
    cfg.fold_stats_csv_filename = add_timestamp_to_path(cfg.fold_stats_csv_filename, timestamp)

    os.makedirs(os.path.dirname(cfg.run_csv_filename), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.fold_stats_csv_filename), exist_ok=True)

    with open(cfg.run_csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", "StratificationType", "Fold_Seed", "Model", "Init_Seed", "Epoch", "Test_Accuracy"])

    with open(cfg.fold_stats_csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", "Stratification_Type", "Fold_Seed", "Fold", "Property", "EMD", "KS_Stat"])



    #DATASET LOOP
    for dataset_id in cfg.datasets:
        print(f"\n{'=' * 40}\nDATASET: {dataset_id}\n{'=' * 40}")

        dataset, input_dim, output_dim, data = DatasetFactory.get_dataset(name=dataset_id, device=device)

        # Generate Folds
        for fold_seed in cfg.fold_seeds:
            stratifier = LabelStratifiedKFold(cfg=cfg, dataset_name=dataset_id, n_splits=cfg.num_folds, seed=fold_seed)
            folds = stratifier.get_folds(data)

            #FOLDS
            for fold in folds:

                #MODELS
                for model_name in cfg.model_names:

                    #MODEL INIT SEEDS
                    for init_seed in cfg.init_seeds:
                        print(f"Fold: {fold_seed} | Model: {model_name} | Init: {init_seed}")

                        torch.manual_seed(init_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed(init_seed)

                        all_models = get_models(cfg, input_dim, output_dim)
                        model = next(m for name, m in all_models if name == model_name)
                        model = model.to(device)

                        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

                        # --- TRAINING LOOP ---
                        for epoch in range(cfg.max_epochs):
                            model.train()
                            optimizer.zero_grad()
                            out = model(data.x, data.edge_index)
                            loss = F.nll_loss(out[fold["train_mask"]], data.y[fold["train_mask"]])
                            loss.backward()
                            optimizer.step()

                            # --- EVALUATION (Every cfg.test_all_n epochs OR on the very last epoch) ---
                            if (epoch + 1) % cfg.test_all_n == 0 or epoch == cfg.max_epochs - 1:
                                model.eval()

                                with torch.no_grad():
                                    logits = model(data.x, data.edge_index)
                                    preds = logits[fold["test_mask"]].argmax(dim=1)
                                    acc = (preds == data.y[fold["test_mask"]]).sum().item() / fold["test_mask"].sum().item()

                                # LOGGING
                                with open(cfg.run_csv_filename, mode='a', newline='') as file:
                                    writer = csv.writer(file)
                                    writer.writerow(
                                        [dataset_id, "ClassBasedStratification", fold_seed, model_name, init_seed,
                                         epoch + 1, acc])

        del dataset, data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()