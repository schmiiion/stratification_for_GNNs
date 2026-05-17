import torch
import torch.nn.functional as F
from factories.dataset_factory import DatasetFactory
from factories.model_factory import get_models
import hydra
from omegaconf import DictConfig
import csv
import os
from datetime import datetime
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from stratify.baseclass_and_naive import LabelStratifiedKFold

def add_timestamp_to_path(filepath, ts):
    directory, filename = os.path.split(filepath)
    new_filename = f"{ts}_{filename}"
    return os.path.join(directory, new_filename)

def logger_process(queue, run_csv_filename):
    """Process running in the background. It listens to the queue and writes incoming data to the csv one row at a time."""
    while True:
        record = queue.get()
        if record == 'KILL':
            break

        with open(run_csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(record)


def worker_task(dataset_id, fold_idx, fold, fold_seed, model_name, init_seed, cfg, data, input_dim, output_dim, log_queue):
    torch.set_num_threads(1)
    device = torch.device('cpu')

    torch.manual_seed(init_seed)
    np.random.seed(init_seed)

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

        # --- EVALUATION ---
        if (epoch + 1) % cfg.test_all_n == 0 or epoch == cfg.max_epochs - 1:
            model.eval()

            with torch.no_grad():
                logits = model(data.x, data.edge_index)
                preds = logits[fold["test_mask"]].argmax(dim=1)
                acc = (preds == data.y[fold["test_mask"]]).sum().item() / fold["test_mask"].sum().item()

            # --- FIRE AND FORGET LOGGING ---
            # Send the data to the central logger instantly
            log_queue.put(
                [dataset_id, "ClassBasedStratification", fold_seed, fold_idx + 1, model_name, init_seed, epoch + 1,
                 acc])

    return f"Finished {dataset_id} | {model_name} | Fold {fold_idx + 1} | Seed {init_seed}"


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg.run_csv_filename = add_timestamp_to_path(cfg.run_csv_filename, timestamp)
    cfg.fold_stats_csv_filename = add_timestamp_to_path(cfg.fold_stats_csv_filename, timestamp)

    os.makedirs(os.path.dirname(cfg.run_csv_filename), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.fold_stats_csv_filename), exist_ok=True)

    with open(cfg.run_csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Dataset", "StratificationType", "Fold_Seed", "Fold", "Model", "Init_Seed", "Epoch", "Test_Accuracy"])

    with open(cfg.fold_stats_csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", "Stratification_Type", "Fold_Seed", "Fold", "Property", "EMD", "KS_Stat"])

    # --- 1. Setup the Queue and Central Logger ---
    manager = multiprocessing.Manager()
    log_queue = manager.Queue()

    # Start the logger as a background process
    logger = multiprocessing.Process(target=logger_process, args=(log_queue, cfg.run_csv_filename))
    logger.start()

    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"Starting ProcessPoolExecutor with {num_workers} workers.")

    # DATASET LOOP
    for dataset_id in cfg.datasets:
        print(f"\n{'=' * 40}\nDATASET: {dataset_id}\n{'=' * 40}")

        dataset, input_dim, output_dim, data = DatasetFactory.get_dataset(name=dataset_id, device='cpu')

        for fold_seed in cfg.fold_seeds:
            stratifier = LabelStratifiedKFold(cfg=cfg, dataset_name=dataset_id, n_splits=cfg.num_folds, seed=fold_seed)
            folds = stratifier.get_folds(data)

            # 2. Build tasks
            tasks = []
            for fold_idx, fold in enumerate(folds):
                for model_name in cfg.model_names:
                    for init_seed in cfg.init_seeds:
                        tasks.append(
                            (dataset_id, fold_idx, fold, fold_seed, model_name, init_seed, cfg, data, input_dim,
                             output_dim, log_queue)
                        )

            # 3. Execute tasks
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(worker_task, *args) for args in tasks]

                for future in as_completed(futures):
                    try:
                        print(future.result())
                    except Exception as exc:
                        print(f"A worker generated an exception: {exc}")

        del dataset, data

    # --- 4. Clean Shutdown ---
    # Once all loops and executors are finished, tell the logger to stop.
    log_queue.put("KILL")
    logger.join()  # Wait for the logger to finish writing its final queue items


if __name__ == "__main__":
    main()