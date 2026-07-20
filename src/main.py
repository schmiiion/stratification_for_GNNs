import torch
import torch.nn.functional as F
from factories.dataset_factory import DatasetFactory
from factories.model_factory import get_models
from factories.stratifier_factory import get_stratifiers
import hydra
from omegaconf import DictConfig
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.csv_logger import CsvLogger
from utils.experiment_utils import as_list
from utils.training_utils import (
    clone_state_dict,
    evaluate_mask,
    run_metrics_writer_process,
)


def worker_task(
        dataset_id, stratification_name, fold_idx, fold, fold_seed, model_name, init_seed, cfg,
        data, adj_hop1, adj_hop2, input_dim, output_dim, log_queue):

    torch.set_num_threads(1)
    device = torch.device('cpu')

    torch.manual_seed(init_seed)
    np.random.seed(init_seed)

    model = get_models(cfg, input_dim, output_dim, model_name).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    model_x = data.h2gcn_x if model_name == "H2GCN" else data.x
    model_adj = {
        "edge_idx": data.edge_index,
        "adj1_hop": adj_hop1,
        "adj2_hop": adj_hop2,
    }

    max_epochs = int(cfg.max_epochs)
    early_stopping_patience = int(cfg.get("early_stopping_patience", 100))

    best_state = None
    best_epoch = 0
    best_val_loss = float("inf")
    best_val_acc = float("-inf")
    epochs_without_improvement = 0
    stopped_epoch = max_epochs
    stopped_early = False

    # --- TRAINING LOOP ---
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(model_x, model_adj)
        loss = F.nll_loss(out[fold["train_mask"]], data.y[fold["train_mask"]])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(model_x, model_adj)
            val_loss, val_acc = evaluate_mask(logits, data.y, fold["val_mask"])

        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = clone_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            stopped_epoch = epoch
            stopped_early = True
            break

    # --- EVALUATION ---
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(model_x, model_adj)
        test_loss, test_acc = evaluate_mask(logits, data.y, fold["test_mask"])

    # --- FIRE AND FORGET LOGGING ---
    # Send the data to the central logger instantly
    log_queue.put(
        [
            dataset_id,
            stratification_name,
            fold_seed,
            fold_idx + 1,
            model_name,
            init_seed,
            stopped_epoch,
            best_epoch,
            best_val_loss,
            best_val_acc,
            test_loss,
            test_acc,
            stopped_early,
            early_stopping_patience,
        ])

    return (
        f"Finished {dataset_id} | {stratification_name} | {model_name} | "
        f"Foldseed {fold_seed} | Fold {fold_idx + 1} | Initseed {init_seed} | "
        f"Best epoch {best_epoch} | Test acc {test_acc:.4f}"
    )


def configured_dataset_requests(cfg):
    for dataset_name in as_list(cfg.datasets):
        canonical_name = str(dataset_name).replace("_", "-").lower()
        if canonical_name != "syn-cora":
            yield str(dataset_name), {"name": str(dataset_name)}
            continue

        ratios = cfg.get("syn_cora_ratios", None)
        if ratios is None:
            ratios = cfg.get("syn-cora-ratio", None)
        if ratios is None:
            ratios = cfg.get("syn_cora_ratio", [0.70])

        realizations = cfg.get("syn_cora_realizations", None)
        if realizations is None:
            realizations = cfg.get("syn-cora-realizations", [1])

        for ratio in as_list(ratios):
            for realization in as_list(realizations):
                ratio_value = float(ratio)
                realization_value = int(realization)
                run_dataset_name = f"syn-cora-h{ratio_value:.2f}-r{realization_value}"
                yield run_dataset_name, {
                    "name": "syn-cora",
                    "syn_cora_homophily": ratio_value,
                    "syn_cora_realization": realization_value,
                }


def maybe_plot_propagated_label_clusters(cfg, dataset_id, data):
    if not cfg.get("plot_propagated_label_clusters", False):
        return

    from stratify.plot_propagated_label_clusters import plot_propagated_label_clusters

    fold_seeds = as_list(cfg.get("fold_seeds", [0]))
    plot_seed = int(fold_seeds[0]) if fold_seeds else 0
    plot_propagated_label_clusters(
        cfg=cfg,
        dataset_name=dataset_id,
        data=data,
        strat_seed=plot_seed,
    )


def maybe_plot_gap_statistic_curve(cfg, dataset_id, data):
    if not cfg.get("plot_gap_statistic_curve", False):
        return

    from stratify.plot_gap_statistic_curve import plot_gap_statistic_curve

    fold_seeds = as_list(cfg.get("fold_seeds", [0]))
    plot_seed = int(fold_seeds[0]) if fold_seeds else 0
    plot_gap_statistic_curve(
        cfg=cfg,
        dataset_name=dataset_id,
        data=data,
        strat_seed=plot_seed,
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    CsvLogger(cfg).initialize()

    # --- 1. Setup the Queue and Central Logger ---
    manager = multiprocessing.Manager()
    log_queue = manager.Queue()

    # Start the logger as a background process
    metrics_writer = multiprocessing.Process(
        target=run_metrics_writer_process,
        args=(log_queue, cfg.run_csv_filename),
    )
    metrics_writer.start()

    # num_workers = max(1, multiprocessing.cpu_count() - 2)
    num_workers = int(cfg.get("num_workers", 4))
    print(f"Starting ProcessPoolExecutor with {num_workers} workers.")

    # DATASET LOOP
    for dataset_id, dataset_kwargs in configured_dataset_requests(cfg):
        print(f"\n{'=' * 40}\nDATASET: {dataset_id}\n{'=' * 40}")

        dataset, input_dim, output_dim, data = DatasetFactory.get_dataset(**dataset_kwargs)
        maybe_plot_propagated_label_clusters(cfg, dataset_id, data)
        maybe_plot_gap_statistic_curve(cfg, dataset_id, data)

        adj_hop1, adj_hop2 = None, None
        if "H2GCN" in cfg.model_names:
            data.h2gcn_x = DatasetFactory.row_normalize_features(data.x)
            adj_hop1, adj_hop2 = DatasetFactory.precompute_h2gcn_hops(data.edge_index, data.num_nodes)

        for fold_seed in cfg.fold_seeds:
            stratifiers = get_stratifiers(cfg=cfg, dataset_name=dataset_id, seed=fold_seed, data=data)
            for stratifier in stratifiers:
                folds = stratifier.get_folds(data)
                stratification_name = stratifier.stratification_method

                # 2. Build tasks
                tasks = []
                for fold_idx, fold in enumerate(folds):
                    for model_name in cfg.model_names:
                        for init_seed in cfg.init_seeds:
                            tasks.append(
                                (
                                    dataset_id,
                                    stratification_name,
                                    fold_idx,
                                    fold,
                                    fold_seed,
                                    model_name,
                                    init_seed,
                                    cfg,
                                    data,
                                    adj_hop1,
                                    adj_hop2,
                                    input_dim,
                                    output_dim,
                                    log_queue,
                                )
                            )

                # 3. Execute tasks
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(worker_task, *args) for args in tasks]

                    for future in as_completed(futures):
                        try:
                            print(future.result())
                        except Exception as exc:
                            print(f"A worker generated an exception: {exc}")
                            raise exc

        del dataset, data

    # --- 4. Clean Shutdown ---
    # Once all loops and executors are finished, tell the logger to stop.
    log_queue.put("KILL")
    metrics_writer.join()


if __name__ == "__main__":
    main()
