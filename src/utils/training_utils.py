import csv

import torch.nn.functional as F


def evaluate_mask(logits, labels, mask):
    masked_logits = logits[mask]
    masked_labels = labels[mask]
    loss = F.nll_loss(masked_logits, masked_labels).item()
    preds = masked_logits.argmax(dim=1)
    acc = (preds == masked_labels).float().mean().item()
    return loss, acc


def clone_state_dict(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.state_dict().items()
    }


def run_metrics_writer_process(queue, run_csv_filename):
    """Listen to the queue and write incoming records to the run metrics CSV."""
    while True:
        record = queue.get()
        if record == "KILL":
            break

        with open(run_csv_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(record)
