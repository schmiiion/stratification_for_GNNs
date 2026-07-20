import csv
from datetime import datetime
import os
from pathlib import Path

from utils.experiment_utils import build_log_path


SRC_ROOT = Path(__file__).resolve().parents[1]


class CsvLogger:
    RUN_METRICS_HEADER = [
        "Dataset",
        "StratificationType",
        "Fold_Seed",
        "Fold",
        "Model",
        "Init_Seed",
        "Stopped_Epoch",
        "Best_Epoch",
        "Best_Val_Loss",
        "Best_Val_Accuracy",
        "Test_Loss",
        "Test_Accuracy",
        "Stopped_Early",
        "Early_Stopping_Patience",
    ]

    FOLD_STATS_HEADER = [
        "Dataset",
        "StratificationMethod",
        "StratSeed",
        "DegreeEmd",
        "NeighHetEmd",
        "PageRankEmd",
        "EigCentralityEmd",
        "ClusteringEmd",
        "PropLabelClusterTvd",
    ]

    def __init__(self, cfg):
        self.cfg = cfg

    def initialize(self):
        timestamp = datetime.now().strftime("%m%d-%H%M")

        self.cfg.run_csv_filename = build_log_path(
            self.cfg.run_csv_filename,
            timestamp,
            self.cfg.datasets,
            "RunMetrics",
        )
        self.cfg.run_csv_filename = self._resolve_log_path(self.cfg.run_csv_filename)

        self.cfg.fold_stats_csv_filename = build_log_path(
            self.cfg.fold_stats_csv_filename,
            timestamp,
            self.cfg.datasets,
            "FoldStatistics",
        )
        self.cfg.fold_stats_csv_filename = self._resolve_log_path(self.cfg.fold_stats_csv_filename)
        self.cfg.run_output_dir = os.path.dirname(self.cfg.run_csv_filename)

        self._write_header(self.cfg.run_csv_filename, self.RUN_METRICS_HEADER)
        self._write_header(self.cfg.fold_stats_csv_filename, self.FOLD_STATS_HEADER)

        return self.cfg

    @staticmethod
    def _resolve_log_path(filepath):
        path = Path(filepath).expanduser()
        if not path.is_absolute():
            path = SRC_ROOT / path
        return str(path)

    @staticmethod
    def _write_header(filepath, header):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
