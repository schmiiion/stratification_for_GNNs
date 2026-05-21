import numpy as np
from sklearn.model_selection import StratifiedKFold

from stratify.baseclass import BaseNodeStratifier


class LabelStratifiedKFold(BaseNodeStratifier):
    """
    Label-balanced 60/20/20 rotating split.

    StratifiedKFold creates label-balanced buckets. The base class then rotates
    those buckets into train/validation/test masks.
    """

    def get_folds(self, data):
        y = data.y.cpu().numpy()
        num_nodes = len(y)

        splitter = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.seed,
        )
        dummy_x = np.zeros(num_nodes)
        fold_buckets = [test_idx for _, test_idx in splitter.split(dummy_x, y)]

        folds = self._masks_from_fold_buckets(fold_buckets, num_nodes)
        self._analyze_distributions(data=data, folds=folds, dataset_name=self.dataset_name)

        return folds
