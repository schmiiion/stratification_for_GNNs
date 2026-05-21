import numpy as np

from stratify.baseclass import BaseNodeStratifier


class RandomKFold(BaseNodeStratifier):
    """
    Purely random 60/20/20 rotating split.

    Labels are ignored. Nodes are shuffled once per seed, divided into buckets.
    """

    def get_folds(self, data):
        num_nodes = data.num_nodes
        rng = np.random.default_rng(self.seed)
        indices = np.arange(num_nodes)
        rng.shuffle(indices)

        fold_buckets = np.array_split(indices, self.n_splits)
        folds = self._masks_from_fold_buckets(fold_buckets, num_nodes)
        self._analyze_distributions(data=data, folds=folds, dataset_name=self.dataset_name)

        return folds
