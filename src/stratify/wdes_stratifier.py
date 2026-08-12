import random
from datetime import datetime
from time import perf_counter

from deap import algorithms, base, creator, tools
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.model_selection import StratifiedKFold

from stratify.baseclass import BaseNodeStratifier


class WDESKFold(BaseNodeStratifier):
    """
    Wasserstein Distance Evolutionary Stratification for graph nodes.

    Each individual assigns every node to one fold bucket. The fitness compares
    each bucket's configured node-property distribution to the full graph's
    distribution and minimizes the mean Wasserstein distance across buckets.
    """

    def __init__(self, cfg, dataset_name, seed, n_splits=5, property_name=None, property_options=None):
        super().__init__(
            cfg=cfg,
            dataset_name=dataset_name,
            n_splits=n_splits,
            seed=seed,
            property_options=property_options,
        )
        self.property_name = self.canonical_property_name(property_name)
        self.stratification_method = f"WDES_{self.PROPERTY_METHOD_NAMES[self.property_name]}"
        self.n_gen = int(self.cfg.get("wdes_n_gen", 50))
        self.n_pop = int(self.cfg.get("wdes_n_pop", 100))
        self.cxpb = float(self.cfg.get("wdes_cxpb", 0.5))
        self.mutpb = float(self.cfg.get("wdes_mutpb", 0.2))
        self.tournament_size = int(self.cfg.get("wdes_tournament_size", 3))
        self.sampling_method = self._normalize_sampling_method(
            self.cfg.get("sampling_method", "ga")
        )
        if self.sampling_method == "sklearn":
            self.stratification_method = f"Sklearn_{self.PROPERTY_METHOD_NAMES[self.property_name]}"

    @staticmethod
    def _normalize_sampling_method(method):
        method = str(method).replace("-", "_").lower()
        if method in {"ga", "genetic", "genetic_algorithm"}:
            return "ga"
        if method == "sklearn":
            return "sklearn"
        raise ValueError(
            "sampling_method must be 'ga' or 'sklearn'. Configure skf_num_bins "
            "with one or more candidate bin counts for scalar sklearn properties."
        )

    def get_folds(self, data):
        props = self._compute_node_properties(data)
        self.property_values = np.asarray(props[self.property_name])
        if not np.all(np.isfinite(self.property_values)):
            raise ValueError(f"WDES property '{self.property_name}' contains non-finite values.")
        if self.is_categorical_property(self.property_name):
            self.property_values = self.property_values.astype(np.int64)
        else:
            self.property_values = self.property_values.astype(float)

        self.num_nodes = data.num_nodes
        self.target_fold_counts = np.array(
            [len(bucket) for bucket in np.array_split(np.arange(self.num_nodes), self.n_splits)],
            dtype=np.int64,
        )

        if self.is_categorical_property(self.property_name):
            if self.sampling_method == "ga":
                raise ValueError(
                    f"Property '{self.property_name}' is categorical and only supports "
                    "sampling_method='sklearn'."
                )
            fold_buckets = self._categorical_stratified_kfold_buckets()
        elif self.sampling_method == "sklearn":
            fold_buckets = self._sklearn_buckets()
        else:
            best_assignment = np.asarray(self._optimize(), dtype=np.int64)
            fold_buckets = [
                np.flatnonzero(best_assignment == fold_idx)
                for fold_idx in range(self.n_splits)
            ]

        folds = self._masks_from_fold_buckets(fold_buckets, self.num_nodes)
        self._analyze_distributions(data=data, folds=folds, dataset_name=self.dataset_name)

        return folds

    def _property_strata(self, requested_bins=None):
        if self.is_categorical_property(self.property_name):
            return self.property_values.astype(np.int64)

        if requested_bins is None:
            requested_bins = int(self._sklearn_candidate_bins()[0])
        else:
            requested_bins = int(requested_bins)
        num_bins = min(requested_bins, max(1, self.num_nodes // self.n_splits))

        if num_bins <= 1:
            return np.zeros(self.num_nodes, dtype=np.int64)

        order = np.argsort(self.property_values, kind="mergesort")
        strata = np.empty(self.num_nodes, dtype=np.int64)
        for stratum_idx, node_indices in enumerate(np.array_split(order, num_bins)):
            strata[node_indices] = stratum_idx

        return strata

    def _categorical_stratified_kfold_buckets(self):
        self.stratification_method = (
            f"SklearnCategorical_{self.PROPERTY_METHOD_NAMES[self.property_name]}"
        )
        cluster_count = None
        for option_name in ("propagated_label_num_clusters", "neighborhood_count_num_clusters"):
            if option_name in self.property_options:
                cluster_count = self.property_options[option_name]
                break
        if cluster_count is not None:
            self.stratification_method += f"_k{int(cluster_count)}"
        print(
            f"Starting categorical sklearn StratifiedKFold for {self.dataset_name}: "
            f"property={self.property_name}, strata={len(np.unique(self.property_values))}"
        )
        start_counter = perf_counter()
        self.optimization_start_time = datetime.now().isoformat(timespec="seconds")

        fold_buckets = self._stratified_kfold_buckets_for_bins(requested_bins=None)

        self.optimization_stop_time = datetime.now().isoformat(timespec="seconds")
        self.optimization_seconds = perf_counter() - start_counter
        return fold_buckets

    def _stratified_kfold_buckets_for_bins(self, requested_bins):
        strata = self._property_strata(requested_bins=requested_bins)
        splitter = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.seed,
        )
        dummy_x = np.zeros(self.num_nodes)
        return [test_idx for _, test_idx in splitter.split(dummy_x, strata)]

    def _sklearn_buckets(self):
        candidate_bins = self._sklearn_candidate_bins()
        print(
            f"Starting sklearn StratifiedKFold for {self.dataset_name}: "
            f"property={self.property_name}, candidate_bins={candidate_bins}"
        )
        start_counter = perf_counter()
        self.optimization_start_time = datetime.now().isoformat(timespec="seconds")

        best_bins = None
        best_score = float("inf")
        best_fold_buckets = None
        self.dynamic_skf_scores = {}

        for requested_bins in candidate_bins:
            fold_buckets = self._stratified_kfold_buckets_for_bins(requested_bins)
            score = self._target_property_mean_emd(fold_buckets)
            self.dynamic_skf_scores[int(requested_bins)] = score
            if score < best_score:
                best_bins = int(requested_bins)
                best_score = score
                best_fold_buckets = fold_buckets

        self.selected_skf_num_bins = best_bins
        self.selected_skf_target_emd = best_score
        self.stratification_method = f"Sklearn_{self.PROPERTY_METHOD_NAMES[self.property_name]}"
        self.optimization_stop_time = datetime.now().isoformat(timespec="seconds")
        self.optimization_seconds = perf_counter() - start_counter

        print(
            f"Selected bins={best_bins} for {self.dataset_name}: "
            f"property={self.property_name}, mean_target_emd={best_score:.6g}"
        )
        return best_fold_buckets

    def _sklearn_candidate_bins(self):
        candidate_bins = self.cfg.get("skf_num_bins", None)
        if candidate_bins is None:
            candidate_bins = [20]
        if isinstance(candidate_bins, (str, int, float)):
            candidate_bins = [candidate_bins]

        cleaned_bins = []
        for value in candidate_bins:
            requested_bins = int(value)
            if requested_bins < 1:
                raise ValueError("skf_num_bins must contain positive integers.")
            cleaned_bins.append(requested_bins)

        return sorted(set(cleaned_bins))

    def _target_property_mean_emd(self, fold_buckets):
        distances = []
        for fold_bucket in fold_buckets:
            fold_values = self.property_values[np.asarray(fold_bucket, dtype=np.int64)]
            if len(fold_values) == 0:
                return float("inf")
            distances.append(wasserstein_distance(fold_values, self.property_values))
        return float(np.mean(distances))

    def _fitness(self, individual):
        assignment = np.asarray(individual, dtype=np.int64)
        distances = []

        for fold_idx in range(self.n_splits):
            fold_values = self.property_values[assignment == fold_idx]
            if len(fold_values) == 0:
                return (float("inf"),)
            distances.append(wasserstein_distance(fold_values, self.property_values))

        return (float(np.mean(distances)),)

    def _create_individual(self):
        """Create one exact-size random fold assignment."""
        individual = []
        for fold_idx, count in enumerate(self.target_fold_counts):
            individual.extend([fold_idx] * int(count))

        random.shuffle(individual)
        return individual

    def _correct_distribution(self, individual):
        current_counts = np.bincount(individual, minlength=self.n_splits)

        excess_indices = []
        deficits = []
        for fold_idx, (current, target) in enumerate(zip(current_counts, self.target_fold_counts)):
            difference = int(current - target)
            if difference > 0:
                fold_excess_indices = [
                    idx for idx, assignment in enumerate(individual)
                    if assignment == fold_idx
                ]
                random.shuffle(fold_excess_indices)
                excess_indices.extend(fold_excess_indices[:difference])
            elif difference < 0:
                deficits.extend([fold_idx] * abs(difference))

        random.shuffle(excess_indices)
        random.shuffle(deficits)
        for idx, fold_idx in zip(excess_indices, deficits):
            individual[idx] = fold_idx

        corrected_counts = np.bincount(individual, minlength=self.n_splits)
        if not np.array_equal(corrected_counts, self.target_fold_counts):
            raise ValueError("WDES failed to correct fold-bucket sizes after crossover.")

        return individual

    def _uniform_mate_correct(self, ind1, ind2, indpb=0.5):
        """Uniform crossover followed by exact fold-size repair."""
        for idx in range(min(len(ind1), len(ind2))):
            if random.random() < indpb:
                ind1[idx], ind2[idx] = ind2[idx], ind1[idx]

        return self._correct_distribution(ind1), self._correct_distribution(ind2)

    def _optimize(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

        if not hasattr(creator, "FitnessMinWDES"):
            creator.create("FitnessMinWDES", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "IndividualWDES"):
            creator.create("IndividualWDES", list, fitness=creator.FitnessMinWDES)

        toolbox = base.Toolbox()
        toolbox.register(
            "individual",
            tools.initIterate,
            creator.IndividualWDES,
            self._create_individual,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._fitness)
        toolbox.register("mate", self._uniform_mate_correct, indpb=0.5)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

        population = toolbox.population(n=self.n_pop)
        hall_of_fame = tools.HallOfFame(1)

        for individual in population:
            individual.fitness.values = toolbox.evaluate(individual)

        print(
            f"Starting WDES for {self.dataset_name}: property={self.property_name}, "
            f"population={self.n_pop}, generations={self.n_gen}"
        )
        start_counter = perf_counter()
        self.optimization_start_time = datetime.now().isoformat(timespec="seconds")
        algorithms.eaSimple(
            population,
            toolbox,
            cxpb=self.cxpb,
            mutpb=self.mutpb,
            ngen=self.n_gen,
            halloffame=hall_of_fame,
            verbose=False,
        )
        self.optimization_stop_time = datetime.now().isoformat(timespec="seconds")
        self.optimization_seconds = perf_counter() - start_counter

        return hall_of_fame[0]
