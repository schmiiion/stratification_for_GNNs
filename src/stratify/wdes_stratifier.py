import random
from datetime import datetime
from time import perf_counter

from deap import algorithms, base, creator, tools
import numpy as np
from scipy.stats import wasserstein_distance

from stratify.baseclass import BaseNodeStratifier


class WDESKFold(BaseNodeStratifier):
    """
    Wasserstein Distance Evolutionary Stratification for graph nodes.

    Each individual assigns every node to one fold bucket. The fitness compares
    each bucket's configured node-property distribution to the full graph's
    distribution and minimizes the mean Wasserstein distance across buckets.
    """

    def __init__(self, cfg, dataset_name, seed, n_splits=5, property_name=None):
        super().__init__(cfg=cfg, dataset_name=dataset_name, n_splits=n_splits, seed=seed)
        raw_property_name = property_name
        if raw_property_name is None:
            raw_property_name = self._first_configured_wdes_property()
        try:
            self.property_name = self._canonical_property_name(raw_property_name)
        except ValueError as exc:
            available = ", ".join(self.PROPERTY_NAMES)
            raise ValueError(
                f"Unknown WDES property '{raw_property_name}'. Available properties: {available}"
            ) from exc
        self.stratification_method = f"WDES_{self.PROPERTY_METHOD_NAMES[self.property_name]}"
        self.n_gen = int(self._cfg_get("wdes_n_gen", 50))
        self.n_pop = int(self._cfg_get("wdes_n_pop", 100))
        self.cxpb = float(self._cfg_get("wdes_cxpb", 0.5))
        self.mutpb = float(self._cfg_get("wdes_mutpb", 0.2))
        self.tournament_size = int(self._cfg_get("wdes_tournament_size", 3))

    def _cfg_get(self, key, default):
        if hasattr(self.cfg, "get"):
            return self.cfg.get(key, default)
        return getattr(self.cfg, key, default)

    def _first_configured_wdes_property(self):
        configured_properties = self._cfg_get("wdes_properties", ["Degree"])
        if isinstance(configured_properties, str):
            return configured_properties

        configured_properties = list(configured_properties)
        if configured_properties:
            return configured_properties[0]

        return "Degree"

    def get_folds(self, data):
        props = self._compute_node_properties(data)
        self.property_values = np.asarray(props[self.property_name], dtype=float)
        if not np.all(np.isfinite(self.property_values)):
            raise ValueError(f"WDES property '{self.property_name}' contains non-finite values.")

        self.num_nodes = data.num_nodes
        self.target_fold_counts = np.array(
            [len(bucket) for bucket in np.array_split(np.arange(self.num_nodes), self.n_splits)],
            dtype=np.int64,
        )

        best_assignment = np.asarray(self._optimize(), dtype=np.int64)
        fold_buckets = [
            np.flatnonzero(best_assignment == fold_idx)
            for fold_idx in range(self.n_splits)
        ]

        folds = self._masks_from_fold_buckets(fold_buckets, self.num_nodes)
        self._analyze_distributions(data=data, folds=folds, dataset_name=self.dataset_name)

        return folds

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
