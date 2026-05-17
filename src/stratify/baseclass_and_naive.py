import torch
import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold
from torch_geometric.graphgym import cfg
from torch_geometric.utils import to_networkx, degree
from deap import base, creator, tools, algorithms
import random
import csv
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import wasserstein_distance, ks_2samp

class BaseNodeStratifier(ABC):
    """
    Abstract Base Class for generating k-fold cross-validation masks
    for transductive node classification.
    """

    def __init__(self, cfg, dataset_name, n_splits=5, seed=42):
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.n_splits = n_splits
        self.seed = seed

    @abstractmethod
    def get_folds(self, data):
        """Must be implemented by child classes to return the final list of masks."""
        pass

    def _masks_from_generator(self, generator, num_nodes):
        """
        Consumes an sklearn-style generator (yielding train_idx, test_idx)
        and converts them directly into PyTorch boolean masks that can be fed to torch geometric.
        """
        folds = []

        for train_idx, test_idx in generator:
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            train_mask[train_idx] = True
            test_mask[test_idx] = True

            folds.append({
                'train_mask': train_mask,
                'test_mask': test_mask
            })

        return folds

    def _compute_node_properties(self, data):
        """
        Precomputes the 5 required node-level properties for the entire graph.
        """
        num_nodes = data.num_nodes

        # 1. Degree
        row, col = data.edge_index
        deg = degree(col, num_nodes).cpu().numpy()

        # 2. Neighborhood Heterogeneity (Label-Based)
        # Fraction of neighbors with the SAME class
        y = data.y
        same_class_match = (y[row] == y[col]).float()

        # Sum matches per node
        same_class_neighbors = torch.zeros(num_nodes, device=data.edge_index.device)
        same_class_neighbors.scatter_add_(0, col, same_class_match)

        # Calculate fraction, avoiding division by zero
        deg_tensor = torch.tensor(deg, device=data.edge_index.device)
        heterogeneity = (same_class_neighbors / deg_tensor.clamp(min=1)).cpu().numpy()

        # Convert to NetworkX for structural centralities
        G = to_networkx(data, to_undirected=True)

        # 3. PageRank
        pagerank = np.array(list(nx.pagerank(G).values()))

        # 4. Eigenvector Centrality
        try:
            eigen = np.array(list(nx.eigenvector_centrality(G, max_iter=1000).values()))
        except nx.PowerIterationFailedConvergence:
            print("Eigenvector centrality failed to converge. Defaulting to Degree Centrality.")
            eigen = np.array(list(nx.degree_centrality(G).values()))

        # 5. Clustering Coefficient
        clustering = np.array(list(nx.clustering(G).values()))

        return {
            "Degree": deg,
            "Neighborhood Heterogeneity": heterogeneity,
            "PageRank": pagerank,
            "Eigenvector Centrality": eigen,
            "Clustering Coefficient": clustering
        }

    def _analyze_distributions(self, data, folds, dataset_name):
        """
        Always computes EMD/KS stats and logs them to CSV.
        Only plots Train vs Test distributions if 'plot' is True.
        """
        props = self._compute_node_properties(data)
        prop_names = list(props.keys())

        num_folds = len(folds)

        # 1. ONLY initialize the figure/canvas if plotting is requested
        if self.cfg.plot_fold_statistics:
            fig, axes = plt.subplots(num_folds, 5, figsize=(25, 4 * num_folds))
            fig.suptitle(f'Train vs Test Distributions across Folds - {self.__class__.__name__}', fontsize=20)

        # 2. ALWAYS iterate through folds to calculate and log stats
        for fold_idx, fold in enumerate(folds):
            train_mask = fold['train_mask'].cpu().numpy()
            test_mask = fold['test_mask'].cpu().numpy()

            for prop_idx, prop_name in enumerate(prop_names):
                prop_data = props[prop_name]

                # Extract Train and Test data
                train_data = prop_data[train_mask]
                test_data = prop_data[test_mask]

                # --- ALWAYS COMPUTE & LOG ---
                emd = wasserstein_distance(train_data, test_data)
                ks_stat, _ = ks_2samp(train_data, test_data)

                with open(self.cfg.fold_stats_csv_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        dataset_name,
                        self.__class__.__name__,
                        self.seed,
                        fold_idx + 1,
                        prop_name,
                        emd,
                        ks_stat
                    ])

                # --- ONLY DRAW IF REQUESTED ---
                if self.cfg.plot_fold_statistics:
                    ax = axes[fold_idx, prop_idx]

                    sns.kdeplot(train_data, ax=ax, label='Train', fill=True, alpha=0.4, color='blue',
                                warn_singular=False)
                    sns.kdeplot(test_data, ax=ax, label='Test', fill=True, alpha=0.4, color='orange',
                                warn_singular=False)

                    if fold_idx == 0:
                        ax.set_title(prop_name, fontsize=14)
                    if prop_idx == 0:
                        ax.set_ylabel(f"Fold {fold_idx + 1}\nDensity", fontsize=12)
                    else:
                        ax.set_ylabel("")

                    if fold_idx == 0 and prop_idx == 0:
                        ax.legend()

        # 3. ONLY format and show the final plot if requested
        if self.cfg.plot_fold_statistics:
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()


class LabelStratifiedKFold(BaseNodeStratifier):
    """
    Condition 2 Baseline: Folds are stratified strictly by the target node labels.
    Generates standard train/test splits.
    """

    def get_folds(self, data):
        y = data.y.cpu().numpy()
        num_nodes = len(y)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        dummy_x = np.zeros(num_nodes)

        folds = self._masks_from_generator(skf.split(dummy_x, y), num_nodes)

        self._analyze_distributions(data=data, folds=folds, dataset_name=self.dataset_name)

        return folds


# class WDESDegreeStratifiedKFold(BaseNodeStratifier):
#     """
#     Wasserstein-Driven Evolutionary Stratification.
#     Optimizes splits to balance both node class distribution and structural degree.
#     """
#
#     def __init__(self, n_splits=5, seed=42, n_gen=50, n_pop=100, class_weight=10.0):
#         super().__init__(n_splits=n_splits, seed=seed)
#         self.n_gen = n_gen
#         self.n_pop = n_pop
#         self.class_weight = class_weight  # Multiplier to ensure classes stay strictly balanced
#         self.r = np.asarray([1 / self.n_splits] * self.n_splits)
#
#         random.seed(self.seed)
#         np.random.seed(self.seed)
#
#     def get_folds(self, data):
#         """Main entry point to calculate folds."""
#         self.num_samples = data.num_nodes
#         self.node_classes = data.y.cpu().numpy()
#         self.num_classes = len(np.unique(self.node_classes))
#
#         # Calculate degrees using PyTorch Geometric
#         self.node_degrees = degree(data.edge_index[0], num_nodes=self.num_samples).cpu().numpy()
#         self.desired_n_samples_in_fold = self.r * self.num_samples
#
#         # 1. Run the evolutionary optimization to get the best fold assignment array
#         self.best_folds = np.asarray(self.optimize())
#
#         folds = []
#         # 2. Build the exact train/val/test masks from the resulting fold assignments
#         for i in range(self.n_splits):
#             test_idx = np.where(self.best_folds == i)[0]
#             val_idx = np.where(self.best_folds == ((i + 1) % self.n_splits))[0]
#
#             # Train is everything that isn't Test or Val
#             train_idx = np.where((self.best_folds != i) & (self.best_folds != ((i + 1) % self.n_splits)))[0]
#
#             train_mask = torch.zeros(self.num_samples, dtype=torch.bool)
#             val_mask = torch.zeros(self.num_samples, dtype=torch.bool)
#             test_mask = torch.zeros(self.num_samples, dtype=torch.bool)
#
#             train_mask[train_idx] = True
#             val_mask[val_idx] = True
#             test_mask[test_idx] = True
#
#             folds.append({
#                 'train_mask': train_mask,
#                 'val_mask': val_mask,
#                 'test_mask': test_mask
#             })
#
#         return folds
#
#     # ---------------------------------------------------------
#     # Genetic Algorithm Core Methods
#     # ---------------------------------------------------------
#
#     def _fitness(self, individual):
#         """Calculates the fitness penalty (lower is better)."""
#         fold_penalties = []
#
#         for fold_number in range(self.n_splits):
#             samples_in_fold = [i == fold_number for i in individual]
#
#             # METRIC 1: Degree Distribution (Wasserstein Distance)
#             fold_degrees = self.node_degrees[samples_in_fold]
#             if len(fold_degrees) == 0:
#                 degree_penalty = float('inf')
#             else:
#                 degree_penalty = wasserstein_distance(fold_degrees, self.node_degrees)
#
#             # METRIC 2: Class Proportions (Absolute Difference)
#             fold_classes = self.node_classes[samples_in_fold]
#             class_penalty = 0
#             for c in range(self.num_classes):
#                 global_prop = np.mean(self.node_classes == c)
#                 fold_prop = np.mean(fold_classes == c) if len(fold_classes) > 0 else 0
#                 class_penalty += abs(global_prop - fold_prop)
#
#             fold_penalties.append(degree_penalty + (class_penalty * self.class_weight))
#
#         return (np.mean(fold_penalties),)
#
#     def create_equal_distribution_individual(self, num_params, folds):
#         base_count = num_params // folds
#         remainder = num_params % folds
#         result = []
#         for i in range(folds):
#             result.extend([i] * base_count)
#         for _ in range(remainder):
#             result.append(random.randint(0, folds - 1))
#         random.shuffle(result)
#         return result
#
#     def float_to_int_array(self, float_array):
#         total_sum = sum(float_array)
#         int_array = [int(num) for num in float_array]
#         remaining_sum = int(total_sum) - sum(int_array)
#         for i in range(remaining_sum):
#             int_array[i % len(int_array)] += 1
#         return int_array
#
#     def correct_distribution(self, individual):
#         """Forces the individual to strictly adhere to fold size requirements."""
#         desired_ed = self.float_to_int_array(self.desired_n_samples_in_fold)
#         max_i = len(individual)
#         iter_count = 0
#
#         while iter_count < max_i:
#             if all(abs(individual.count(i) - desired_ed[i]) == 0 for i in range(len(desired_ed))):
#                 break
#             excess_number, deficit_number = -1, -1
#
#             for i in range(len(desired_ed)):
#                 if individual.count(i) > desired_ed[i]:
#                     excess_number = i
#                     break
#             for i in range(len(desired_ed)):
#                 if individual.count(i) < desired_ed[i]:
#                     deficit_number = i
#                     break
#
#             individual[individual.index(excess_number)] = deficit_number
#             iter_count += 1
#
#         return individual
#
#     def uniform_mate_correct(self, ind1, ind2, indpb=0.5):
#         size = min(len(ind1), len(ind2))
#         for i in range(size):
#             if random.random() < indpb:
#                 ind1[i], ind2[i] = ind2[i], ind1[i]
#         ind1 = self.correct_distribution(ind1)
#         ind2 = self.correct_distribution(ind2)
#         return ind1, ind2
#
#     def optimize(self):
#         """Main setup and execution for DEAP."""
#         # Check if DEAP creator attributes already exist to avoid warnings on repeated calls
#         if not hasattr(creator, "FitnessMin"):
#             creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#         if not hasattr(creator, "Individual"):
#             creator.create("Individual", list, fitness=creator.FitnessMin)
#
#         toolbox = base.Toolbox()
#         toolbox.register("individual", tools.initIterate, creator.Individual,
#                          lambda: self.create_equal_distribution_individual(self.num_samples, self.n_splits))
#         toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#         toolbox.register("evaluate", self._fitness)
#         toolbox.register("mate", self.uniform_mate_correct, indpb=0.5)
#         toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
#         toolbox.register("select", tools.selTournament, tournsize=3)
#
#         pop = toolbox.population(n=self.n_pop)
#         hof = tools.HallOfFame(1)
#
#         for ind in pop:
#             ind.fitness.values = toolbox.evaluate(ind)
#
#         print("Starting WDES for Target Stratification...")
#         for _ in tqdm(range(1, self.n_gen + 1), desc="Generations"):
#             pop, _ = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=1,
#                                          halloffame=hof, verbose=False)
#
#         return hof[0]