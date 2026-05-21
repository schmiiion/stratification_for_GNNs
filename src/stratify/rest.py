"""
Retired stratifier drafts.

This file keeps the old commented-out WDES prototype out of the active base
class. It is intentionally not imported by the training pipeline.
"""

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
