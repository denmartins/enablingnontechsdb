import util
import random, operator
import numpy as np
from sklearn.utils import check_random_state
from gplearn.functions import make_function
from gplearn.fitness import make_fitness
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor

class GeneticProgLearnQBE(object):
    def __init__(self, dataframe, desired_indexes):
        total_indexes = [t for t in range(dataframe.shape[0])]

        all_data = dataframe.copy(True)
        all_data['class'] = [int(x in desired_indexes) for x in total_indexes]

        # Shuffle data
        rng = check_random_state(0)
        perm = rng.permutation(all_data.shape[0])
        shuffled_data = all_data.values[perm]

        # Subtract the index of the class value which is always at the end of the vector
        num_features = all_data.shape[1] - 1
        
        self.training_data = shuffled_data[:,:num_features]
        self.training_target = shuffled_data[:, -1]
        self.columns = dataframe.columns

        self.configure_genetic_programming(dataframe)
    
    def configure_genetic_programming(self, dataframe):
        # Define custom logical operators for our Genetic Programming
        def and_op(x1, x2):
            return np.logical_and(x1, x2)

        def or_op(x1, x2):
            return np.logical_or(x1, x2)

        def gt_op(x1, x2):
            return operator.gt(x1, x2)

        def lt_op(x1, x2):
            return operator.lt(x1, x2)

        def eq_op(x1, x2):
            return np.equal(x1, x2)

        def ge_op(x1, x2):
            return operator.ge(x1, x2)
        
        def le_op(x1, x2):
            return operator.le(x1, x2)

        def ne_op(x1, x2):
            return np.not_equal(x1, x2)

        # Use make_function to create function nodes in our GP
        and_ = make_function(function=and_op, name='and_', arity=2)
        or_ = make_function(function=or_op, name='or_', arity=2)
        gt_ = make_function(function=gt_op, name='gt', arity=2)
        lt_ = make_function(function=lt_op, name='lt', arity=2)
        eq_ = make_function(function=eq_op, name='eq', arity=2)
        ge_ = make_function(function=ge_op, name='ge', arity=2)
        le_ = make_function(function=ge_op, name='le', arity=2)        
        ne_ = make_function(function=ne_op, name='ne', arity=2) 

        # Create the function set
        self.function_set = [gt_, ge_, lt_, le_, eq_, or_, and_]
        
        def fitness(y, y_pred, w):
            total_positives = len([p for p in y if p == 1])
            total_negatives = len(y) - total_positives
            true_positives = 0
            true_negatives = 0
            for i in range(len(y)):
                if y[i] == 1 and y_pred[i] == 1:
                    true_positives += 1
                elif y[i] == 0 and y_pred[i] == 0:
                    true_negatives += 1

            recall = 0
            try:
                recall = true_positives / total_positives
            except ZeroDivisionError:
                recall = 0

            specificity = 0
            try:
                specificity = true_negatives / total_negatives
            except ZeroDivisionError:
                specificity = 0

            if recall < 1.0:
                recall = 0

            return 100 - 100 * (recall * specificity)

        self.fitness_function = make_fitness(fitness, greater_is_better=False)
        
    def search_best_predicate(self, population_size, crossover_rate, mutation_rate, num_generations, max_gen_without_gain, verbose=True):
        # Setup parameters of our Genetic Programming algorithm
        # It will create a estimator (classifier)
        gp_estimator = SymbolicRegressor(population_size, generations=num_generations,
                                    verbose=int(verbose), function_set=self.function_set,
                                    metric=self.fitness_function, init_depth=(2,10))

        gp_estimator.fit(self.training_data, self.training_target)

        best_predicate = str(gp_estimator._program)

        for i in range(len(self.columns)):
            arg = 'X' + str(i)
            best_predicate = best_predicate.replace(arg, self.columns[i])
        
        return util.genetic_2_predicate(best_predicate)