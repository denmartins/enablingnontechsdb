import qbe.util as util
import pandasql as pdsql
import numpy as np
from sklearn.utils import check_random_state
from sklearn.metrics import precision_recall_fscore_support

class AbstractFitnessFunction(object):
    def calulate(self, individual, raw_individual=None):
        raise NotImplementedError('Method not implemented.')

class PersonalizedFitnessFunction(AbstractFitnessFunction):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        self.true_positives = set([i for i in range(len(self.y_train)) if self.y_train[i] == 1])
        self.true_negatives = set([i for i in range(len(self.y_train)) if self.y_train[i] == 0])
        
    def calculate(self, individual, raw_individual):
        predicted = [bool(individual(*x)) for x in self.X_train]
    
        positive_pred = set([i for i in range(len(predicted)) if predicted[i] == 1])
        negative_pred = set([i for i in range(len(predicted)) if predicted[i] == 0])
        
        pos_hits = len(self.true_positives & positive_pred)
        neg_hits = len(self.true_negatives - negative_pred)

        fitness = 1000
        size_selected = len(positive_pred)
        
        ind_size = len(util.genetic_2_predicate(raw_individual).split('and'))
        if size_selected > 0:
            fitness = len(self.true_positives) - pos_hits + 2.0*neg_hits + 0.05*ind_size + 0.1*size_selected
            #fitness = 1 - precision_recall_fscore_support(self.y_train, predicted, average='binary')[2] + 0.05*ind_size

        return fitness

class QueryDiscoveryFitnessFunction(AbstractFitnessFunction):
    def __init__(self, dataframe, desired_indexes):
        super().__init__()
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

    def calculate(self, individual, raw_individual=None):
        total_positives = len([p for p in self.training_target if p == 1])
        total_negatives = len(self.training_target) - total_positives
        true_positives = 0
        true_negatives = 0
        for i in range(len(self.training_target)):
            if self.training_target[i] == 1 and bool(individual(*self.training_data[i])) == 1:
                true_positives += 1
            elif self.training_target[i] == 0 and bool(individual(*self.training_data[i])) == 0:
                true_negatives += 1

        def protected_division(a, b):
            try:
                return a / b
            except ZeroDivisionError:
                return 1

        specificity = protected_division(true_negatives, total_negatives)
        recall = protected_division(true_positives, total_positives)

        if recall < 1.0:
            recall = 0.0
            
        return 100 - 100 * (recall * specificity)

class SQLFitnessFunction(AbstractFitnessFunction):
    def __init__(self, desired_set, undesired_set, pysql):
        super().__init__()
        self.desired_set = desired_set
        self.undesired_set = undesired_set
        self.pysql = pysql
    
    def calculate_genetic_fitness(self, genetic_solution):
        query = util.classification_rule_2_sql(util.genetic_2_predicate(genetic_solution))
        return self.base_calculate(query),
    
    def moop_calculate_genetic_fitness(self, genetic_solution):
        query = util.classification_rule_2_sql(util.genetic_2_predicate(genetic_solution))
        return self.base_calculate(query, multi_objective=True)

    def calculate(self, solution, raw_individual=None):
        query = util.classification_rule_2_sql(solution)
        return self.base_calculate(query)
    
    def base_calculate(self, query, multi_objective=False):
        try:
            actual_output = self.pysql(query)
            actual_set = util.convert_nparray_to_set(actual_output)
        except pdsql.PandaSQLException: # not valid SQL query
            actual_set = set()

        recall, specificity = self.get_recall_and_specificity(actual_set)
        
        factor = 100
        if multi_objective:
            return factor - factor*recall, factor - factor*specificity
        else:
            return factor - factor * (recall * specificity)
    
    def get_recall_and_specificity(self, actual_set):
        if len(actual_set) < 1 or len(actual_set) >= len(self.desired_set | self.undesired_set):
            # Penalize undesired solutions
            recall = 0 
            specificity = 0
        else:
            recall = util.get_recall(actual_set, self.desired_set)
            specificity = util.get_specificity(actual_set, self.desired_set, self.undesired_set)

            # Penalize solutions that do not produce recall 1
            if recall < 1:
                recall = 0
                specificity = 0
        
        return recall, specificity