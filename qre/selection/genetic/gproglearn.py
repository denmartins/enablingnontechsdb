from gplearn.genetic import SymbolicClassifier, SymbolicRegressor
from gplearn.functions import make_function
from gplearn.fitness import make_fitness
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.utils import check_random_state
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report
import operator

from fitnessfunction import ClassificationFitness

if __name__ == "__main__":
    adult = 'C:\\Users\\d_mart04\\Google Drive\\Doutorado\\gitworkspace\\phd-query-synthesis\\PhDCoding\\datasets\\adult_dataset\\adult.data.txt'
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'educationnum', 
        'maritalstatus', 'occupation', 'relationship', 
        'race', 'sex', 'capitalgain', 'capitalloss', 'hoursperweek',
            'nativecountry']
    input_table = pd.read_table(adult, sep=', ', names=columns, index_col=False, engine='python')
    input_table['custom_index'] = [i for i in range(input_table.shape[0])]
    input_table.fillna('ISNULL', inplace=True)
    label = input_table.query('age > 50 and race in ["Black", "Asian-Pac-Islander"]')['custom_index']
    output_table = [int(i in label) for i in range(input_table.shape[0])]
    
    categorical_columns = set(input_table.columns) - set(input_table._get_numeric_data().columns)
    input_table = pd.get_dummies(input_table, columns=list(categorical_columns), dtype='int16')

    for col in input_table.columns:
        unique_values = input_table[col].unique()
        if len(unique_values) == 1:
            del input_table[col] # Removing uninformative columns

    # Renaming columns to avoid syntax errors
    renamed_columns = [col.replace('.0', '') for col in input_table.columns]
    input_table.columns = renamed_columns

    X = input_table.values
    y = output_table

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
    function_set = [
        gt_, 
        lt_, 
        eq_,
        or_, 
        and_
    ]

    def sigmoid(x1):
        with np.errstate(over='ignore', under='ignore'):
            return 1 / (1 + np.exp(np.logical_not(x1)))

    sigfunc = make_function(function=sigmoid, name='sigmoid', arity=1)

    def fitness(y, y_pred, w):
        true_positives = 0
        true_negatives = 0
        false_negatives = 0
        false_positives = 0
        
        for i in range(len(y)):
            if y[i] == 1 and y_pred[i] == 1:
                true_positives += 1
            elif y[i] == 0 and y_pred[i] == 0:
                true_negatives += 1
            elif y[i] == 1 and y_pred[i] == 0:
                false_negatives += 1
            elif y[i] == 0 and y_pred[i] == 1:
                false_positives += 1
        
        recall = 0
        try:
            recall = true_positives / (true_positives + false_negatives)
        except ZeroDivisionError:
            recall = 0

        precision = 0
        try:
            precision = true_positives / (true_positives + false_positives)
        except ZeroDivisionError:
            precision = 0

        try:
            fscore = 2 * precision * recall /(precision + recall)
        except ZeroDivisionError:
            fscore = 0

        return fscore

    fitness_function = make_fitness(fitness, greater_is_better=False)

    est = SymbolicClassifier(parsimony_coefficient=.01,
                         feature_names=input_table.columns, 
                         generations=100,
                         population_size=500,
                         n_jobs=-1,
                         function_set=function_set, 
                         #metric=fitness_function,
                         transformer=sigfunc)
    est.fit(X,y)

    y_true = y
    y_score = est.predict_proba(X)[:,1]
    print('ROC: ', roc_auc_score(y_true, y_score))

    y_predicted = est.predict(X)
    print(classification_report(y_true, y_predicted))

    print(est._program.fitness_)
    print(est._program)