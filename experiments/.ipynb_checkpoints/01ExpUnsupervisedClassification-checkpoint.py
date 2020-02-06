import pandas as pd
import numpy as np
import os
import random
import math

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC, OneClassSVM
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from minisom import MiniSom
import somoclu

import operator
from sklearn.utils import check_random_state
from gplearn.functions import make_function
from gplearn.fitness import make_fitness
from gplearn.genetic import SymbolicClassifier

from gpdeap import GeneticProgrammingQBE

def create_som(data, x_size, y_size):
    learning_rate = 0.1
    sigma = max(x_size, y_size)*0.5

    som = MiniSom(x_size, y_size, data.shape[1], 
                sigma=sigma, learning_rate=learning_rate, 
                neighborhood_function='gaussian')

    return som

def create_gp(logic=False):
    # Define custom logical operators for our Genetic Programming
    def and_op(x1, x2):
        return np.logical_and(x1, x2)

    def or_op(x1, x2):
        return np.logical_or(x1, x2)

    def gt_op(x1, x2):
        return np.greater_equal(x1, x2)

    def lt_op(x1, x2):
        return np.less_equal(x1, x2)

    def eq_op(x1, x2):
        return np.equal(x1, x2)

    # Use make_function to create function nodes in our GP
    and_ = make_function(function=and_op, name='and_', arity=2)
    or_ = make_function(function=or_op, name='or_', arity=2)
    gt_ = make_function(function=gt_op, name='gt', arity=2)
    lt_ = make_function(function=lt_op, name='lt', arity=2)
    eq_ = make_function(function=eq_op, name='eq', arity=2) 

    if logic:
        function_set = [and_, or_, gt_, lt_, eq_]

    est_gp = SymbolicClassifier(population_size=200,
                           generations=50, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=0,
                           parsimony_coefficient=0.01)

    return est_gp

def get_neighborhood(centroid_2d_position, step, x_size, y_size):
    neighborhood = [centroid_2d_position]
    if step > 0:
        search_space = [(centroid_2d_position[0], centroid_2d_position[1]),
                        (centroid_2d_position[0], centroid_2d_position[1]-step), 
                        (centroid_2d_position[0]-step, centroid_2d_position[1]-step),
                        (centroid_2d_position[0]-step, centroid_2d_position[1]),
                        (centroid_2d_position[0]-step, centroid_2d_position[1]+step), 
                        (centroid_2d_position[0], centroid_2d_position[1]+step),
                        (centroid_2d_position[0]+step, centroid_2d_position[1]+step),
                        (centroid_2d_position[0]+step, centroid_2d_position[1]),
                        (centroid_2d_position[0]+step, centroid_2d_position[1]-step)]

        neighborhood = []
        for region in search_space:
            if region[0] in range(x_size) and region[1] in range(y_size):
                neighborhood.append(region)

    return neighborhood

def classify(som, data, class_assignments):
    """Classifies each sample in data in one of the classes definited
    using the method labels_map.
    Returns a list of the same length of data where the i-th element
    is the class assigned to data[i].
    """
    winmap = class_assignments
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result

def classify2(som, data, class_assignments):
    prediction = []
    for d in data:
        winner = som.winner(d)
        if isinstance(class_assignments[winner], list):
            prediction.append(0)
        else:
            predicted = list(class_assignments[winner].keys())
            prediction.append(int(1 in predicted))
    
    return prediction

def create_somoclu(x_size, y_size):
    n_rows, n_columns = y_size, x_size 
    som = somoclu.Somoclu(n_rows=n_rows, n_columns=n_columns, compactsupport=False, initialization='pca')
    return som

from collections import Counter

def somoclu_labels_map(data, som, labels):
    """Returns a dictionary wm where wm[(i,j)] is a dictionary
    that contains the number of samples from a given label
    that have been mapped in position i,j.
    Parameters
    ----------
    data : np.array or list
        Data matrix.
    label : np.array or list
        Labels for each sample in data.
    """
    if not len(data) == len(labels):
        raise ValueError('data and labels must have the same length.')
    winmap = dict()
    for i in range(len(data)):
        w = som.bmus[i]
        w = tuple(w)
        if not w in winmap.keys():
            winmap[w] = []
        winmap[w] = winmap[w] + [labels[i]]

    for position in winmap.keys():
        winmap[position] = Counter(winmap[position])

    return winmap

def somoclu_classify(som, data, class_assignments):
    """Classifies each sample in data in one of the classes definited
    using the method labels_map.
    Returns a list of the same length of data where the i-th element
    is the class assigned to data[i].
    """
    winmap = class_assignments
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.bmus[d]
        if tuple(win_position) in winmap.keys():
            result.append(winmap[tuple(win_position)].most_common()[0][0])
        else:
            result.append(default_class)
    return result

def output_report(results, report, estimator_name, query_id, factor_ex):
    df = pd.DataFrame(report).transpose()
    df = df.iloc[[0, 1]]
    fact = []
    est = []
    qid = []
    posneg = []
    for i in range(df.shape[0]):
        if i % 2 == 0:
            posneg.append('negative')
        else:
            posneg.append('positive')
        
        fact.append(factor_ex)
        est.append(estimator_name)
        qid.append(query_id)

    df['PosNeg'] = posneg
    df['Estimator'] = est
    df['QueryID'] = qid
    df['FactorEx'] = fact
    return results.append(df, ignore_index=True)
    
def experiment(name_data, original_data, preprocessed_data, queries, nexperiments = 30):
    data = preprocessed_data.values
    result = []
    gp = GeneticProgrammingQBE(preprocessed_data)

    print('Data: ', name_data)
    for query_id in range(len(queries)):
        concept = original_data.query(queries[query_id]).index.to_list()
        labels = [int(x in concept) for x in range(1, data.shape[0]+1)]

        print('Query id: ', query_id)
        for factor_ex in [0.2, 0.5, 0.8]:
            print('Factorex : ', factor_ex)
            for i in range(nexperiments):
                print('Experiment no.: ', i)
                X_trainID, X_testID, y_train, y_test = train_test_split([i for i in range(data.shape[0])], labels, test_size=0.1, stratify=labels)
                X_train = data[X_trainID]
                X_test = data[X_testID]
                #X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=factor_ex, stratify=labels)
                
                num_neurons = 5*(data.shape[0]**0.543)
                x_size = int(num_neurons**0.5) +1
                y_size = int(num_neurons**0.5) +1
                som = create_som(data, x_size, y_size)
                som.pca_weights_init(data)
                training_iterations = 5000
                print('somA training started')
                som.train_random(data, training_iterations, verbose=False)
                print('somA training finished')
                radius = 0
                print('somA prediction started')
                class_assignments = som.labels_map(X_train, y_train)
                predicted = classify2(som, X_test, class_assignments)
                print('somA prediction finished')
                
                scores = precision_recall_fscore_support(y_test, predicted, average='binary')
                report = list(scores[:3]) + ['SOM', query_id, factor_ex]
                result.append(report)

                #SOMOCLU
                # sclu = create_somoclu(x_size, y_size) 
                # print('somC training started')
                # sclu.train(data=data, epochs=5000)
                # print('somC training finished')

                # print('somC prediction started')
                # class_assignments = somoclu_labels_map(X_train,sclu, y_train)
                # predicted = somoclu_classify(sclu, X_testID, class_assignments)
                # print('somC prediction finished')
                
                # scores = precision_recall_fscore_support(y_test, predicted, average='binary')
                # report = list(scores[:3]) + ['SOMOCLU', query_id, factor_ex]
                # result.append(report)

                # Decision Tree
                dt = DecisionTreeClassifier()
                print('dt training started')
                dt.fit(X_train, y_train)
                print('dt training finished')
                predicted = dt.predict(X_test)

                scores = precision_recall_fscore_support(y_test, predicted, average='binary')
                report = list(scores[:3]) + ['DT', query_id, factor_ex]
                result.append(report)

                # # OneClassSVM
                # oneclass = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.1)
                # print('svm training started')
                # oneclass.fit(X_train)
                # print('svm training finished')
                # predicted = [int(y < 1) for y in oneclass.predict(X_test)]

                # scores = precision_recall_fscore_support(y_test, predicted, average='binary')[:3]
                # report = list(scores[:3]) + ['OCSVM', query_id, factor_ex]
                # result.append(report)

                # GP
                gp.simple_search(population_size=300, crossover_rate=0.8, 
                            mutation_rate=0.2, num_generations=100,  
                            X_train=X_train, y_train=y_train, verbose=False)
                predicted = gp.predict(X_test)
                scores = precision_recall_fscore_support(y_test, predicted, average='binary')[:3]
                report = list(scores[:3]) + ['GP', query_id, factor_ex]
                result.append(report)

                # results = output_report(results, 
                #                         classification_report(y_test, predicted, output_dict=True), 
                #                         'GP', query_id, factor_ex)
       
    df = pd.DataFrame(data=result, columns=['precision', 'recall', 'f1score', 'estimator', 'queryid', 'factorex'])
    df.to_excel(f'ExpUnsupClassification/data_{name_data}.xls')

if __name__ == "__main__":    
    # cartable = pd.read_pickle(os.path.join('../datasets', 'car_original_dataset.pkl'))
    # cartable.columns = [str.lower(col.replace('.', '_')) for col in cartable.columns]
    # cartable['origin'] = cartable['origin'].map({0: False, 1: True})
    # cartable['automatic_gearbox'] = cartable['automatic_gearbox'].map({0: False, 1: True})

    # preprocessed_data = pd.read_pickle('..//datasets//1993CarsPrep.pkl')

    # queries = ["type == 'Sporty' and origin == 0", 
    # "type != 'Sporty' and origin == 1",
    # "automatic_gearbox == 1 and horsepower >= 150",
    # "luggage_capacity >= 18 and passenger_capacity > 5",
    # "price <= 7000 and mpg >= 26 and automatic_gearbox == 0",
    # "manufacturer == 'Ford' or manufacturer == 'Chevrolet'"]

    #experiment('1993Cars', cartable, preprocessed_data, queries, nexperiments=15)
    
    automobile = pd.read_pickle(os.path.join('../datasets', 'automobileOriginalNoNA.pkl'))
    preprocessed_data = pd.read_pickle('..//datasets//automobilePrep.pkl')

    automobile = automobile.reset_index()
    preprocessed_data = preprocessed_data.reset_index()

    import automobilequeries as aq
    queries = [aq.Q1, aq.Q2, aq.Q3, aq.Q4, aq.Q5]
    experiment('Automobile', automobile, preprocessed_data, queries, nexperiments=15)

    # import adultqueries
    # adult = pd.read_pickle(os.path.join('../datasets', 'adultNoNA.pkl'))
    # preproc = pd.read_pickle(os.path.join('../datasets', 'prepAdultNoNA.pkl'))
    # experiment('Adult', adult, preproc, adultqueries.queries, nexperiments=10)