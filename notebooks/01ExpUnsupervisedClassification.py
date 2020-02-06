import pandas as pd
import numpy as np
import os
import random
import math

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from minisom import MiniSom


def get_label(idx, concept):
    if idx in concept:
        return 'Positive'
    else:
        return 'Unlabeled'

def get_neighborhood(centroid_2d_position, step, x_size, y_size):
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

    return neighborhood#[:min(len(neighborhood), maxnb)-1]

def get_classification(item, labels_map, som, radius, x_size, y_size, data, concept):
    winner = som.winner(item)
    neighborhood = get_neighborhood(winner, radius, x_size, y_size)

    hits = sum([1 for c in concept if som.winner(data[c-1]) in neighborhood])
    
    if hits >0:
        return 'Positive'
        
    return 'Unlabeled'

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

def output_report(results, report, estimator_name, query_id, factor_ex):
    df = pd.DataFrame(report).transpose()
    df = df.iloc[[0, 1]]
    fact = []
    est = []
    qid = []
    for i in range(df.shape[0]):
        fact.append(factor_ex)
        est.append(estimator_name)
        qid.append(query_id)

    df['Estimator'] = est
    df['QueryID'] = qid
    df['FactorEx'] = fact
    return results.append(df, ignore_index=True)

def experiment(name_data, original_data, preprocessed_data, queries, nexperiments = 30):
    num_neurons = 5*(original_data.shape[0]**0.5)
    x_size = int(round(num_neurons**0.5 ,2)) + 1
    y_size = x_size
    learning_rate = 0.8
    sigma = math.ceil(x_size*0.5)
    training_iterations = 10*original_data.shape[0] #5000
    print(training_iterations)
    radius = 0

    som = MiniSom(x_size, y_size, preprocessed_data.shape[1], 
                sigma=sigma, learning_rate=learning_rate, 
                neighborhood_function='gaussian')

    data = preprocessed_data.values
    #som.pca_weights_init(data)
    #som.train_random(data, training_iterations, verbose=False)
    results = pd.DataFrame()
    for query_id in range(len(queries)):
        concept = original_data.query(queries[query_id]).index.to_list()

        for factor_ex in [0.1, 0.3, 0.5, 0.8]:
            somresults = pd.DataFrame()
            svmresults = pd.DataFrame()

            for i in range(nexperiments):    
                example_size = max(1, math.ceil(len(concept) * factor_ex))
                example_ids = random.choices(concept, k=example_size)
                prep_examples = preprocessed_data.loc[example_ids]

                labels = [get_label(x, concept) for x in range(1, data.shape[0]+1)]
                
                X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=factor_ex, stratify=labels)
                
                som.pca_weights_init(X_train)
                som.train_random(X_train, training_iterations, verbose=False)
                class_assignments = som.labels_map(X_train, y_train)

                #expected = [get_label(x, concept) for x in original_data.index.to_list()]
                predicted = [get_classification(x, class_assignments, som, radius, x_size, y_size, data, concept) for x in X_test.tolist()]

                somreport = classification_report(y_test, predicted, output_dict=True)
                
                results = output_report(results, somreport, 'SOM', query_id, factor_ex)

                svm = SVC(kernel='rbf')
                svm.fit(X_train, y_train)
                predicted = svm.predict(X_test)
                
                svmreport = classification_report(y_test, predicted, output_dict=True)
                
                results = output_report(results, svmreport, 'SVM', query_id, factor_ex)

                dt = DecisionTreeClassifier()
                dt.fit(X_train, y_train)
                predicted = dt.predict(X_test)

                dtreport = classification_report(y_test, predicted, output_dict=True)
                results = output_report(results, dtreport, 'DT', query_id, factor_ex)

            #     if somresults.empty:
            #         somresults = pd.DataFrame(somreport).transpose()
            #     else:
            #         somresults = somresults.add(pd.DataFrame(somreport).transpose())
                   
            #     if svmresults.empty:
            #         svmresults = pd.DataFrame(svmreport).transpose()
            #     else:
            #         svmresults = svmresults.add(pd.DataFrame(svmreport).transpose())
                   
            # somresults.div(nexperiments).to_excel(f'ExpUnsupClassification/SOMexpUnsupClass_data_{name_data}_qid_{query_id}_factorex_{factor_ex}_somsize_{x_size}_{y_size}.xls')
            # svmresults.div(nexperiments).to_excel(f'ExpUnsupClassification/SVMexpUnsupClass_data_{name_data}_qid_{query_id}_factorex_{factor_ex}_somsize_{x_size}_{y_size}.xls')
    results.to_excel(f'ExpUnsupClassification/data_{name_data}_somsize_{x_size}_{y_size}_lr{learning_rate}_sigma_{sigma}.xls')

if __name__ == "__main__":    
    cartable = pd.read_pickle(os.path.join('../datasets', 'car_original_dataset.pkl'))
    cartable.columns = [str.lower(col.replace('.', '_')) for col in cartable.columns]
    cartable['origin'] = cartable['origin'].map({0: False, 1: True})
    cartable['automatic_gearbox'] = cartable['automatic_gearbox'].map({0: False, 1: True})

    preprocessed_data = pd.read_pickle('..//datasets//1993CarsPrep.pkl')

    queries = ["type == 'Sporty' and origin == 0", 
    "type != 'Sporty' and origin == 1",
    "automatic_gearbox == 1 and horsepower >= 150",
    "luggage_capacity >= 18 and passenger_capacity > 5",
    "price <= 7000 and mpg >= 26 and automatic_gearbox == 0",
    "manufacturer == 'Ford' or manufacturer == 'Chevrolet'"]

    #experiment('1993Cars', cartable, preprocessed_data, queries, nexperiments=30)
    automobile = pd.read_pickle(os.path.join('../datasets', 'automobileOriginalNoNA.pkl'))
    preprocessed_data = pd.read_pickle('..//datasets//automobilePrep.pkl')

    automobile = automobile.reset_index()
    preprocessed_data = preprocessed_data.reset_index()

    import automobilequeries as aq
    queries = [aq.Q1, aq.Q2, aq.Q3, aq.Q4, aq.Q5]

    experiment('Automobile', automobile, preprocessed_data, queries, nexperiments=30)