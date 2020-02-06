import pandas as pd
import numpy as np
import os
import random
import math
import operator

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier

from minisom import MiniSom
from sklearn.utils import check_random_state

from gpdeap import GeneticProgrammingQBE

from scipy.spatial import distance as spd

from sklearn.tree import DecisionTreeClassifier

from gpdeap import GeneticProgrammingQBE

def get_pos_neg(query, data, som, num_selected_items):
    """Select indexes based on their proximity in the map"""    
    item_relevance_mapping = {}
    winner = som.winner(query)
    for index in range(len(data)):
        elem = data[index]
        w = som.winner(elem)
        distance = spd.cityblock(list(winner), list(w))
        item_relevance_mapping[index+1] = distance
    
    sorted_candidates = sorted(item_relevance_mapping.items(), key=operator.itemgetter(1))
    positives = [x[0] for x in sorted_candidates[:num_selected_items]]

    negatives = []
    for j in range(1,num_selected_items+1):
        negatives.append(sorted_candidates[-j][0])

    return positives, negatives

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

results = []

gp = GeneticProgrammingQBE(preprocessed_data)

for query_id in range(len(queries)):
    data = preprocessed_data.values
    concept = cartable.query(queries[query_id]).index.to_list()
    #X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=factor_ex, stratify=labels)

    num_neurons = 5*(data.shape[0]**0.543)
    x_size = int(num_neurons**0.5) +1
    y_size = int(num_neurons**0.5) +1

    learning_rate = 0.8
    sigma = max(x_size, y_size)*0.5

    som = MiniSom(x_size, y_size, data.shape[1], 
                sigma=sigma, learning_rate=learning_rate, 
                neighborhood_function='gaussian')

    training_iterations = 1000

    for fac in [1.0, 1.5, 2.0]:
        for it in range(10):
            e = random.choice(concept)
            used_ids = [i for i in preprocessed_data.index if i != e]

            labels = [int(x in concept) for x in range(1, preprocessed_data.shape[0]+1)]

            query = data[int(e)-1]

            data = preprocessed_data.loc[used_ids].values
            y_train = [int(x in concept) for x in used_ids]

            som.pca_weights_init(data)
            som.train_random(data, training_iterations, verbose=False)

            pos, neg = get_pos_neg(query, data, som, int(len(concept)*fac))

            #class_assignments = som.labels_map(data, y_train)
            #predicted = classify(som, X_test, class_assignments)   

            predicted = [int(i in pos) for i in preprocessed_data.index]
            y_test = labels

            scores = precision_recall_fscore_support(y_test, predicted, average='binary')
            report = list(scores[:3]) + [query_id, fac, 'SOM']

            results.append(report)

            dt = DecisionTreeClassifier()
            dt.fit(query.reshape(1,-1), np.array([1]).reshape(-1, 1))
            predicted = dt.predict(preprocessed_data.values)

            scores = precision_recall_fscore_support(y_test, predicted, average='binary')
            report = list(scores[:3]) + [query_id, fac, 'DT']
            results.append(report)
                    
            gp.simple_search(population_size=150, crossover_rate=0.8, 
                            mutation_rate=0.2, num_generations=50,  
                        X_train=query.reshape(1,-1), y_train=np.array([1]).reshape(-1, 1), verbose=False)
            predicted = gp.predict(preprocessed_data.values)
            scores = precision_recall_fscore_support(y_test, predicted, average='binary')[:3]
            report = list(scores[:3]) + [query_id, fac, 'GP']
            results.append(report)

df = pd.DataFrame(data=results, columns=['precision', 'recall', 'f1score', 'query_id', 'selfactor', 'estimator'])
df.to_excel('data_cartable_outsidedb_example_comple_all_techniques.xls')

print(df.mean().to_latex(index=False))