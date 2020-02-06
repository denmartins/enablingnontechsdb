import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn import datasets
from sklearn.svm import OneClassSVM
from sklearn import preprocessing
from fitnessfunction import InOutFitnessFunction
from geneticprogramming import GeneticProgramming


preprocessed_cardataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'datasets', 'dummy_cartable.pkl')
preprocessed_cartable = pd.read_pickle(preprocessed_cardataset_path)

cardatapath = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'datasets', 'car_original_dataset.pkl')
cardataset = pd.read_pickle(cardatapath)
del cardataset['imagepath']

input_table = cardataset.copy(deep=True)
output_table = input_table.query("make == 'Chevrolet Camaro' or make == 'Ford Mustang'")[['make','manufacturer', 'type', 'price']]
#output_table = input_table.query("type == 'Sporty' & Origin == 0")[['make', 'price']]

X_train = preprocessed_cartable.loc[output_table.index.values].sample(n=100, replace=True).copy(deep=True)
X_test = preprocessed_cartable

positive_indices = []

algorithm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
algorithm.fit(X_train)
predicted_train = algorithm.predict(X_train)
#print(predicted_train)
test_predicted = algorithm.predict(X_test)
#print(test_predicted)
positive_indices = list(np.where(test_predicted == 1)[0])

positive_indices = positive_indices + output_table.index.values.tolist()
print('Positive: ', positive_indices)

enhanced_output_table = input_table.loc[list(set(positive_indices))].copy(deep=True)
print(enhanced_output_table[['make', 'type', 'Origin']])

evolution = GeneticProgramming(input_table, enhanced_output_table, InOutFitnessFunction(0.001, 2), 
                population_size=300, min_individual_size=2,
                max_individual_size=10, elite_size=None, 
                crossover_rate=0.9, mutation_rate=0.65,
                new_individual_rate=0.1, pexp=0.9)

result = evolution.search(100, verbose=True)
final_relation = cardataset.query(result.str_representation())[['make', 'type', 'Origin']]

print('Best query found: ' + result.str_representation())
print('Final relation:')
print(final_relation.info())
print(final_relation)

axes = plt.gca()
axes.set_ylim([0,3])
axes.set_xlim([0,len(evolution.progress)+1])
plt.plot(evolution.progress)
plt.ylabel('Error')
plt.xlabel('Generation')
plt.show()
