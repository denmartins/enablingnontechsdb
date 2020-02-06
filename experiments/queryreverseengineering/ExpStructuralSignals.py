#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import random
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

cartable = pd.read_pickle(os.path.join('../datasets', 'car_original_dataset.pkl'))
cartable.columns = [str.lower(col.replace('.', '_')) for col in cartable.columns]
cartable['origin'] = cartable['origin'].map({0: False, 1: True})
cartable['automatic_gearbox'] = cartable['automatic_gearbox'].map({0: False, 1: True})

preprocessed_data = pd.read_pickle('..//datasets//1993CarsPrep.pkl')
data = preprocessed_data.values

query_workload = ["type == 'Sporty' and origin == 0", "type != 'Sporty' and origin == 1", "make == 'Ford' or make == 'Chevrolet'"]

nexperiments = 15 
results = pd.DataFrame()

x_size = 15
y_size = 15
learning_rate = 0.5
sigma = math.ceil(x_size*0.5)
training_iterations = 5000
factor_ex = 0.5
query_id = 0

for i in range(nexperiments):
    concept = cartable.query(query_workload[query_id]).index.to_list()

    example_size = math.ceil(len(concept) * factor_ex)

    example_ids = random.choices(concept, k=example_size)

    prep_examples = preprocessed_data.loc[example_ids]

    def get_label(idx, concept, example_ids):
        if idx in example_ids:
            return 'Positive'
        else:
            return 'Unlabeled'

    labels = [get_label(x, concept, example_ids) for x in range(1, data.shape[0]+1)]

    from minisom import MiniSom
    som = MiniSom(x_size, y_size, len(data[0]), sigma=sigma, learning_rate=learning_rate, 
                neighborhood_function='gaussian')

    som.pca_weights_init(data)
    som.train_random(data, training_iterations, verbose=False)

    # labels_map = som.labels_map(data, labels)
    # label_names = np.unique(labels)

    # plt.figure(figsize=(x_size, y_size))
    # ax = plt.gca()
    # ax.set_facecolor('xkcd:white')
    # the_grid = GridSpec(x_size, y_size)
    # for position in labels_map.keys():
    #     label_fracs = [labels_map[position][l] for l in label_names]
    #     plt.subplot(the_grid[x_size-1-position[1], position[0]], aspect=1)
    #     patches, texts = plt.pie(label_fracs)
    # plt.legend(patches, label_names, bbox_to_anchor=(0, 1.5), ncol=3)
    # #plt.savefig('som_car_pies.png')
    # plt.show()

    labels_map = som.labels_map(data, labels)
    label_names = labels

    def get_classification(item, labels_map, som):
        winner = som.winner(item)
        predicted_labels = list(labels_map[winner].keys())
        
        if 'Positive' in predicted_labels:
            return 'Positive'
        else:
            return 'Negative'

    #print('Quantization error: ', som.quantization_error(data))

    def get_expected_label(idx, concept):
        if idx in concept:
            return 'Positive'
        else:
            return 'Negative'
        
    expected = [get_expected_label(x, concept) for x in cartable.index.to_list()]
    predicted = [get_classification(x, labels_map, som) for x in data.tolist()]

    report = classification_report(expected, predicted, output_dict=True)

    if results.empty:
        results = pd.DataFrame(report).transpose()
    else:
        results = results.add(pd.DataFrame(report).transpose())

    #print(results)

filename = str.format('queryid{0}_size_{1}_{2}_it_{3}_lr{4}_sig{5}_factex{6}.xls', 
                        query_id+1, x_size, y_size, training_iterations, learning_rate, sigma, factor_ex)

results.div(nexperiments).to_excel(filename)