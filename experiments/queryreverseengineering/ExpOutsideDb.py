#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import random
import math

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

cartable = pd.read_pickle(os.path.join('../datasets', 'car_original_dataset.pkl'))
cartable.columns = [str.lower(col.replace('.', '_')) for col in cartable.columns]
cartable['origin'] = cartable['origin'].map({0: False, 1: True})
cartable['automatic_gearbox'] = cartable['automatic_gearbox'].map({0: False, 1: True})

preprocessed_data = pd.read_pickle('..//datasets//1993CarsPrep.pkl')

query_workload = ["type == 'Sporty' and origin == 0", "type != 'Sporty' and origin == 1", "make == 'Ford' or make == 'Chevrolet'"]

x_size = 10
y_size = 10
learning_rate = 0.5
sigma = math.ceil(x_size*0.5)
training_iterations = 5000
excluded_size = 1
query_id = 1
radius = 1
maxnb = 5

results = []
for i in range(30):
    concept = cartable.query(query_workload[query_id]).index.to_list()
    excluded = list(random.choices(concept, k=excluded_size))

    used_ids = [i for i in cartable.index.to_list() if not i in excluded]

    # Exclude id from concept
    concept = [i for i in concept if not i in excluded]

    #example = preprocessed_data.loc[excluded + random.choices(concept, k=3)].values
    example = preprocessed_data.loc[excluded].values

    data = preprocessed_data.loc[used_ids].values

    def get_label(idx, concept):
        if idx in concept:
            return 'Positive'
        else:
            return 'Unlabeled'

    labels = [get_label(x, concept) for x in used_ids]

    from minisom import MiniSom
    som = MiniSom(10, y_size, len(data[0]), sigma=sigma, learning_rate=learning_rate, 
                neighborhood_function='gaussian')

    som.pca_weights_init(data)
    som.train_random(data, training_iterations, verbose=False)

    labels_map = som.labels_map(data, labels)

    def visualize_som(labels):
        labels_map = som.labels_map(data, labels)
        label_names = np.unique(labels)

        plt.figure(figsize=(x_size, y_size))
        ax = plt.gca()
        ax.set_facecolor('xkcd:white')
        the_grid = GridSpec(x_size, y_size)
        for position in labels_map.keys():
            label_fracs = [labels_map[position][l] for l in label_names]
            plt.subplot(the_grid[x_size-1-position[1], position[0]], aspect=1)
            patches, texts = plt.pie(label_fracs)
        plt.legend(patches, label_names, bbox_to_anchor=(0, 1.5), ncol=3)
        plt.show()

    #visualize_som(expected)
    #visualize_som(predicted)

    def get_neighborhood(centroid_2d_position, step=radius):
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
        
        return neighborhood[:min(len(neighborhood), maxnb)-1]

    def calc_results(center):
        neighborhood = get_neighborhood(center, radius)
            
        print('Radius: ', radius)
        print('Concept: ', concept)
        print('Excluded: ', excluded)
        print('Excluded BMU: ', center)
        print('Neighborhood: ', neighborhood)
        print('Concept BMUs: ', [som.winner(preprocessed_data.loc[c].values) for c in concept])

        hits = sum([1 for c in concept if som.winner(preprocessed_data.loc[c].values) in neighborhood])

        from collections import Counter
        id_label_map = som.labels_map(data, [i for i in range(len(data))])

        neighbors = []
        for n in neighborhood:
            if isinstance(id_label_map[n], Counter):
                neighbors.extend(list(id_label_map[n].keys()))

        neighbors = list(set(neighbors))

        accuracy = hits/len(neighbors) if len(neighbors) > 0 else 0
        print('Neighbors:', neighbors)
        print('Hit Neighbors: ', hits)
        print('Accuracy neighbors: ', accuracy)

        return accuracy

    randbmu = random.randint(0, x_size-1), random.randint(0, y_size-1)
    sombmu = som.winner(example)
    
    from sklearn.metrics.pairwise import cosine_similarity 
    similarities = cosine_similarity(data, example.reshape(1,-1))
    df = pd.DataFrame(data=data, columns=preprocessed_data.columns)
    df['label'] = labels
    df['similarity'] = similarities
    df.sort_values(by=['similarity'], ascending=False)
    
    simacc = sum([1 for i in range(maxnb) if df.loc[i]['label'] == 'Positive'])/maxnb
    results.append((calc_results(sombmu), calc_results(randbmu), simacc))

pd.DataFrame(data=results, columns=['SOM', 'RANDOM', 'SIM']).mean().to_excel(f'expoutside_qid_{query_id}_somsize_{x_size}_{y_size}_radius_{radius}_excluded_{excluded_size}.xls')

#pd.DataFrame(data=results, columns=['SOM', 'RANDOM']).mean().to_excel(f'expoutside_qid_{query_id}_somsize_{x_size}_{y_size}_radius_{radius}_excluded_{excluded_size}.xls')

