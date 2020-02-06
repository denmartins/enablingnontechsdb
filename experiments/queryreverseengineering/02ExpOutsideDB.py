import pandas as pd
import numpy as np
import os
import random
import math

from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity 

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

    return neighborhood

def get_accuracy(item, labels_map, som, radius, x_size, y_size, data, concept, maxnb):
    winner = som.winner(item)
    neighborhood = get_neighborhood(winner, radius, x_size, y_size)

    hits = sum([1 for c in concept if som.winner(data.loc[c].values) in neighborhood])
    
    id_label_map = som.labels_map(data.values, [i for i in range(data.shape[0])])

    neighbors = []
    for n in neighborhood:
        if isinstance(id_label_map[n], Counter):
            neighbors.extend(list(id_label_map[n].keys()))

    neighbors = list(set(neighbors))

    if len(neighbors) > 0:
        neighbors = neighbors[:max(1, maxnb)]

    accuracy = hits/len(neighbors) if len(neighbors) > 0 else 0
    
    return accuracy

def experiment(name_data, original_data, preprocessed_data, queries, nexperiments = 30):
    num_neurons = 5*(original_data.shape[0]**0.5)
    x_size = int(round(num_neurons**0.5 ,2)) + 1
    y_size = x_size
    learning_rate = 0.5
    sigma = math.ceil(x_size*0.8)
    training_iterations = 10*original_data.shape[0] #5000
    radius = 0
    maxnb = 5

    som = MiniSom(x_size, y_size, preprocessed_data.shape[1], 
                sigma=sigma, learning_rate=learning_rate, 
                neighborhood_function='gaussian')

    data = preprocessed_data.values
    som.pca_weights_init(data)
    som.train_random(data, training_iterations, verbose=False)
    results = []
    
    for query_id in range(len(queries)):    
        for i in range(nexperiments):
            concept = original_data.query(queries[query_id]).index.to_list()
            excluded = [random.choice(concept)]
            used_ids = [i for i in original_data.index.to_list() if not i in excluded]

            # Exclude id from concept
            concept = [i for i in concept if not i in excluded]

            example = preprocessed_data.loc[excluded].values

            data = preprocessed_data.loc[used_ids].values

            labels = [get_label(x, concept) for x in used_ids]
            
            som.pca_weights_init(data)
            som.train_random(data, training_iterations, verbose=False)
            labels_map = som.labels_map(data, labels)

            randexamples = random.choices(labels, k=maxnb)
            randacc = sum([1 for i in randexamples if i == 'Positive'])/maxnb
            
            similarities = cosine_similarity(data, example.reshape(1,-1))
            df = pd.DataFrame(data=data, columns=preprocessed_data.columns)
            df['label'] = labels
            df['similarity'] = similarities
            df.sort_values(by=['similarity'], ascending=False)
            
            simacc = sum([1 for i in range(maxnb) if df.loc[i+1]['label'] == 'Positive'])/maxnb
            
            somacc = get_accuracy(example, labels_map, som, radius, x_size, y_size, preprocessed_data.loc[used_ids], concept, maxnb)

            results.append([somacc, simacc, randacc, query_id])
            
    pd.DataFrame(data=results, columns=['SOM', 'SIM', 'RAND', 'QueryID']).to_excel(f'OutsideDbExpResults/OutsideDB_data_{name_data}_somsize_{x_size}_{y_size}_lr{learning_rate}_sigma_{sigma}.xls')

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

    experiment('1993Cars', cartable, preprocessed_data, queries, nexperiments=30)
    
    automobile = pd.read_pickle(os.path.join('../datasets', 'automobileOriginalNoNA.pkl'))
    preprocessed_data = pd.read_pickle('..//datasets//automobilePrep.pkl')

    automobile = automobile.reset_index()
    preprocessed_data = preprocessed_data.reset_index()

    import automobilequeries as aq
    queries = [aq.Q1, aq.Q2, aq.Q3, aq.Q4, aq.Q5]

    experiment('Automobile', automobile, preprocessed_data, queries, nexperiments=30)