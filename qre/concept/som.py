# Self-organizing maps implementation based on http://www.ai-junkie.com/ann/som/som1.html
import random
import numpy as np
from math import exp,log
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_neighborhood_radius(current_iteration, initial_neighborhood_radius, time_constant_neighborhood):
    return initial_neighborhood_radius * np.exp(- current_iteration / time_constant_neighborhood)

def get_learning_rate(current_iteration, initial_learning_rate, num_iterations):
    return initial_learning_rate * np.exp(- current_iteration / num_iterations)

def get_gaussian_neighborhood(distance, squared_neighborhood_radius):
    return np.exp(-distance / (2 * squared_neighborhood_radius))

def update_weight(input_vector, weight_vector, influence_area, learning_rate):
    new_weight_vector = weight_vector + influence_area * learning_rate * (input_vector - weight_vector)
    return new_weight_vector

def get_winner_node_index(input_vector, weights, num_weights):
    distances = np.zeros(num_weights)
    for k in range(num_weights):
        distances[k] = np.linalg.norm(input_vector - weights[k]) # calculate euclidean distance
    return np.argmin(distances)

def epoch(dataset, weights, current_iteration, time_constant_neighborhood, initial_neighborhood_radius, initial_learning_rate, num_iterations):
    num_observations = len(dataset)
    num_weights = len(weights)
    learning_rate = initial_learning_rate
    
    for i in range(num_observations):
        input_vector = dataset[np.random.randint(num_observations)]
        index_winner_node = get_winner_node_index(input_vector, weights, num_weights)
        winner_node = weights[index_winner_node]
        radius_neighborhood = get_neighborhood_radius(current_iteration, initial_neighborhood_radius, time_constant_neighborhood)
        
        for j in range(num_weights):
            distance = np.linalg.norm(winner_node - weights[j])
            squared_radius = radius_neighborhood * radius_neighborhood
            if(distance < squared_radius):
                influence_area = get_gaussian_neighborhood(distance, squared_radius)
                weights[j] = update_weight(input_vector, weights[j], influence_area, learning_rate)
        
        learning_rate = get_learning_rate(current_iteration, initial_learning_rate, num_iterations)
    
    return weights

def main():
    # SOM constants
    map_size = [1, 2]
    num_iterations = 1000

    # Radius constants
    initial_neighborhood_radius = max(map_size)/2
    time_constant_neighborhood = num_iterations/np.log(initial_neighborhood_radius)

    # Learning rate constants
    initial_learning_rate = 0.1

    input_size = 4
    dataset = np.matrix('-1 -1 1 0; 1 -1 1 1; -1 -1 -1 0; 1 1 -1 1; -1 1 1 0') # dataset from Meuser's book
    weights = np.random.rand(map_size[0] * map_size[1], input_size)

    for i in range(10):
        np.random.shuffle(dataset)
        weights = epoch(dataset, weights, i+1, time_constant_neighborhood, initial_neighborhood_radius, initial_learning_rate, num_iterations)

    for i in range(len(dataset)):
        data = dataset[i]
        print(get_winner_node_index(data, weights, len(weights)))

if __name__ == "__main__":
    main()