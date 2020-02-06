from minisom import MiniSom
import operator
import numpy as np
import pylab as plt
from scipy.spatial import distance as spd
from collections import defaultdict
from decision.baseselector import BaseSelector

class SomSelector(BaseSelector):
    """Uses a Kohonen Self-Organizing Map to select positive and negative examples"""
    def __init__(self, som_size=(3,3), learning_rate=0.5, sigma=1.0, num_iterations=1000, neighborhood_function='gaussian'):
        super().__init__()
        self.som_size = som_size
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.num_iterations = num_iterations
        self.neighborhood_function = neighborhood_function
        self.som = None

    def train(self, data):
        self.som = MiniSom(x=self.som_size[0], y=self.som_size[1], 
                            input_len=len(data[0]), sigma=self.sigma, 
                            learning_rate=self.learning_rate, 
                            neighborhood_function=self.neighborhood_function)
        
        self.som.random_weights_init(data)
        self.som.train_random(data=data, num_iteration=self.num_iterations)

    def get_neighbors(self, query, data, num_selected_items):
        """Select indexes based on their proximity in the map"""
        if self.som == None: self.train(data)
            
        self.item_relevance_mapping = {}
        winner = self.som.winner(query)
        for index in range(len(data)):
            elem = data[index]
            w = self.som.winner(elem)
            distance = spd.cityblock(list(winner), list(w))
            self.item_relevance_mapping[index] = distance
        
        sorted_candidates = sorted(self.item_relevance_mapping.items(), key=operator.itemgetter(1))
        neighbors = [x[0] for x in sorted_candidates[:num_selected_items]]
        return neighbors
        
    def get_neighbors_region_method(self, query, data, radius=1):
        if self.som == None: self.train(data)

        self.item_relevance_mapping = {}
        # calculate winner map
        win_map = defaultdict(list)
        for index in range(len(data)):
            x = data[index]
            win_map[self.som.winner(x)].append(index)

        neighbors = []
        step = 0
        centroid_2d_position = self.som.winner(query)

        while step < radius:
            search_space = [(centroid_2d_position[0], centroid_2d_position[1]),
                            (centroid_2d_position[0], centroid_2d_position[1]-step), 
                            (centroid_2d_position[0]-step, centroid_2d_position[1]-step),
                            (centroid_2d_position[0]-step, centroid_2d_position[1]),
                            (centroid_2d_position[0]-step, centroid_2d_position[1]+step), 
                            (centroid_2d_position[0], centroid_2d_position[1]+step),
                            (centroid_2d_position[0]+step, centroid_2d_position[1]+step),
                            (centroid_2d_position[0]+step, centroid_2d_position[1]),
                            (centroid_2d_position[0]+step, centroid_2d_position[1]-step)]
            for region in search_space:
                if region[0] in range(self.som_size[0]) and region[1] in range(self.som_size[1]):
                    neighbors.extend(win_map[(region[0], region[1])])
                    for n in neighbors: 
                        self.item_relevance_mapping[n] = step
            step += 1
        return list(set(neighbors))

    def select(self, query, dataset, num_selected_items):
        """Select indexes based on their proximity in the map"""
        data = dataset.data_matrix.tolist()
        return self.get_neighbors(query, data, num_selected_items)

    def print(self, dataset, labels):
        plt.bone()
        plt.pcolor(self.som.distance_map().T, cmap='Greys', edgecolors='Grey', linewidths=1)
        plt.colorbar()

        winners = {}
        for i in range(len(dataset)):
            w = self.som.winner(dataset[i])
        
            if not w in winners:
                winners[w] = labels[i]
            else:
                winners[w] = winners[w] + '\n' + labels[i]
    
        
        for win, text in winners.items():
            plt.text(win[0] + 0.2, win[1] + 0.5, str(text), fontsize=8, 
                        bbox=dict(facecolor='white', alpha=0.8, pad = 0.2))

        plt.axis([0, self.som.weights.shape[0], 0, self.som.weights.shape[1]])
        plt.show() #Print U-Matrix: Light areas can be thought as clusters and dark areas as cluster separator.