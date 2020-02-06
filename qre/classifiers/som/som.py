import pandas as pd
import numpy as np
import os
import random
import operator

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

from scipy.spatial import distance as spd

from minisom import MiniSom
import somoclu

class AbstractSomClassifier(object):
    def __init__(self, x_size=None, y_size=None, learning_rate=0.1, initial_radius=0.5):
        self.x_size = x_size
        self.y_size = y_size
        self.learning_rate = learning_rate
        self.initial_radius = initial_radius
        self.som = None

    def fit(self, X_train, y_train, training_iterations=1000):
        if self.x_size == None and self.y_size == None:
            num_neurons = 5*(X_train.shape[0]**0.543)
            self.x_size = int(num_neurons**0.5) +1
            self.y_size = int(num_neurons**0.5) +1
        
        self.train(X_train, y_train, training_iterations)

    def train(self, X_train, y_train, training_iterations, batch_training=False, verbose=False):
        raise NotImplementedError('Method train was not implemented')

    def predict(self, X_test):
        assert self.som != None, 'The SOM was not initialized'
        return self.make_prediction(X_test)
        
    def make_prediction(self, X_test):
        raise NotImplementedError()

class MiniSomClassifier(AbstractSomClassifier):
    def train(self, X_train, y_train, training_iterations=None, batch_training=False, verbose=False):
        self.som = MiniSom(self.x_size, self.y_size, X_train.shape[1], 
                            sigma=self.initial_radius, learning_rate=self.learning_rate, 
                            neighborhood_function='gaussian')

        self.som.random_weights_init(X_train)#pca_weights_init(X_train)
        if training_iterations == None:
            training_iterations = 10*len(X_train)

        if batch_training:
            self.som.train_batch(X_train, training_iterations, verbose=verbose)
        else:
            self.som.train_random(X_train, training_iterations, verbose=verbose)

        self.class_assignments = self.som.labels_map(X_train, y_train)

    def get_neighborhood(self, centroid_2d_position, step):
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
                if region[0] in range(self.x_size) and region[1] in range(self.y_size):
                    neighborhood.append(region)

        return neighborhood

    def classify(self, X_test, step, majority_vote):
        pred = []
        for x in X_test:
            winner = self.som.winner(x)
            neighbors = self.get_neighborhood(winner, step)
            lab = {0: 0, 1:0}
            for n in neighbors:
                if n in self.class_assignments:
                    if self.class_assignments[n].most_common()[0][0] > 0:
                        lab[1] += 1
                    else:
                        lab[0] += 1

            if majority_vote:
                print('Majority vote')
                pred.append(max(lab, key=lab.get))

            else:
                print('Single positive vote')
                c = 1 if lab[1] > 0 else 0
                pred.append(c)

        return pred

    def make_prediction(self, X_test):
        return self.classify(X_test, step=0, majority_vote=False)


    def get_neighbors(self, X_test, data, num_selected_items):
        """Select indexes based on their proximity in the map"""
        assert self.som != None    
        
        item_relevance_mapping = {}
        winner = self.som.winner(X_test)
        for index in data.index:
            elem = data.loc[index].values
            w = self.som.winner(elem)
            distance = np.linalg.norm(np.array(winner) - np.array(w))
            #abs(spd.cityblock(list(winner), list(w)))
            item_relevance_mapping[index] = distance
        
        sorted_candidates = sorted(item_relevance_mapping.items(), key=operator.itemgetter(1))
        neighbors = [x[0] for x in sorted_candidates[:num_selected_items]]
        return neighbors

class SomocluSomClassifier(AbstractSomClassifier):
     def train(self, X_train, y_train, training_iterations=1000, batch_training=False, verbose=False):
         self.som = somoclu.Somoclu(n_rows=self.y_size, n_columns=self.x_size, compactsupport=False, initialization='pca')
         self.som.train(data=X_train, epochs=training_iterations, radius0=self.initial_radius, scale0=self.learning_rate)

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

def outside_db():
    cartable = pd.read_pickle('car_original_dataset.pkl')
    cartable.columns = [str.lower(col.replace('.', '_')) for col in cartable.columns]
    cartable['origin'] = cartable['origin'].map({0: False, 1: True})
    cartable['automatic_gearbox'] = cartable['automatic_gearbox'].map({0: False, 1: True})

    preprocessed_data = pd.read_pickle('1993CarsPrep.pkl')

    queries = ["type == 'Sporty' and origin == 0", 
    "price <= 7000 and mpg >= 26 and automatic_gearbox == 0",
    "manufacturer == 'Ford' or manufacturer == 'Chevrolet'",
    "manufacturer != 'Volkswagen' and manufacturer != 'Dodge' and automatic_gearbox == 1",
    "make == 'Ford Mustang' or make=='Ford Probe'"]
    iterations = 30
    for q in queries:
        tp = []
        fp = []
        fscore = []
        dt_fscore = []
        for it in range(iterations):
            concept = cartable.query(q).index.to_list()
            labels = [int(x in concept) for x in range(1, cartable.shape[0]+1)]

            example = [random.choice(concept)]
            #print(example)
            #X_train, X_test, y_train, y_test = train_test_split(preprocessed_data.values, labels, test_size=0.2, stratify=labels)
            
            used_ids = [i for i in cartable.index if not i in example]

            ### Single example outside the database
            X_train = preprocessed_data.loc[used_ids].values 
            y_train = np.array([int(i in concept) for i in used_ids])
            X_test = preprocessed_data.loc[example].values.reshape(1, -1)
            y_test = [1]#np.array([1]).reshape(-1, 1)
            
            som = MiniSomClassifier()
            som.fit(X_train, y_train)
            #nearst_region = som.get_neighborhood(som.som.winner(X_test), step=1)
            #neighbors = [i for i in used_ids if som.som.winner(preprocessed_data.loc[i].values) in nearst_region]

            neighbors = som.get_neighbors(X_test, preprocessed_data.loc[used_ids], int(len(concept)/1))
            
            is_in_concept = [int(j in concept) for j in neighbors]
            true_positives = sum(is_in_concept)
            false_positives = len(is_in_concept) - true_positives
            #print('Retrieved true positives: ', true_positives)
            #print('Retrieved false positives: ', false_positives)

            predicted = [int(x in neighbors and x in concept) for x in used_ids]
            
            fscore.append(precision_recall_fscore_support(y_train, predicted, average='binary')[2])

            tp.append(true_positives)
            fp.append(false_positives)

            dt = DecisionTreeClassifier()
            dt.fit(X_test, y_test)
            pred = dt.predict(X_train)
            dt_fscore.append(precision_recall_fscore_support(y_train, pred, average='binary')[2])
            
        
        print('Query: ', q)
        print('SOM TP: ', sum(tp)/iterations)
        print('SOM FP: ', sum(fp)/iterations)
        print('SOM Fscore: ', sum(fscore)/iterations)
        print('DT Fscore: ', sum(dt_fscore)/iterations)

if __name__ == "__main__":    
    outside_db()