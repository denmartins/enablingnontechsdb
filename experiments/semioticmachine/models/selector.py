from models.clause import *
from sklearn import cluster
import numpy as np
import datamanagement.dataaccess as dt
import models.costfunctions as cf

class FeatureBasedSelector:
    def select(self, query, dataset, num_selected_items):
        selected_data = dataset

        for clause in query:
            selected_data = selected_data.loc[clause.operation(selected_data[clause.left_token], clause.right_token)]

        if(num_selected_items != None):
            return selected_data[:num_selected_items]

        return selected_data

    def sort_items(self, items_index):
        return items_index


class CostBasedSelector:
    def __init__(self):
        self.item_cost_mapping = {}

    # Sort dictionary by value
    def get_sorted_indexes(self, dict):
        sorted_items = sorted(dict, key=operator.itemgetter(1))
        sorted_indexes = [key for (key, value) in sorted_items]
        return sorted_indexes

    # Sort items by cost
    def sort_items(self, items_index):
        dict = [(key, value) for key, value in self.item_cost_mapping.items() if key in items_index]
        return self.get_sorted_indexes(dict)

    def select(self, query, dataset, num_selected_items):
        self.item_cost_mapping = {}
        for index, row in dataset.iterrows():
            cost = 0.0
            for clause in query:
                if(clause.operation == Operation.EQUALS):
                    cost += cf.categorical_cost_function(clause.right_token, row[clause.left_token])
                elif(clause.operation == Operation.GREATER_THAN_EQUALS):
                    cost += cf.bottom_limit_cost_function(clause.right_token, row[clause.left_token])
                elif(clause.operation == Operation.LESS_THAN_EQUALS):
                    cost += cf.top_limit_cost_function(clause.right_token, row[clause.left_token])

            self.item_cost_mapping[index] = float(cost)

        sorted_indexes = self.get_sorted_indexes(self.item_cost_mapping.items())

        if(num_selected_items != None):
            sorted_indexes = sorted_indexes[:num_selected_items]
        
        return dataset.loc[sorted_indexes]
    
class ClusterBasedSelector:
    def select(self, query, dataset, num_selected_items):
        preprocessed_data = dt.prepocess_data(dataset)

        k_means = cluster.KMeans(n_clusters=5)
        k_means.fit(preprocessed_data.values)
        predicted_cluster = k_means.predict(preprocessed_data.iloc[query])

        labels = k_means.labels_
        indexes = np.where(labels == predicted_cluster)
        selected_data = dataset.iloc[indexes]
        
        return selected_data[:num_selected_items]


class BayesianSelector:
    def __init__(self, relevance):
        self.relevance = relevance

    def get_initial_probability(relevance):
        a_priori_prob = []
        for i in range(len(relevance)):
            a_priori_prob.append(relevance[i]/sum(relevance))
        
        return a_priori_prob
    
    def get_prob_soft(distance):
        prob = []
        sum_exp = np.sum(np.exp(-distance))
        for i in range(len(distance)):
            prob.append(np.exp(-distance[i]) / sum_exp)
        
        return prob

    def update_prob(relevance, a_priori_prob):
        updated_prob = []
        distance = np.array([1, 1, 1, 1]) / (np.array([1, 1, 1, 1]) + np.array(relevance))
        for i in range(len(distance)):
            p_i = get_prob_soft(distance)[i] * a_priori_prob[i] / np.sum(np.array(get_prob_soft(distance)) * np.array(a_priori_prob))
            updated_prob.append(p_i)
    
        return updated_prob
