import numpy as np
from random import choice
from scipy.spatial import distance
import math

class CandidateSelector(object):
    def __init__(self):
        self.already_selected = []

    def select(self, candidates, history):
        x = None
        while x == None or x in self.already_selected:
            x = self.selection_strategy(candidates, history)
        self.already_selected.append(x)

        return x

    def selection_strategy(self, candidates, history):
        raise NotImplementedError()

class RandomCandidateSelector(CandidateSelector):
    def select(self, candidates, history):
        return choice(candidates)

class KnnCandidateSelector(CandidateSelector):
    def select(self, candidates, history):
        query = [k for k,v in history.items() if v == True][0]
        closest = None
        min_distance = float('inf')
        for c in candidates:
            dist = distance.jaccard(query, c)
            if dist < min_distance:
                min_distance = dist
                closest = c
        return closest

class DistanceBasedCandidateSelector(CandidateSelector):
    def select(self, candidates, history):
        '''Selects the candidate that is, simultaneously, (1) closest to the example(s) or to positive candidates 
        and (2) far way from the negative candidates already evaluated by the user'''
        relevance = []
        for c in candidates:
            relevance.append(self.constraint_satisfaction(c, history))
        return candidates[np.argmax(relevance)]

    def constraint_satisfaction(self, candidate, history):
        positives = [k for k,v in history.items() if v==True]
        negatives = [k for k,v in history.items() if v==False]
        together = [x for x in zip(positives, negatives)]
    
        return 1/len(together) * sum([self.indicator(candidate, tog[0], tog[1]) for tog in together])

    def indicator(self, candidate, positive, negative):
        return distance.euclidean(candidate, positive) < distance.euclidean(candidate, negative)


class ContraintSatisfactionCandidateSelector(CandidateSelector):
    def select(self, candidates, history):
        '''Selects the tuple that is, simultaneously, (1) closest to the example(s) 
        and (2) far way from the negative candidates already evaluated by the user'''
        satisfaction = []
        for tp in candidates:
            satisfaction.append(self.constraint_satisfaction(tp, candidates))
        return np.argmax(satisfaction)

    def constraint_satisfaction(self, candidate, history):
        positives = [k for k,v in history.items() if v==True]
        negatives = [k for k,v in history.items() if v==False]
        together = [x for x in zip(positives, negatives)]
    
        return 1/len(together) * sum([self.indicator(candidate, tog[0], tog[1]) for tog in together])

    def indicator(self, candidate, positive, negative):
        return distance.euclidean(candidate, positive) < distance.euclidean(candidate, negative)

class DiderotCandidateSelector(CandidateSelector):
    def __init__(self, examples, dataset):
        self.influence_weights = [1.0 for x in range(len(dataset))]

    def select(self, candidates, history):
        '''Selects the tuple that is closest to the center of mass of the positive candidates already evaluated by the user, 
        taking into consideration that candidates evaluated recently have higher weight than those evaluated in iterations near to the begining of the process. 
        '''
        raise NotImplementedError()

    def get_influence_decay(self, candidate, timestep):
        initial_influence = 1.0
        minimum_influence = 0.01
        decay = math.exp(-timestep/5)
        return max(minimum_influence, decay)

class BayesianCandidateSelector(CandidateSelector):
    def select(self, candidates, history):
        '''Selects the tuple that maximizes the probability of being evaluated as positive by the user. The model (Naive Bayes classifier) is updated after every interaction.
        '''
        raise NotImplementedError()