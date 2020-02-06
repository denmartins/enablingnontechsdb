import candidateselector as cs
from sklearn.datasets import load_iris
import numpy as np
import os
import pandas as pd

class InterAct(object):
    def __init__(self, selector, dataset):
        self.interaction_history = dict()
        self.selector = selector
        self.dataset = dataset
        
    def select_candidate(self):
        remaining_candidates = [x for x in self.dataset if not tuple(x) in self.interaction_history.keys()]
        return self.selector.select(remaining_candidates, self.interaction_history)
    
    def get_feedback(self, candidate):
        self.interaction_history[candidate] = input('Is this candidate relevant? (y/n)\t') in ['y', 'Y']


if __name__ == "__main__":
    preprocessed_cardataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'datasets', 'dummy_cartable.pkl')
    preprocessed_cartable = pd.read_pickle(preprocessed_cardataset_path)

    cardatapath = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'datasets', 'car_original_dataset.pkl')
    cardataset = pd.read_pickle(cardatapath)
    del cardataset['imagepath']

    input_table = cardataset.copy(deep=True)
    output_table = input_table.query("make == 'Chevrolet Camaro' or make == 'Ford Mustang'")[['make','manufacturer', 'type', 'price']]

    X_train = preprocessed_cartable.loc[output_table.index.values]

    iris = load_iris()
    selector = cs.DistanceBasedCandidateSelector()
    machine = InterAct(selector, iris)
    candidate = machine.select_candidate()
    print(candidate)