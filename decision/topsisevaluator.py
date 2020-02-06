# Source: https://scikit-criteria.readthedocs.io/en/latest/index.html
from skcriteria import Data, MIN, MAX
from skcriteria.madm import closeness, simple
from decision.baseevaluator import BaseEvaluator

class TopsisEvaluator(BaseEvaluator):
    def __init__(self, candidate_names=None):
        self.candidate_names = candidate_names
        self.data_topsis = None

    def get_optimal_candidates(self, candidates, criteria):
        names = self.candidate_names.iloc[candidates.index].values
         
        min_max = []
        weights = []
        columns = []
        
        for c in criteria:
            if c.maximize:
                min_max.append(MAX)
            else:
                min_max.append(MIN)
            
            weights.append(c.weight)
            columns.append(c.index)

        self.data_topsis = Data(candidates[columns].as_matrix().tolist(), min_max, weights=weights, anames=names, cnames=columns)
        model = closeness.TOPSIS()
        choice = model.decide(self.data_topsis)
        return  [candidates.index[int(choice.best_alternative_)]] # Return must be a list

    def print(self):
        if self.data_topsis != None:
            self.data_topsis.plot()