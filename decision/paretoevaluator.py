from decision.baseevaluator import BaseEvaluator

class ParetoEvaluator(BaseEvaluator):
    def dominates(self, s1, s2, criteria):
        dom = True
        for cr in criteria:
            if cr.maximize:
                if s1[cr.index] <= s2[cr.index]:
                    dom = False
            elif s1[cr.index] > s2[cr.index]:
                dom = False
        return dom

    def get_optimal_candidates(self, candidates, criteria):
        dominated = []
        for index1, cand1 in candidates.iterrows():
            for index2, cand2 in candidates.iterrows():
                if not (cand1 == cand2).all():
                    if self.dominates(cand1, cand2, criteria):
                        dominated.append(index2)
        optimal_indexes = [k for k, v in candidates.iterrows() if k not in dominated]
        return optimal_indexes