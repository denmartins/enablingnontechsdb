from decision.relevanceselector import RelevanceSelector
from decision.criterion import Criterion
from decision.paretoevaluator import ParetoEvaluator
from decision.discriminator import NullObjectDiscriminator
from sada.basesada import BaseSADA

class DecisionSADA(BaseSADA):
    def __init__(self, dataset, selector=RelevanceSelector(), evaluator=ParetoEvaluator(), discriminator=NullObjectDiscriminator(), adjuster=None):
        super().__init__(dataset, selector, evaluator, discriminator, adjuster)

    def select(self, query, num_of_selected_candidates = 5):
        selected_indexes = self.selector.select(query, self.dataset, num_of_selected_candidates)
        return selected_indexes

    # Apply Multicriteria Decision Analysis
    def assess(self, candidates, criteria):
        optimal_indexes = self.evaluator.get_optimal_candidates(candidates, criteria)
        return optimal_indexes

    def discriminate(self, candidates_to_discriminate):
        self.discriminator.discriminate(candidates_to_discriminate, self.dataset)
        return None

    def adjust(self, query):
        refined_query = query
        return refined_query

    def get_recommendations(self, query, criteria):
        selected_indexes = self.select(query)
        optimal_indexes = self.assess(self.dataset.preprocessed_data.iloc[selected_indexes], criteria)
        optimal_sorted_indexes = self.selector.sort_items_by_relevance(optimal_indexes)

        selected_candidates = self.dataset.original_data.iloc[selected_indexes]
        optimal_candidates = self.dataset.original_data.iloc[optimal_sorted_indexes]

        #self.discriminate(optimal_candidates)

        return selected_candidates, optimal_candidates

    def print_recommendations(self, query, criteria):
        selected_candidates, optimal_candidates = self.get_recommendations(query, criteria)

        print('-------------------------------------------------------------------------------')
        print('Selected candidates')
        print(self.dataset.original_data.iloc[selected_candidates.index])
        
        print('-------------------------------------------------------------------------------')
        print('Optimal candidates')
        print(self.dataset.original_data.iloc[optimal_candidates.index])