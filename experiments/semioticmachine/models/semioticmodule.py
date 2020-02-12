from models.selector import *
from models.mcda import *
from models.discriminator import *
from models.suggestion import *
import datamanagement.dataaccess as dt
import copy

class SemioticModule:
    def __init__(self, candidates, original_dataset, preprocessed_dataset):
        self.total_candidates = candidates
        self.original_dataset = original_dataset
        self.discriminator = KmeansDiscriminator(preprocessed_dataset)
        self.selector = CostBasedSelector()
        self.suggestion_strategy = SomSuggestionStrategy(candidates, preprocessed_dataset)
    
    def select(self, query, candidates, num_of_displayed_options = 5):
        return self.selector.select(query, candidates, num_of_displayed_options)

    # Apply Multicriteria Decision Analysis
    def assess(self, candidates, criteria):        
        pareto_front = ParetoFront()
        optimal_indexes = pareto_front.get_optimals(candidates, criteria)
        return self.total_candidates.loc[optimal_indexes]

    def discriminate(self, candidates_to_discriminate):
        self.discriminator.discriminate(candidates_to_discriminate, self.total_candidates)
        return None

    def suggest_via_example(self, example):
        return self.suggestion_strategy.get_suggestions_via_example(example)
    
    def suggest(self, optimal_candidates, num_of_similar_candidates = 2):
        return self.suggestion_strategy.get_suggestions(num_of_similar_candidates, optimal_candidates)

    def adjust(self, query):
        refined_query = query
        return refined_query
    
    def get_preprocess_query(self, query):
        preprocessed_query = copy.deepcopy(query)
        for clause in preprocessed_query:
            if clause.left_token in dt.DAO.normalizers:
                clause.right_token = dt.DAO.normalizers[clause.left_token].get_normalized(clause.right_token)

        return preprocessed_query

    def get_recommendations(self, query, criteria, show_discrimination_graph=False):
        preprocessed_query = self.get_preprocess_query(query)

        selected = self.select(preprocessed_query, self.total_candidates)
        optimal = self.assess(selected, criteria)
        similar = self.suggest(optimal)
        
        optimal_sorted_indexes = self.selector.sort_items(optimal.index)
        similar_sorted_indexes = self.selector.sort_items(similar)

        selected_candidates = self.original_dataset.loc[selected.index]
        optimal_candidates = self.original_dataset.loc[optimal_sorted_indexes]
        similar_candidates = self.original_dataset.loc[similar_sorted_indexes]

        if show_discrimination_graph:
            self.discriminate(optimal_candidates)

        return selected_candidates, optimal_candidates, similar_candidates