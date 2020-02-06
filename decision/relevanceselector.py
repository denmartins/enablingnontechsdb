from decision.clause import DiadicClause, Operation 
import numpy as np
import decision.costfunctions as cf
from decision.baseselector import BaseSelector

class RelevanceSelector(BaseSelector):
    def __init__(self):
        super().__init__()
    
    def select(self, query, dataset, num_selected_items):
        """Return items of most relevant items"""
        self.item_relevance_mapping = {}
        for index, row in dataset.preprocessed_data.iterrows():
            cost = 0.0
            for clause in query:
                if(clause.operation == Operation.EQUALS):
                    cost += cf.categorical_cost_function(clause.right_token, row[clause.left_token])
                elif(clause.operation == Operation.GREATER_THAN_EQUALS):
                    cost += cf.bottom_limit_cost_function(clause.right_token, row[clause.left_token])
                elif(clause.operation == Operation.LESS_THAN_EQUALS):
                    cost += cf.top_limit_cost_function(clause.right_token, row[clause.left_token])

            self.item_relevance_mapping[index] = float(cost)

        sorted_indexes = self.get_sorted_indexes(self.item_relevance_mapping.items())

        if(num_selected_items != None):
            sorted_indexes = sorted_indexes[:num_selected_items]
        
        return sorted_indexes