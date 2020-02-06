import operator

class BaseSelector(object):
    def __init__(self):
        self.item_relevance_mapping = {}
    
    def select(self, query, dataset, num_selected_items):
        raise NotImplementedError('Select method not implemented')

    def get_sorted_indexes(self, dictionary):
        """Sort dictionary by value"""
        sorted_items = sorted(dictionary, key=operator.itemgetter(1))
        sorted_indexes = [key for (key, value) in sorted_items]
        return sorted_indexes

    def sort_items_by_relevance(self, items_index):
        """Sort items by cost"""
        dictionary = [(key, value) for key, value in self.item_relevance_mapping.items() if key in items_index]
        return self.get_sorted_indexes(dictionary)