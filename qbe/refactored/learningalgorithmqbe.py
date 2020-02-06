class BaseLearningAlgorithm(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def find_selection_predicate(self, positive_indices, negative_indices, verbose):
        raise NotImplementedError('Method not implemented.')

    def configure(self, dataframe):
        raise NotImplementedError('Method not implemented.')