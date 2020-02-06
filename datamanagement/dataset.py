class Dataset(object):
    def __init__(self, original_data, preprocessed_data):
        self.original_data = original_data
        self.preprocessed_data = preprocessed_data
        self.data_matrix = preprocessed_data.as_matrix()

    def get_original_indices(self, tuples):
        indices = []
        for row in tuples.values:
            for i in range(self.original_data.shape[0]):
                if (row == self.original_data.values[i]).all():
                    indices.append(i)

        return indices