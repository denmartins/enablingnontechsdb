from sklearn import preprocessing
import numpy as np

class ZeroOneNormalizer():
    def __init__(self):
        self.min_max_scaler = preprocessing.MinMaxScaler()

    def normalize(self, values):
        scaled = self.min_max_scaler.fit_transform(values)
        return scaled

    def get_normalized(self, value):
        array = [float(value)]
        return self.min_max_scaler.transform(np.array(array).reshape(-1, 1))

class DataNormalizer():
    def __init__(self, lower_bound = 0.0, upper_bound=1.0):
        self.min_value = 0
        self.max_value = 0
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    
    def normalize(self, data, min = None, max = None):
        if min == None:
            self.min_value = data.min()
        if max == None:
            self.max_value = data.max()

        scaled = self.lower_bound + ( (data - self.min_value) * (self.upper_bound - self.lower_bound) / (self.max_value - self.min_value) )
        return scaled

    def get_normalized(self, data):
        return self.normalize(data, self.min_value, self.max_value)