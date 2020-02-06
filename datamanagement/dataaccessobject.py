import sklearn.datasets as ds
import pandas as pd
import numpy as np
from sklearn import preprocessing
import os 

class Dataset(object):
    def __init__(self, original_data, preprocessed_data):
        self.original_data = original_data
        self.preprocessed_data = preprocessed_data
        self.data_matrix = preprocessed_data.as_matrix()

    def get_original_data_from_prepr_data(self, selected_tuples):
        indexes = []
        for row in selected_tuples.as_matrix():
            for i in range(self.data_matrix.shape[0] - 1):
                if (row == self.data_matrix[i]).all():
                    indexes.append(i)

        return self.original_data.loc[indexes]

class DataAccessObject(object):
    """Class implementing the access to datasets"""
    def __init__(self):
        self.min_max_scaler = preprocessing.MinMaxScaler((0, 1))

    def get_normalized_data(self, dataset):
        normalized_dataset = self.min_max_scaler.fit_transform(dataset)
    
        return normalized_dataset
    
    def get_iris_dataset(self):
        iris = ds.load_iris()
        iris.feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

        normalized = self.get_normalized_data(iris.data)
        features_and_target = pd.DataFrame(data=np.c_[dataset.data, dataset.target], columns=iris['feature_names'] + ['target'])
        dataset = Dataset(iris.data, features_and_target)

        return dataset
    
    def get_car_dataset(self):
        # Load car data
        original_cars = pd.read_pickle(os.path.join('datasets', 'car_original_dataset.pkl'))
        preprocessed_cars = pd.read_pickle(os.path.join('datasets', 'preprocessed_car_dataset.pkl'))

        ### Selected features for using learning algorithms
        # selected_features = [col for col in preprocessed_cars.columns if not col in 
        #         ['price', 'num_of_cylinders', 'horsepower', 'fuel_tank_capacity',
        #         'Wheelbase', 'Rear.seat.room', 'Weight', 'length', 'width',
        #         'passenger_capacity', 'Front', 'RPM', 'luggage_capacity']]

        dataset = Dataset(original_cars, preprocessed_cars)
        return dataset