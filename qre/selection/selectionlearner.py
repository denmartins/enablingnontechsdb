import pandas as pd
from sklearn.utils import resample
from random import sample, shuffle

class AbstractSelectionLearner(object):
    def get_k_fold_training_data(self, k, training_data):
        pseudo_negatives = training_data[training_data[self.classlabel_column] == 0]
        positives = training_data[training_data[self.classlabel_column] == 1]
        
        fold_length = len(positives)
        folds = []
        
        for i in range(k):
            fold = []
            fold.extend(positives.values.tolist())
            fold.extend(sample(pseudo_negatives.values.tolist(), fold_length))
            shuffle(fold)
            dataframe = pd.DataFrame(fold, columns=training_data.columns)
            folds.append(dataframe)
        
        return folds

    def balance_training_set(self, training_data):
        # Separate majority and minority classes
        df_majority = training_data[training_data[self.classlabel_column]==0]
        df_minority = training_data[training_data[self.classlabel_column]==1]

        # Upsample minority class
        df_minority_upsampled = resample(df_minority, 
                                        replace=True,     # sample with replacement
                                        n_samples=df_majority.shape[0],    # to match majority class
                                        random_state=123) # reproducible results

        # Combine majority class with upsampled minority class
        df_upsampled = pd.concat([df_majority, df_minority_upsampled])
        return df_upsampled

    def create_training_data(self, example, sketch_result, positive_indicies, convert_categorical_values=False, verbose=False):
        self.classlabel_column='class_label'
        
        index_columns = []
        not_considered_columns =[]
        for col in sketch_result.columns:
            #TODO: automatically find index columns
            if 'id' in col[:2]:
                index_columns.append(col)
            elif col == 'title':
                not_considered_columns.append(col)

        # Removing duplicate columns
        training_data = sketch_result.loc[:,~sketch_result.columns.duplicated()].copy(deep=True)

        # Fill missing data with 'missing' tag
        training_data.fillna('ISNULL', inplace=True)

        # Adding class label
        training_data[self.classlabel_column] = pd.Series([int(i in positive_indicies) for i in range(training_data.shape[0])], index=training_data.index, dtype='bool')

        if verbose:
            print(str.format('#1 Training data balance: {0} pos, {1} neg', len(positive_indicies), training_data.shape[0] - len(positive_indicies)))

        # Converting categorical values into a binary representation
        if convert_categorical_values:
            categorical_columns = set(training_data.columns) - set(training_data._get_numeric_data().columns) - set(not_considered_columns)
            training_data = pd.get_dummies(training_data, columns=list(categorical_columns), dtype='bool')

        # Solving the problem of rows refering to the same entity
        #training_data = training_data.groupby(by=index_columns).sum()

        # Correcting class label after groupby
        #training_data['class_label'].transform(lambda x: 1 if x > 0 else 0)

        for col in training_data.columns:
            unique_values = training_data[col].unique()
            if len(unique_values) == 1 or col in not_considered_columns and not col in index_columns:
                del training_data[col] # Removing uninformative columns
        
        # Renaming columns to avoid syntax errors
        renamed_columns = [col.replace('.0', '') for col in training_data.columns]
        training_data.columns = renamed_columns

        self.training_data = training_data
        self.k_fold_training_data = self.get_k_fold_training_data(5, training_data)
        
        self.feature_columns=[col for col in training_data.columns if col != self.classlabel_column]
        self.num_tuples_in_example = example.shape[0]
        self.X = training_data[self.feature_columns]
        self.y = training_data[self.classlabel_column]

    def search(self):
        raise NotImplementedError()