import pandas as pd
import numpy as np
import re

class BaseQueryLearner(object):
    def __init__(self, dataset, tablename, pysql):
        self.dataset = dataset
        self.tablename = tablename
        self.pysql = pysql
        self.learned_query = None
        self.positive_indices = set()
        self.negative_indices = set()
        self.projection_statement = None
        self.selection_predicate = None
        
    def learn_query(self, example, learning_algorithm, verbose=False):
        self.verbose = verbose
        self.projection_statement = self.find_projection_statement(example)
        self.selection_predicate = self.find_selection_predicates(example, learning_algorithm)
        self.learned_query = self.projection_statement + ' WHERE ' + self.selection_predicate
        return self.learned_query
    
    def find_projection_statement(self, example):
        self.create_global_inverted_index()
        columns_to_project, tables_to_project = self.find_projection_columns_and_tables(example)
        
        return 'SELECT ' + ', '.join(columns_to_project) + ' FROM ' + ', '.join(tables_to_project)
        
    def create_global_inverted_index(self):
        inverted_index = []
        for col in self.dataset.original_data.columns:
            for val in set(self.dataset.original_data[col].values):
                inverted_index.append([val, type(val), col, self.tablename])
        self.global_inverted_index = pd.DataFrame(data=inverted_index, columns=['Value', 'Type', 'Name', 'Table'])
        
    def find_projection_columns_and_tables(self, example):
        columns_to_project = []
        tables_to_project = set()
        for attribute_example in [example[col][0] for col in example.columns]:
            selected_rows = self.global_inverted_index.loc[self.global_inverted_index['Type'] == type(attribute_example)]
            for index, row in selected_rows.iterrows():
                if type(attribute_example) is str and str.lower(attribute_example) in str.lower(row['Value']):
                    # Use string lengths to create a probability table and verify if attribute_example is likely to be part of the data column
                    strlength = [len(x) for x in self.global_inverted_index['Value'].loc[self.global_inverted_index['Name'] == row['Name']].values]
                    probability = dict()
                    for i in strlength:
                        if not i in probability.keys():
                            probability[i] = 1/len(strlength)
                        else:
                            probability[i] = probability[i] + 1/len(strlength)
                    if len(attribute_example) in probability.keys() and probability[len(attribute_example)] >= min(list(probability.values())):
                        tables_to_project.add(row['Table'])
                        columns_to_project.append(str.format('{0}.{1}', row['Table'], row['Name']))
                    else:
                        continue
                elif attribute_example == row['Value']:
                    tables_to_project.add(row['Table'])
                    columns_to_project.append(str.format('{0}.{1}', row['Table'], row['Name']))
                        
        # Remove duplicates
        resulting_columns = []
        for col in columns_to_project:
            if not col in resulting_columns:
                resulting_columns.append(col)

        return resulting_columns, tables_to_project
        
    def find_selection_predicates(self, example, learning_algorithm):
        self.positive_indices, self.negative_indices = self.get_positive_and_negative_indices(example)
        preprocessed_selection_predicate = learning_algorithm.find_selection_predicate(self.positive_indices, self.negative_indices, self.verbose)
        return self.preprocessed_predicate_to_original(preprocessed_selection_predicate)

    def get_positive_and_negative_indices(self, example):
        resulting_tuples = self.pysql(self.projection_statement)
        positive_indices = set()
        negative_indices = set()

        for ex in example.values:
            for i in range(resulting_tuples.shape[0]):
                matched = []
                for j in range(example.shape[1]):
                    if type(ex[j]) is str and str.lower(ex[j]) in str.lower(resulting_tuples.values[i][j]):
                        matched.append(True) 
                    elif ex[j] == resulting_tuples.values[i][j]:
                        matched.append(True) 
                    else:
                        matched.append(False)
                if False in matched:
                    negative_indices.add(i)
                else:
                    positive_indices.add(i)
        
        negative_indices = negative_indices - positive_indices

        return positive_indices, negative_indices
    
    # Only works when enconded variables are binary
    def preprocessed_predicate_to_original(self, predicate, numeric_processing_function=lambda x,y: x.max() * y):
        original_predicate = predicate.replace('(','').replace(')', '')
        extracted_preds = []
        for and_pred in original_predicate.split('AND'):
            if 'OR' in and_pred:
                for or_pred in and_pred.split('OR'):
                    extracted_preds.append(or_pred.strip())
            else:
                extracted_preds.append(and_pred.strip())

        for pred in extracted_preds:
            if '##' in pred:
                for col in self.dataset.original_data.columns:
                    if col in pred:
                        encoded = re.split(r"##", pred)
                        equal = False
                        for op in ['>', '>=']:
                            if op in encoded[2]:
                                equal = True
                                break
                        if equal:
                            new_pred = str.format('{0} == \'{1}\'', encoded[0], encoded[1])
                        else:
                            new_pred = str.format('{0} <> \'{1}\'', encoded[0], encoded[1])
                        original_predicate = original_predicate.replace(pred, new_pred)
            else:
                split_pred = re.split(string=pred, pattern='(<[=>]?|==|>=?|\&\&|\|\|)') # Matching comparison operators
                column = split_pred[0].replace(' ','')
                if column in self.dataset.original_data.columns:
                    if np.issubdtype(self.dataset.original_data[column].dtype, np.number):
                        denormalized_value = numeric_processing_function(self.dataset.original_data[column].max(), float(split_pred[-1]))
                        new_pred = pred.replace(str(split_pred[-1]), str(denormalized_value))
                        original_predicate = original_predicate.replace(pred, new_pred)

        return original_predicate    

class SemanticQueryLearner(BaseQueryLearner):
    def __init__(self, dataset, tablename, pysql, semantic_selector):
        super().__init__(dataset, tablename, pysql)
        self.semantic_selector = semantic_selector

    def get_positive_and_negative_indices(self, example):
        positive_indices, negative_indices = BaseQueryLearner.get_positive_and_negative_indices(self, example)
        return self.update_indices_using_som_similarity(positive_indices, negative_indices)
    
    def update_indices_using_som_similarity(self, positive_indices, negative_indices):
        selected_indices = []
        for posidx in positive_indices:
            selected_indices.extend(self.semantic_selector.select(query=self.dataset.data_matrix[posidx], dataset=self.dataset, num_selected_items=3))
        
        positive_indices = list(set(selected_indices) | positive_indices)
        negative_indices = list(negative_indices - set(positive_indices))
                
        return positive_indices, negative_indices
