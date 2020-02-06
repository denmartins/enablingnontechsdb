import pandas as pd
from sklearn.utils import resample

class QuerySketch(object):
    __is_valid = None

    def __init__(self, query_tables, projection_columns, join_tables, join_conditions, selection_criteria):
        self.query_tables = query_tables
        self.projection_columns = projection_columns
        self.join_tables = join_tables
        self.join_conditions = join_conditions
        self.selection_criteria = selection_criteria
        self.result = None
        self.positive_indicies = None
        self.table_column_separator = 'q2w3'
    
    def get_query(self, table_dict=None, without_projection=False, without_selection=False, additional_selection=''):
        query = ''

        if len(self.query_tables) > 0:
            query = 'SELECT {projection} FROM {tables}'
            query = query.replace('{tables}', ', '.join(self.query_tables + self.join_tables))

            if without_projection == False and len(self.projection_columns) > 0:
                query = query.replace('{projection}', ', '.join(self.projection_columns))
            elif not table_dict is None:
                columns_to_project = []
                for tb in self.query_tables + self.join_tables:
                    for col in table_dict[tb]['name'].unique():
                        projcol = str.format('{0}.{1} AS {0}{2}{1}', tb, col, self.table_column_separator)
                        columns_to_project.append(projcol)

                query = query.replace('{projection}', ', '.join(columns_to_project))
            else:
                query = query.replace('{projection}', '*')
                
            predicates = []
            if len(self.join_conditions) > 0:
                predicates.append(self.join_conditions)

            if without_selection == False and len(self.selection_criteria) > 0:
                predicates.extend(self.selection_criteria)
 
            where_statement = []
            if len(predicates) > 0:
                where_statement.append(' AND '.join(predicates))

            if len(additional_selection) > 0:
                where_statement.append(additional_selection)
            
            if len(where_statement) > 0:
                query = query + ' WHERE ' + ' AND '.join(where_statement)

        return query
    
    def contains_examples(self, example):
        if self.result is None or self.result.empty:
            return False
        
        if self.__is_valid == None:
            matched_tuples_id = []
            matched_count = set()
            for i in range(example.shape[0]):
                filters = [] 
                for col in example.columns:
                    val = example[col].iloc[i]
                    if type(val) == str:
                        filters.append(str.format('{0} == "{1}"', col, val))
                    else:
                        filters.append(str.format('{0} == {1}', col, val))
                matched = self.result.query(' and '.join(filters)).index
                matched_count.update(matched)
                matched_tuples_id.extend(matched)
            
            self.positive_indicies = matched_tuples_id

            self.__is_valid = len(matched_count) >= example.shape[0]
        
        return self.__is_valid