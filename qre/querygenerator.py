import operator
import pandas as pd
from querysketch import QuerySketch


def calculate_precision_recall(relevant, retrieved):
    if len(retrieved) == 0:
        return 0.0, 0.0
    
    matched = 0
    for i in range(len(relevant)):
        for j in range(len(retrieved)):
            if (relevant[i] == retrieved[j]).all():
                matched += 1
                break
    
    precision = matched/len(retrieved)
    recall = matched/len(relevant)
    return precision, recall

class QueryGenerator(object):
    def __init__(self, example, tables, join_finder, selection_learner):
        self.example = example
        self.tables = tables
        self.join_finder = join_finder
        self.selection_learner = selection_learner

    def enumerate_sketches(self, query_tables, projection, joins, selection):
        sketches = []
        sketches.append(QuerySketch(query_tables, projection, [], [], selection))

        for join_conditions, join_tables in joins:
            sk = QuerySketch(query_tables, projection, join_tables, join_conditions, selection)
            sketches.append(sk)
        return sketches
        
    def get_ranked_sketches(self, sketches, database_connector):
        ranking = {}
        sketches.sort(key=lambda x: len(x.join_tables), reverse=False)
        for sk in sketches:
            estimated_cardinality = 1
            for jointable in sk.join_tables:
                cardinality = database_connector.execute_command(str.format('SELECT count(*) FROM {0}', jointable))
                estimated_cardinality *= int(cardinality[0])
            ranking[sk] = estimated_cardinality
        
        # sort sketches by estimated cardinality
        return [x[0] for x in sorted(ranking.items(), key=operator.itemgetter(1))]

    def generate_query(self, database_connector, verbose=False):
        produced_query = None
        query_tables = []
        columns_to_project = []
        selection_criteria = []

        # find projection columns, query tables, and exact selection criteria
        for col in self.example.columns:
            columns_to_project.append(col)
            query_tables.extend([tab for tab, data in self.tables.items() if col in data['name'].values])
            values = self.example[col].unique()
            if len(values) == 1:
                selection_criteria.append(str.format('{0} = {1}', col, values[0]))

        query_tables = list(set(query_tables))

        sketches = []

        # find candidate joins        
        for table in query_tables:
            candidate_joins_and_tables = self.join_finder.enumerate_candidate_joins(table)
            candidate_joins = [join for (join, tab) in candidate_joins_and_tables]
            join_tables = [tab for (join, tab) in candidate_joins_and_tables]

            # create query sketches based on candidate joins
            sketches.extend(self.enumerate_sketches(query_tables, columns_to_project, candidate_joins_and_tables, selection_criteria))

        ranked_sketches = self.get_ranked_sketches(sketches, database_connector)
        
        for query_sketch in ranked_sketches:
            query_sketch.result = database_connector.execute_query_to_pandas(query_sketch.get_query(without_projection=True))
            # check whether the sketch contains the examples
            if query_sketch.result.empty or not query_sketch.contains_examples(self.example):
                continue

            query_sketch.result = database_connector.execute_query_to_pandas(query_sketch.get_query(self.tables, without_projection=True))
            
            self.selection_learner.create_training_data(self.example, query_sketch.result, query_sketch.positive_indicies)
            selection_predicate = self.selection_learner.search()
            selection_predicate = selection_predicate.replace(query_sketch.table_column_separator, '.')
            produced_query = query_sketch.get_query(self.tables, additional_selection = selection_predicate)
            obtained_result = database_connector.execute_query_to_pandas(produced_query)
            precision, recall = calculate_precision_recall(self.example.values, obtained_result.values)

            if verbose:
                print('===========================================')
                print('Query tables: ', query_tables)
                print('Join tables: ', join_tables)
                print('Join conditions: ', candidate_joins)
                print('------------------------------------------')
                print('Query Sketch: ', query_sketch.get_query(self.tables))
                print('-------------------------------------------')
                print('Generated query: ', produced_query)
                print('-------------------------------------------')
                print(obtained_result.info(verbose=True))
                print('-------------------------------------------')
                print('Precision: ', precision)
                print('Recall: ', recall)
                print('===========================================')

            stop_qre = False
            valid_response = False
            while not valid_response:
                next_sketch = input('Execute next sketch (Y/N)? ')
                if next_sketch in ['y', 'Y']:
                    valid_response = True
                elif next_sketch in ['n', 'N']:
                    valid_response = True
                    stop_qre = True

            if stop_qre:
                break

        return produced_query