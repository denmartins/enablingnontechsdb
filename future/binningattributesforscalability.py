import pandas as pd
import os
from greedyqre import QueryCriteriaSearch
from fitnessfunction import InOutFitnessFunction, AbstractFitnessFunction
from deapgp import DEAPGeneticProgramming
from sklearn.preprocessing import KBinsDiscretizer
import utilities as util
from geneticprogramming import GeneticProgramming
from gaqre import GAQRE

if __name__ == "__main__":
    input_table = pd.read_pickle(
        os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.realpath(__file__)
                )
            ), 'datasets', 'car_original_dataset.pkl')
        )
    
    del input_table['imagepath']

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    columns_to_bin = [c for c in input_table.select_dtypes(include=numerics).columns if not c in ['num_of_cylinders', 'automatic_gearbox', 'Origin']]

    encoders = dict()
    for col in columns_to_bin:
        enc = KBinsDiscretizer(n_bins=5, encode='ordinal')
        transformed = enc.fit_transform(input_table[[col]])
        input_table[[col]] = [float(x) for x in transformed]
        encoders[col] = enc

    # output_table = input_table.query(
    #     "type == 'Small' & price < 5000"
    #     )#[['make', 'type']]

    #preprocessing = QueryCriteriaSearch(input_table)
    #columns = preprocessing.find_projection_columns_and_tables(output_table)

    #print(columns)

    output_table = input_table.query(
        "type == 'Small' and automatic_gearbox == 0 and price < 2.0"
        )
    
    def recall_specificity(actual_table, output_table):
        if actual_table.empty:
            return 100.0

        hits = sum([int(item.tolist() in actual_table.values.tolist()) for item in output_table.values])
        recall = hits/output_table.shape[0]
        specificity = hits/actual_table.shape[0]

        return 1.0 - (recall * specificity)

    def individual_2_table(individual):
        actual_table = pd.DataFrame()
        try:
            selection_criteria = util.genetic_2_predicate(individual) 
            actual_table = input_table.query(selection_criteria)
        except:
            actual_table = pd.DataFrame()
        
        return actual_table
    
    def fitness_function(individual):
        actual_table = individual_2_table(individual)
        return recall_specificity(actual_table, output_table)

    #gp = DEAPGeneticProgramming(input_table, output_table, fitness_function)
    #ind = gp.simple_search(500, 0.9, 0.1, 30, 20, verbose=False)
    #ind = gp.search_best_predicate(300, 0.8, 0.5, 30, 15, verbose=True)

    #print(input_table.query(ind))
    #print(ind)

    def custom_fitness(actual_table, output_table):
        if actual_table.empty: return 0.0
        if actual_table.shape[0] < output_table.shape[0]: return 0.0

        hits = sum([int(item.tolist() in actual_table.values.tolist()) for item in output_table.values])
        recall = hits/output_table.shape[0]
        specificity = hits/actual_table.shape[0]

        return recall * specificity
    
    fitness = AbstractFitnessFunction(0.001, 1)
    fitness.evaluate = custom_fitness

    # evolution = GeneticProgramming(input_table, output_table, fitness, 
    #                 population_size=500, min_individual_size=2,
    #                 max_individual_size=6, elite_size=None, 
    #                 crossover_rate=0.9, mutation_rate=0.5,
    #                 new_individual_rate=0.1, pexp=0.9)


    evolution = GAQRE(input_table, output_table, fitness, 
                    population_size=300, 
                    min_individual_size=1, 
                    max_individual_size=10, 
                    elite_size=50, 
                    crossover_rate=0.9, 
                    mutation_rate=0.2)

    result = evolution.search(10, verbose=True)
    print(result)