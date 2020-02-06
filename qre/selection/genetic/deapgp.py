import utilities as util
import random, operator
import numpy as np
import pandas as pd
from deap import base, creator, tools, gp, algorithms
from sklearn.utils import check_random_state

class NumericDataColumn(object):
    pass

class BooleanDataColumn(object):
    pass

class TextDataColumn(object):
    pass

class TextAttribute(object):
    pass

class NumericAttribute(object):
    pass

class BooleanAttribute(object):
    pass

class DataColumn(object):
    def __init__(self, column_name):
        self.name = str(column_name)
        self.attribute_type_name = self.name + '_attribute'
        self.column_type = type(self.name, (), {})
        self.atribute_type = type(self.attribute_type_name, (), {})
        self.column_values = []
        self.generic_type = None

class DEAPGeneticProgramming(object):
    def __init__(self, dataframe, output_table, fitness_function):
        self.dataframe = dataframe
        self.output_table = output_table
        self.fitness_function = fitness_function
        self.toolbox = self.configure_genetic_programming(dataframe)

    def evaluate_fitness(self, individual):
        return self.fitness_function(individual),

    def create_columns(self, dataframe):
        columns = []
        
        for i in range(dataframe.shape[1]):
            col = dataframe.columns[i]
            datacolumn = DataColumn(col)

            if set(dataframe[col].values) == set([0, 1]):
                datacolumn.generic_type = bool        
            else:
                if dataframe.dtypes[i] == np.dtype('O'):
                    datacolumn.generic_type = str
                else:
                    datacolumn.generic_type = float

            values = dataframe[col].unique()
            for v in values:
                datacolumn.column_values.append((v, datacolumn.atribute_type))

            columns.append(datacolumn)
        
        for dtcol in columns:
            globals()[dtcol.name] = dtcol.column_type
            globals()[dtcol.attribute_type_name] = dtcol.atribute_type

        return columns

    def configure_genetic_programming(self, dataframe):    
        columns = self.create_columns(dataframe)
        
        def notapplied(val):
            return val

        pset = gp.PrimitiveSetTyped("MAIN", [dtcol.column_type for dtcol in columns], bool)

        pset.addPrimitive(operator.and_, [bool, bool], bool)
        pset.addPrimitive(operator.or_, [bool, bool], bool)
        pset.addTerminal(True, bool)

        visited_pairs = []
        for i in range(len(columns)):
            for j in range(len(columns)):
                if i != j and not (i, j) in visited_pairs and columns[i].generic_type != str:
                    pset.addPrimitive(operator.eq, [columns[i].column_type, columns[j].column_type], bool)
                    pset.addPrimitive(operator.ne, [columns[i].column_type, columns[j].column_type], bool)

                    if columns[j].generic_type == float and (columns[i].generic_type == columns[j].generic_type):
                        pset.addPrimitive(operator.gt, [columns[i].column_type, columns[j].column_type], bool)
                        pset.addPrimitive(operator.lt, [columns[i].column_type, columns[j].column_type], bool)

                visited_pairs.append((i,j))
                visited_pairs.append((j,i))

            dtcol = columns[i]
            pset.addPrimitive(operator.eq, [dtcol.column_type, dtcol.atribute_type], bool)
            pset.addPrimitive(operator.ne, [dtcol.column_type, dtcol.atribute_type], bool)
            
            if dtcol.generic_type == float:
                pset.addPrimitive(operator.gt, [dtcol.column_type, dtcol.atribute_type], bool)
                pset.addPrimitive(operator.lt, [dtcol.column_type, dtcol.atribute_type], bool)

            pset.addPrimitive(notapplied, [dtcol.atribute_type], dtcol.atribute_type)
            pset.addPrimitive(notapplied, [dtcol.column_type], dtcol.column_type)

            for v in dtcol.column_values:
                pset.addTerminal(v[0], v[1])

        col_names = {}
        for i in range(len(dataframe.columns)):
            arg = 'ARG' + str(i)
            col_names[arg] = dataframe.columns[i]
        
        pset.renameArguments(**col_names)

        # Define maximization problem
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        # Initialization strategy
        toolbox.register('expr', gp.genHalfAndHalf, pset=pset, min_=2, max_=5)

        toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        # Register fitness function
        toolbox.register('evaluate', self.evaluate_fitness)
        # Selection strategy        
        toolbox.register("select", tools.selTournament, tournsize=5)
        # Crossover strategy
        #toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
        toolbox.register("mate", gp.cxOnePoint)
        # Mutation strategy
        toolbox.register("expr_mut", gp.genFull, min_=1, max_=3)
        #toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        toolbox.register("mutate", gp.mutInsert, pset=pset)

        # Restrict max individual size
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        return toolbox

    def simple_search(self, population_size, crossover_rate, mutation_rate, num_generations, max_gen_without_gain, verbose=True):    
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(lambda ind: len(ind))
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        
        pop = self.toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)
        pop, log = algorithms.eaSimple(pop, self.toolbox, crossover_rate, mutation_rate, num_generations, 
                                       stats=mstats, halloffame=hof, verbose=verbose)
        self.best_solution = hof[0]
        return util.genetic_2_predicate(hof[0])

    def search_best_predicate(self, population_size, crossover_rate, mutation_rate, num_generations, max_gen_without_gain, verbose=True):
        population = self.toolbox.population(n=population_size)
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        past_fitness = []
        hall_of_fame = {'indvidual': None, 'fitness': None}

        if verbose:
            print('##############################')
            print("Generation \t Min")

        for generation in range(1, num_generations):
            # Select next generation
            offspring = self.toolbox.select(population, len(population))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace population
            population[:] = offspring

            fits = [ind.fitness.values[0] for ind in population]
            min_fitness = min(fits)

            if hall_of_fame['fitness'] == None or min_fitness < hall_of_fame['fitness']:
                hall_of_fame['individual'] = population[fits.index(min_fitness)]
                hall_of_fame['fitness'] = min_fitness

            past_fitness.append(min_fitness)

            if verbose:
                print('{0}          \t {1:02.4f}'.format(generation, min_fitness))

            # Stop evolution if fitness reach global optima
            if min_fitness <= 0.0:
                print('Fitness reached zero')
                break
            # Stop evolution if it does not improve
            if generation > max_gen_without_gain and past_fitness[-max_gen_without_gain:] == [min_fitness for x in range(max_gen_without_gain)]:
                break

        if verbose: 
            print('####### Evolution ended #######')

        # Get best solution
        self.best_solution = hall_of_fame['individual']
        return util.genetic_2_predicate(hall_of_fame['individual'])