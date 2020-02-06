import random, operator
import numpy as np
import qbe.refactored.utilities as utilities
from deap import base, creator, tools, gp, algorithms
from sklearn.utils import check_random_state
from qbe.refactored.learningalgorithmqbe import BaseLearningAlgorithm
from qbe.refactored.fitnessfunction import QueryDiscoveryFitnessFunction

class GeneticProgrammingQBE(BaseLearningAlgorithm):
    def __init__(self, dataset, population_size=100, crossover_rate=0.8, mutation_rate=0.2, num_generations=100, max_gen_without_gain=10):
        super().__init__(dataset)
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations 
        self.max_gen_without_gain = max_gen_without_gain

    def find_selection_predicate(self, positive_indices, negative_indices, verbose):
        dataframe = self.dataset.preprocessed_data.copy(deep=True)
        dataframe.columns = [col.replace('##', '_1_') for col in dataframe.columns]

        self.toolbox = self.configure(dataframe, positive_indices)
        population = self.toolbox.population(n=self.population_size)
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        past_fitness = []
        hall_of_fame = {'indvidual': None, 'fitness': None}

        if verbose:
            print('##############################')
            print("Generation \t Min")

        for generation in range(1, self.num_generations):
            # Select next generation
            offspring = self.toolbox.select(population, len(population))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.mutation_rate:
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
            if generation > self.max_gen_without_gain and past_fitness[-self.max_gen_without_gain:] == [min_fitness for x in range(self.max_gen_without_gain)]:
                break

        if verbose: 
            print('####### Evolution ended #######')

        # Get best solution
        self.best_solution = hall_of_fame['individual']
        
        predicate = utilities.genetic_2_predicate(hall_of_fame['individual'])
        predicate = predicate.replace('_1_', '##')
        print('Min fitness found: ', min_fitness)

        return predicate

    def configure(self, dataframe, positive_indices):
        self.fitness_function = QueryDiscoveryFitnessFunction(dataframe, positive_indices)

        pset = gp.PrimitiveSetTyped("MAIN", [str for x in dataframe.columns], bool)
        pset.addPrimitive(operator.and_, [bool, bool], bool)
        pset.addPrimitive(operator.or_, [bool, bool], bool)
        pset.addPrimitive(operator.eq, [str, float], bool)
        pset.addPrimitive(operator.gt, [str, float], bool)
        pset.addPrimitive(operator.ge, [str, float], bool)
        pset.addPrimitive(operator.lt, [str, float], bool)
        pset.addPrimitive(operator.le, [str, float], bool)
        pset.addPrimitive(operator.ne, [str, float], bool)

        def notapplied(val):
            return val

        pset.addPrimitive(notapplied, [str], str)
        pset.addPrimitive(notapplied, [float], float)
        pset.addTerminal(True, bool)

        # Get all possible values in the dataframe and use them as terminals
        terminals = set()
        for col in dataframe.columns:
            for val in set(dataframe[col].values):
                terminals.add(val)

        for t in terminals:
            pset.addTerminal(t, float)

        col_names = {}
        for i in range(len(dataframe.columns)):
            arg = 'ARG' + str(i)
            col_names[arg] = dataframe.columns[i]
        
        pset.renameArguments(**col_names)

        # Define minimization problem
        creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
        creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMin)

        min_depth = 2
        max_depth = 10

        toolbox = base.Toolbox()

        # Initialization strategy
        toolbox.register('expr', gp.genHalfAndHalf, pset=pset, min_=min_depth, max_=max_depth)

        toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        # Register fitness function
        toolbox.register('evaluate', self.evaluate_fitness)
        # Selection strategy        
        toolbox.register("select", tools.selDoubleTournament, parsimony_size=1.4, fitness_size=6, fitness_first=True)
        # Crossover strategy
        toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
        # Mutation strategy
        toolbox.register("expr_mut", gp.genFull, min_=min_depth, max_=max_depth)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        # Restrict max individual size
        toolbox.decorate('mate', gp.staticLimit(operator.attrgetter('height'), max_value=max_depth))

        return toolbox

    def evaluate_fitness(self, individual):
        # Transform the tree expression into a callable function
        func = self.toolbox.compile(expr=individual)
        return self.fitness_function.calculate(func),