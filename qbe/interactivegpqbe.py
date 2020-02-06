import util
from deap import base, creator, tools, gp, algorithms
import random, operator
import numpy as np
from sklearn.utils import check_random_state

class InteractiveGPQBE(object):
    def __init__(self, dataframe, fitness_function, subjective_eval_function):
        self.fitness_function = fitness_function
        self.subjective_eval = subjective_eval_function
        self.toolbox = self.configure_genetic_programming(dataframe)
        self.evolved_generations = 0
        self.hall_of_fame = None

    def evaluate_fitness(self, individual):
        # Transform the tree expression in a callable function
        func = self.toolbox.compile(expr=individual)
        objective_fitness = self.fitness_function.calculate(func)
        subjective_fitness = self.subjective_eval(individual)

        return objective_fitness * subjective_fitness,
         
    def configure_genetic_programming(self, dataframe):    
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

        min_depth = 3
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
        toolbox.register("select", tools.selTournament, tournsize=10)
        # Crossover strategy
        toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
        # Mutation strategy
        toolbox.register("expr_mut", gp.genFull, min_=min_depth, max_=max_depth)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        # Restrict max individual size
        toolbox.decorate('mate', gp.staticLimit(operator.attrgetter('height'), max_value=20))

        return toolbox

    def evolve_population(self, population=None, population_size=10, crossover_rate=0.9, mutation_rate=0.5, num_generations=1, verbose=True):
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        
        if population == None:
            population = self.toolbox.population(n=population_size)

        hof = tools.HallOfFame(1)
        evolved_population, log = algorithms.eaSimple(population, self.toolbox, 
                                        crossover_rate, mutation_rate, 
                                        num_generations, stats=stats, 
                                        halloffame=hof, verbose=False)

        self.hall_of_fame = hof
        self.evolved_generations += num_generations

        if verbose:
            min_fitness = hof[0].fitness.values[0]
            print('Minimum fitness: {0:02.4f}'.format(min_fitness))
            if min_fitness <= 0.0:
                print('Individual {0} reached fitness zero'.format(hof[0]))

        return evolved_population

    def interactive_search(self, population=None, population_size=10, crossover_rate=0.9, mutation_rate=0.5, verbose=True):
        if population == None:
            population = self.toolbox.population(n=population_size)
        
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        hall_of_fame = {'indvidual': None, 'fitness': None}

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

        if verbose:
            print('Minimum fitness: {0:02.4f}'.format(min_fitness))
            if min_fitness <= 0.0:
                print('Individual {0} reached fitness zero'.format(hall_of_fame['individual']))

        self.evolved_generations += 1

        return population