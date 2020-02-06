import random
import operator
import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from sklearn.metrics import precision_recall_fscore_support

class GeneticProgrammingQBE(object):
    def __init__(self, dataframe):
        self.best = None
        self.toolbox = self.configure_genetic_programming(dataframe)
    
    def evalfitness(self, individual):
        predicted = self.get_prediction(individual, self.X_train)
        _, _, fscore, _ = precision_recall_fscore_support(self.y_train, predicted, average='binary')
        return fscore,

    def get_prediction(self, individual, X):
        # Transform the tree expression in a callable function
        func = self.toolbox.compile(expr=individual)
        # Predict 
        predicted = [bool(func(*x)) for x in X]
        
        return predicted

    def predict(self, X_test):
        return self.get_prediction(self.best, X_test)

    def configure_genetic_programming(self, dataframe):
        pset = gp.PrimitiveSetTyped("MAIN", [float for x in dataframe.columns], bool)
        pset.addPrimitive(operator.and_, [bool, bool], bool)
        pset.addPrimitive(operator.le, [float, float], bool)
        pset.addPrimitive(operator.ge, [float, float], bool)
        pset.addPrimitive(operator.eq, [float, float], bool)
        
        def notapplied(val):
            return val

        #pset.addPrimitive(notapplied, [str], str)
        pset.addPrimitive(notapplied, [float], float)
        pset.addTerminal(True, bool)
        pset.addTerminal(False, bool)
        pset.addTerminal(1, float)
        pset.addTerminal(0, float)
        pset.addTerminal(0.5, float)
        
        for i in range(5):
            pset.addEphemeralConstant("rand"+str(i), lambda: random.random(), float)
    
        col_names = {}
        for i in range(len(dataframe.columns)):
            arg = 'ARG' + str(i)
            col_names[arg] = dataframe.columns[i]
        
        pset.renameArguments(**col_names)

        creator.create('FitnessMin', base.Fitness, weights=(1.0,))
        creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMin)

        min_depth = 1
        max_depth = 5
        total_max_depth = 10

        toolbox = base.Toolbox()

        # Initialization strategy
        toolbox.register('expr', gp.genHalfAndHalf, pset=pset, min_=min_depth, max_=max_depth)

        toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        # Register fitness function
        toolbox.register('evaluate', self.evalfitness)
        # Selection strategy        
        toolbox.register("select", tools.selTournament, tournsize=3)
        # Crossover strategy
        toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
        # Mutation strategy
        toolbox.register("expr_mut", gp.genFull, min_=min_depth, max_=max_depth)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        # Restrict max individual size
        toolbox.decorate('mate', gp.staticLimit(operator.attrgetter('height'), max_value=total_max_depth))

        return toolbox

    def simple_search(self, population_size, crossover_rate, mutation_rate, num_generations, X_train, y_train, verbose=True):
        self.X_train = X_train
        self.y_train = y_train
        stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(key=len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        
        mstats.register("min", np.min)
        mstats.register("max", np.max)        
        mstats.register("avg", np.mean)
        
        pop = self.toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)
        pop, log = algorithms.eaSimple(pop, self.toolbox, crossover_rate, mutation_rate, num_generations, 
                                        stats=mstats, halloffame=hof, verbose=verbose)

        self.best = hof[0]

        return hof[0]