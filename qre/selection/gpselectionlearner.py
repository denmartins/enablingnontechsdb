from selectionlearner import AbstractSelectionLearner
from genetic.fitnessfunction import ClassificationFitness
from genetic.deapgp import DEAPGeneticProgramming
from genetic.geneticalgorithm import GeneticAlgorithm
import genetic.utilities as util
from genetic.gp import GP

class DEAPSelectionLearner(AbstractSelectionLearner):
    def search(self):
        fitfunction = ClassificationFitness(0.0001, 1.0)
        def calculate_fitness(individual):
            fitness_value = fitfunction.default_value
            query = util.genetic_2_predicate(individual)

            if sum([1 for col in self.X.columns if col in query]) == 0:
                fitness_value = fitfunction.default_value
            else:
                try:
                    fitness_value = fitfunction.evaluate(self.X.query(query), self.y)
                except TypeError: None
            
            return fitness_value

        evolution = DEAPGeneticProgramming(self.X, self.y, calculate_fitness)
        result = evolution.simple_search(population_size=200, crossover_rate=0.9, mutation_rate=0.1, 
                                        num_generations=100, max_gen_without_gain=10, verbose=True)

        return result

class CustomGPSelectionLearner(AbstractSelectionLearner):
    def search(self):
        evolution = GP(self.X, self.y, ClassificationFitness(0.0001, 1.0), 
                population_size=256, min_individual_size=1,
                max_individual_size=5, elite_size=2, 
                crossover_rate=0.8, mutation_rate=0.2,
                new_individual_rate=0.01)
        
        classification_rule = evolution.search(iterations=100, max_iterations_without_improvement=25, verbose=True)
        return classification_rule.str_representation()

class GASelectionLearner(AbstractSelectionLearner):
    def search(self):
        evolution = GeneticAlgorithm(self.X, self.y, ClassificationFitness(0.0001, 1.0), 
                    population_size=256, min_individual_size=1, 
                    max_individual_size=5, elite_size=10, 
                    crossover_rate=0.8, mutation_rate=0.2)

        classification_rule = evolution.search(iterations=100, verbose=True)
        return classification_rule