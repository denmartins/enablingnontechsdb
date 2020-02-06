import pandas as pd
import random, operator
import numpy as np

# Based on https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35
class AbstractGeneticAlgorithm(object):
    def __init__(self, fitness_function, population_size, min_individual_size, 
                        max_individual_size, elite_size, 
                        crossover_rate, mutation_rate):
        self.fitness_function = fitness_function    
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.min_individual_size = min_individual_size
        self.max_individual_size = max_individual_size
        self.progress = []
        
        if elite_size == None:
            self.elite_size = int(0.3*self.population_size)
        else:
            self.elite_size = elite_size

        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def search(self, iterations, verbose):
        population = self.create_population(self.population_size, self.min_individual_size, self.max_individual_size)

        hall_of_fame = (None, self.fitness_function.default_value)
        for i in range(iterations):        
            ranked_population = self.rank_population(population)
            best_individual = ranked_population[0][0]
            best_fitness = ranked_population[0][1]
            self.progress.append(self.fitness_function.cut_value - best_fitness)
            
            if verbose: 
                print(str.format('Iteration: {0:000}, Best Fitness: {1:0.3f}', i+1, best_fitness))
            
            if hall_of_fame[0] == None or best_fitness > hall_of_fame[1]:
                hall_of_fame = (best_individual, best_fitness)

            if best_fitness >= self.fitness_function.cut_value:
                if verbose:
                    print('------------------------')
                    print('Solution FOUND')
                    print('------------------------')
                break
            
            selected_individuals = self.select_individuals(ranked_population, self.elite_size)
            offspring = self.breed_population(selected_individuals, self.elite_size)
            population = self.mutate_population(offspring, self.mutation_rate)

        return hall_of_fame[0]

    def rank_population(self, population):
        fitness_population = []
        for p in population:
            fitness_population.append((p, self.calculate_fitness(p)))
        return sorted(fitness_population, key=operator.itemgetter(1), reverse=True)        

    def calculate_fitness(self, individual):
        return self.fitness_function.evaluate(individual)

    def select_individuals(self, sorted_population, eliteSize):
        selected_individuals = []
        df = pd.DataFrame(np.array(sorted_population), columns=["Individual","Fitness"])
        df['cum_sum'] = df.Fitness.cumsum()
        df['cum_perc'] = df.cum_sum/df.Fitness.sum()
        
        for i in range(0, eliteSize):
            selected_individuals.append(sorted_population[i][0])

        for i in range(0, len(sorted_population) - eliteSize):
            pick = random.random()
            for i in range(0, len(sorted_population)):
                if pick <= df.iat[i,3]:
                    selected_individuals.append(sorted_population[i][0])
                    break

        return selected_individuals

    def breed_population(self, selected_individuals, elite_size):
        offspring = []
        length = len(selected_individuals) - elite_size
        pool = random.sample(selected_individuals, len(selected_individuals))

        for i in range(0, elite_size):
            offspring.append(selected_individuals[i])

        for i in range(0, length):
            child = self.breed(pool[i], pool[len(selected_individuals)-i-1])
            offspring.append(child)

        return offspring

    def select_parents(self, fitness_dictionary, parents_count):
        selected_individuals = []
        individual_prob = [(1/fitness_dictionary[i])/sum(fitness_dictionary.values()) for i in range(len(fitness_dictionary))]
        
        while len(selected_individuals) < parents_count:
            for i in range(len(individual_prob)):
                if random.random() <= individual_prob[i]:
                    if not fitness_dictionary.keys()[i] in selected_individuals:
                        selected_individuals.append(fitness_dictionary.keys()[i])    

        return selected_individuals
    
    def breed(self, a, b):
        child = []
        child_p1 = []
        child_p2 = []

        geneA = int(random.random() * len(a))
        geneB = int(random.random() * len(a))

        start_gene = min(geneA, geneB)
        end_gene = max(geneA, geneB)

        for i in range(start_gene, end_gene):
            child_p1.append(a[i])
        
        child_p2 = [item for item in b if item not in child_p1]
        child = child_p1 + child_p2
        
        return child

    def mutate_population(self, population, mutation_rate):
        mutated = []
        for p in population:
            mutated.append(self.mutate(p, mutation_rate))
        return mutated

    def mutate(self, individual, mutation_rate):
        mutated = [x for x in individual]
        if random.random() < mutation_rate:
            mutation_point = random.randint(0, len(mutated)-1)
            mutated[mutation_point] = self.create_random_feature(mutated)
        return mutated

    def create_random_individual(self, individual_size):
        individual = []
        while len(individual) < individual_size:
            feature = self.create_random_feature(individual)
            individual.append(feature)
        return individual

    def create_random_feature(self, individual):
        raise NotImplementedError('create_random_feature has to be implemented')

    def create_population(self, population_size, min_individual_size, max_individual_size):
        population = []
        for i in range(population_size):
            individual_size = random.randint(min_individual_size, max_individual_size)
            population.append(self.create_random_individual(individual_size))
        return population