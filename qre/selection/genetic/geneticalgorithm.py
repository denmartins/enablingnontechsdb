import pandas as pd
import random, operator
import numpy as np

class GeneElement(object):
    def mutate(self):
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()

class OrOpGeneElement(GeneElement):
    def mutate(self):
        return AndOpGeneElement()

    def __repr__(self):
        return ' or '

class AndOpGeneElement(GeneElement):
    def mutate(self):
        return OrOpGeneElement()

    def __repr__(self):
        return ' and '

class FeatureGeneElement(GeneElement):
    def __init__(self, criterion):
        self.criterion = criterion
    
    def mutate(self, new_criterion):
        return FeatureGeneElement(new_criterion)

    def __repr__(self):
        return self.criterion

class GeneticAlgorithm(object):
    def __init__(self, input_table, output_table, fitness_function, 
                population_size, min_individual_size, 
                max_individual_size, elite_size, 
                crossover_rate, mutation_rate):
        self.input_table = input_table
        self.output_table = output_table
        self.terminals = self.get_terminals()
        self.functions = [OrOpGeneElement(), AndOpGeneElement()]

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

    def rank_population(self, population):
        fitness_population = []
        for p in population:
            fitness_population.append((p, self.calculate_fitness(p)))
        return sorted(fitness_population, key=operator.itemgetter(1), reverse=True)        

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

    def get_terminals(self):
        comparisom_op = ['>', '<', '==']
        all_selection_criteria = []
        
        column_types = dict()

        for col in self.input_table.columns:
            for val in set(self.input_table[col].values):
                for op in comparisom_op:
                    if type(val) is str:
                        if op == '==':
                            all_selection_criteria.append(str.format('{0} {1} "{2}"', col, op, val))
                        else:
                            continue
                    else:
                        all_selection_criteria.append(str.format('{0} {1} {2}', col, op, val))

            if set(self.input_table[col].values) == set([0, 1]):
                column_types[col] = bool
            else:
                if self.input_table[col].dtype == np.dtype('O'):
                    column_types[col] = str
                else:
                    column_types[col] = float
        
        visited_pairs = []
        for i in range(self.input_table.shape[1]):
            for j in range(self.input_table.shape[1]):
                if i != j and not (i, j) in visited_pairs and column_types[self.input_table.columns[i]] == column_types[self.input_table.columns[j]]:
                    if column_types[self.input_table.columns[i]] != str:
                        for op in comparisom_op:
                            all_selection_criteria.append(str.format('{0} {1} {2}', self.input_table.columns[i], op, self.input_table.columns[j]))
                    else:
                        all_selection_criteria.append(str.format('{0} == {1}', self.input_table.columns[i], self.input_table.columns[j]))
                visited_pairs.append((i, j))
                visited_pairs.append((j, i))
        
        return all_selection_criteria
    
    def individual_to_criteria(self, individual):
        return ''.join([str(x) for x in individual])

    def calculate_fitness(self, individual):
        if len(individual) >= self.max_individual_size:
            return self.fitness_function.default_value

        criteria = self.individual_to_criteria(individual)

        try:
            tab = self.input_table.query(criteria)
        except:
            tab = pd.DataFrame()
        
        fit = self.fitness_function.evaluate(tab, self.output_table)
        return fit

    def create_random_feature(self, individual):
        feature = None
        while feature == None:
            cr = self.terminals[random.randint(0, len(self.terminals)-1)]
            if not cr in [str(e) for e in individual]:
                feature = FeatureGeneElement(cr)
        return feature

    def create_population(self):
        population = []

        for i in range(self.population_size):
            individual_size = random.randint(self.min_individual_size, self.max_individual_size)
            individual = []

            while True:
                individual.append(self.create_random_feature(individual))
                if len(individual) < individual_size:
                    bop = random.choice(self.functions)
                    individual.append(bop)
                else:
                    break

            population.append(individual)
        return population

    def search(self, iterations, verbose):
        population = self.create_population()

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
            population.extend([p[0] for p in ranked_population[:self.elite_size]])

        best_individual =  hall_of_fame[0]
        selection = self.individual_to_criteria(best_individual)
        return selection