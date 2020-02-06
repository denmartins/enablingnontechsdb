from random import sample, random, randint, choice, shuffle
from copy import deepcopy
import operator
import numpy as np
from math import log
from sklearn.utils import resample

# A wrapper for functions that will be used on function nodes
# function = function itself
# name = function name
# childcount = number of parameter
class fwrapper:
    def __init__(self, function, childcount, name):
        self.function = function
        self.childcount = childcount
        self.name = name

# Function nodes (nodes with children)
# When evaluate is called, it evaluates the child nodes and then applies the function to their results
class node:
    def __init__(self, fw, children):
        self.function = fw.function
        self.name = fw.name
        self.children = children

    def evaluate(self, inp):
        results = [n.evaluate(inp) for n in self.children]
        return self.function(results)

    def display(self, indent=0):
        text = ' '*indent + self.name
        for c in self.children:
            text = text + '\n' + c.display(indent+1)
        return text

    def str_representation(self):
        params = []
        for c in self.children:
            params.append(c.str_representation())
        text = '(' + params[0] + ' ' + self.name + ' ' + params[1] + ')'
        return text

    def get_depth(self):
        depths = []
        for c in self.children:
            depths.append(c.get_depth())
        return max(depths) + 1

# Nodes that only return one of the parameters passed to the program
# Its evaluate method return the parameter specified by idx
class paramnode:
    def __init__(self, idx):
        self.idx = idx
    
    def evaluate(self, inp):
        return inp[self.idx]

    def display(self, indent=0):
        return ' '*indent + str(self.idx)

    def str_representation(self):
        return str(self.idx)

    def get_depth(self, depth=0):
        return 0

# Nodes that return a constant value
class constnode:
    def __init__(self, v):
        self.v = v
    
    def evaluate(self, inp):
        return self.v

    def display(self, indent=0):
        return ' '*indent + str(self.v)

    def str_representation(self):
        if isinstance(self.v, str):
            return str.format('"{0}"', self.v)
        else:
            return str(self.v)
    
    def get_depth(self, depth=0):
        return 0

class GP(object):
    def __init__(self, input_table, output_table, 
                    fitness_function, 
                    population_size, 
                    min_individual_size, 
                    max_individual_size,
                    elite_size, 
                    crossover_rate, 
                    mutation_rate, 
                    new_individual_rate):
        self.input_table = input_table
        self.output_table = output_table

        orw = fwrapper(lambda l:l[0] or l[1], 2, 'or')
        andw = fwrapper(lambda l:l[0] and l[1], 2, 'and')
        gtw = fwrapper(lambda l:l[0] > l[1], 2, '>')
        ltw = fwrapper(lambda l:l[0] < l[1], 2, '<')
        eqw = fwrapper(lambda l:l[0] == l[1], 2, '==')
        
        self.function_set = set([orw, andw, gtw, ltw, eqw])

        self.logical_operators = [orw, andw]
        self.comparison_op = [gtw, ltw, eqw]
        
        #self.terminal_set = self.get_balanced_terminals()
        #self.all_nodes = list(self.function_set) + self.terminal_set

        #self.terminal_set = self.get_all_terminals()
        self.terminal_set = self.get_all_possible_sub_trees()
        self.all_nodes = self.function_set | self.terminal_set
        self.fitness_function = fitness_function
        self.progress = []

        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.min_individual_size = min_individual_size
        self.max_individual_size = max_individual_size

        self.elite_size = elite_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.new_individual_rate = new_individual_rate
        self.num_fitness_calculations = 0

    def get_all_terminals(self):
        terminals = []
        for c in self.input_table.columns:
            terminals.append(paramnode(c))
            terminals.extend([constnode(val) for val in self.input_table[c].unique()])
        return set(terminals)

    def get_balanced_terminals(self):
        terminals_dict = self.get_all_composed_terminals()
        max_column = None
        max_terminal_length = -1
        avg_length = 0
        count = 0
        for col, term in terminals_dict.items():
            if len(term) > max_terminal_length:
                max_column = col
                max_terminal_length = len(term)
            avg_length = len(term)
            count += 1

        avg_length = int(avg_length/count)
        
        balanced_terminals = []
        for col, terminals in terminals_dict.items():
            term = terminals
            if col == max_column:
                sampled = resample(list(terminals), 
                                        replace=True,     # sample with replacement
                                        n_samples=min(len(term), avg_length),    # to match majority class
                                        random_state=123) # reproducible results
                
                term = sampled
            balanced_terminals.extend(term)

        shuffle(balanced_terminals)
        return balanced_terminals

    def get_all_composed_terminals(self):
        terminals_dict = dict()
        columns_subtrees = set()

        for col1 in self.input_table.columns:
            subtrees = set()
            for col2 in self.input_table.columns:
                if col1 != col2:
                    columns_subtrees.add(node(self.comparison_op[-1], [paramnode(col1), paramnode(col2)]))
        
        terminals_dict['columns'] = columns_subtrees

        for col in self.input_table.columns:
            subtrees = set()
            for op in self.comparison_op:
                values = set(self.input_table[col].values)
                if len(values) == 1:
                    subtrees.add(node(self.comparison_op[-1], [paramnode(col), constnode(list(values)[0])]))
                else:
                    for val in values:
                        if isinstance(val, str):
                            subtrees.add(node(self.comparison_op[-1], [paramnode(col), constnode(val)]))
                        else:
                            for op in self.comparison_op:
                                subtrees.add(node(op, [paramnode(col), constnode(val)]))
            terminals_dict[col] = subtrees

        return terminals_dict

    def get_all_possible_sub_trees(self):
        subtrees = set()
        
        for op in self.comparison_op:
            for col in self.input_table.columns:                
                for col2 in self.input_table.columns:
                    if col2 != col:
                        subtrees.add(node(op, [paramnode(col), paramnode(col2)]))
                values = set(self.input_table[col].values)
                if len(values) == 1:
                    subtrees.add(node(self.comparison_op[-1], [paramnode(col), constnode(list(values)[0])]))
                else:
                    for val in values:
                        if isinstance(val, str):
                            subtrees.add(node(self.comparison_op[-1], [paramnode(col), constnode(val)]))
                        else:
                            for op in self.comparison_op:
                                subtrees.add(node(op, [paramnode(col), constnode(val)]))

        return subtrees

    def choose_at_random(self, nodes):
        return sample(nodes, 1)[0]

    def grow_method(self, depth):
        if depth > 0 and depth >= self.min_individual_size:
            new_node = self.choose_at_random(self.all_nodes)
            if isinstance(new_node, fwrapper):
                children = [self.grow_method(depth - 1) for c in range(new_node.childcount)]
                new_node = node(new_node, children)
        else:
            new_node = self.choose_at_random(self.terminal_set)

        return new_node

    def full_method(self, depth):
        if depth > 0:
            new_node = self.choose_at_random(self.logical_operators)
            children = [self.full_method(depth - 1) for c in range(new_node.childcount)]
            new_node = node(new_node, children)
        else:
            new_node = self.choose_at_random(self.terminal_set)

        return new_node
    
    def ramped_half_and_half_create_population(self, population_size):
        population = []
        num_groups = self.max_individual_size
        elements_per_group = int(population_size / num_groups)

        depth = self.min_individual_size
        for g in range(num_groups):
            for e in range(elements_per_group):
                if e < int(elements_per_group / 2):
                    new_node = self.grow_method(depth)
                else:
                    new_node = self.full_method(depth)
                population.append(new_node)
            depth += 1

        return population

    def mutation(self, individual):
        mutated = deepcopy(individual)
        if random() < self.mutation_rate:
            point = randint(1, len(individual.children)-1)
            depth = randint(self.min_individual_size, 3)
            mutated.children[point] = self.full_method(depth)

        return mutated

    def mutate(self, individual):
        if random() < self.mutation_rate:
            depth = randint(self.min_individual_size, self.max_individual_size-1)
            return self.full_method(depth)
        else:
            result = deepcopy(individual)
            if isinstance(individual, node):
                result.children = [self.mutate(c) for c in individual.children]
            return result

    def recombine(self, t1, t2):
        offspring = deepcopy(t1)

        if random() < self.crossover_rate:
            cxpointA = 1
            cxpointB = 0

            if isinstance(t1, node):
                if len(t1.children) > 1:
                    cxpointA = randint(1, len(t1.children)-1)
            
            if isinstance(t2, node):
                if len(t2.children) > 1:
                    cxpointB = randint(1, len(t1.children)-1)

            offspring.children[cxpointA:] = t2.children[:cxpointB]
            
        return offspring

    def crossover(self, t1, t2, top=1):
        if random() < self.crossover_rate and not top:
            return deepcopy(t2)
        else:
            result = deepcopy(t1)
            if isinstance(t1, node) and isinstance(t2, node):
                result.children = [self.crossover(c, choice(t2.children), 0) for c in t1.children]
            return result
            
    def calculate_fitness(self, individual):
        if individual.get_depth() > 2*self.max_individual_size or individual.get_depth() < self.min_individual_size:
            return self.fitness_function.default_value
        
        try:
            actual_table = self.input_table.query(individual.str_representation()) #self.pysqldf('SELECT * FROM input_table WHERE '+ individual.str_representation()) 
        except:
            return self.fitness_function.default_value

        self.num_fitness_calculations += 1
        return self.fitness_function.evaluate(actual_table, self.output_table)

    def rank_population(self, population):
        fitpopulation = [(p, self.calculate_fitness(p)) for p in population]
        return sorted(fitpopulation, key=operator.itemgetter(1), reverse=True)

    def tournament_selection(self, ranked_population, tournament_size, num_selected_individuals):
        selected_individuals = []
        while len(selected_individuals) < num_selected_individuals:
            participants = [randint(0, len(ranked_population)-1) for t in range(tournament_size)]
            winner = max(participants)
            if not ranked_population[winner] in selected_individuals:
                selected_individuals.append(ranked_population[winner])

        return selected_individuals

    def select_index(self):
        return min(self.population_size - 1, int(log(random())/log(0.7)))

    def search(self, iterations, max_iterations_without_improvement=10, verbose=False):
        no_improvement_count = 0
        
        population = self.ramped_half_and_half_create_population(self.population_size)
        hall_of_fame = (population[0], self.fitness_function.default_value)

        for i in range(iterations):
            ranked_population = self.rank_population(population)
            self.progress.append(self.fitness_function.cut_value - ranked_population[0][1])
            
            if verbose: 
                print(str.format('Iteration: {0:000}, Best Fitness: {1:0.8f}, Fitness Calculations: {2}', i+1, ranked_population[0][1], self.num_fitness_calculations))
            
            if ranked_population[0][1] > hall_of_fame[1]:
                hall_of_fame = ranked_population[0]
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= max_iterations_without_improvement:
                print('Reached max iterations without improving')
                break

            if ranked_population[0][1] >= self.fitness_function.cut_value:
                if verbose:
                    print('------------------------')
                    print('Solution FOUND')
                    print('------------------------')
                break
            
            newpop = [p[0] for p in ranked_population[:self.elite_size]]

            while len(newpop) < self.population_size:
                new_individual = None
                if random() > self.new_individual_rate:
                    parents = self.tournament_selection(ranked_population, 10, 2)
                    t1 = parents[0][0]
                    t2 = parents[1][0]
                    a_offspring = self.mutation(self.recombine(t1, t2))
                    b_offspring = self.mutation(self.recombine(t2, t1))
                    newpop.append(a_offspring)
                    newpop.append(b_offspring)

                else:
                    new_individual = self.full_method(randint(self.min_individual_size, self.max_individual_size-1))
                    newpop.append(new_individual)
                    
            population = newpop

        return hall_of_fame[0]