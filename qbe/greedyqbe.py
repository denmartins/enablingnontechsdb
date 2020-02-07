import qbe.util as util

class GreedySearchQBE(object):
    def __init__(self, fitness_function, dataframe):
        self.fitness_function = fitness_function
        self.original_data_columns = dataframe.columns
        self.data_columns = ['x' + str(i) for i in range(len(dataframe.columns))]
        self.possible_predicates = self.enumerate_predicates(dataframe)
    
    def enumerate_predicates(self, dataframe):
        comparison_operators = ['>', '>=', '<', '<=', '!=', '=='] #['operator.lt', 'operator.le', 'operator.gt', 'operator.ge', 'operator.eq', 'operator.ne']
        predicates = []
        for i in range(len(dataframe.columns)):
            col = dataframe.columns[i]
            support_values = set(dataframe[col])
            for val in support_values:
                    for cop in comparison_operators:
                        predicates.append('(' + self.data_columns[i] +  ' ' + cop + ' ' + str(val) + ')')
                        #predicates.append(cop + '(' + self.data_columns[i] + ', ' + str(val) + ')')
        return predicates

    def search_best_predicate(self, max_iterations=100, threshold=0.001, verbose=True):
        best_fitness = float('inf') # Initialize with max value
        best_predicate = ''
        iteration = 1

        if verbose:
            print('------------------------------')
            print('Iteration \t Best fitness')

        # Search for a solution with only one predicate
        best_predicate, best_fitness, reached_threshold = self.search_predicate(best_predicate, best_fitness, threshold)
        if verbose: print('{0} \t\t {1:02.4f}'.format(iteration, best_fitness))

        # If the solution was not found, search for more predicates to compose the solution
        while not reached_threshold and iteration < max_iterations:
            last_fitness_found = best_fitness
            iteration += 1
            for bop in [' and ', ' or ']:
                predicate_best_found, fitness_best_found, reached_threshold = self.search_predicate(best_predicate + bop, best_fitness, threshold)

                if fitness_best_found < best_fitness:
                    best_fitness = fitness_best_found
                    best_predicate = predicate_best_found
                    
                    if reached_threshold:
                        break
            
            # Stop loop if it got no improvement
            if last_fitness_found == best_fitness:
                break

            if verbose: print('{0} \t\t {1:02.4f}'.format(iteration, best_fitness))

        if verbose: print('-------- Search ended --------')
        return self.format_predicate_to_sql(best_predicate)
        

    def search_predicate(self, best_predicate, best_fitness, threshold):
        current_best_predicate = best_predicate
        current_best_fitness = best_fitness
        reached_threshold = False

        for predicate in self.possible_predicates:
            if predicate not in best_predicate:
                current_predicate = best_predicate + predicate
                func = self.predicate_to_function(current_predicate)
                fitness = self.fitness_function.calculate(func)
                
                if fitness < current_best_fitness:
                    current_best_fitness = fitness
                    current_best_predicate = current_predicate
                    
                    if current_best_fitness <= threshold:
                        reached_threshold = True
                        break

        return current_best_predicate, current_best_fitness, reached_threshold
    
    def predicate_to_function(self, predicate):
        function_body = 'lambda {0}: {1}'

        params = ''
        for col in self.data_columns:
            params += col + ','

        # Remove last comma
        params = params[:-1]
        function_body = function_body.format(params, predicate)

        func = eval(function_body)
        
        return func

    def format_predicate_to_sql(self, predicate):
        sql_predicate  = predicate[:]
        sql_predicate = sql_predicate.replace(' and ', ' AND ')
        sql_predicate = sql_predicate.replace(' or ', ' OR ')
        sql_predicate = sql_predicate.replace('!=', '<>')
        for c in range(len(self.data_columns)):
            sql_predicate = sql_predicate.replace(self.data_columns[c], self.original_data_columns[c])

        return sql_predicate