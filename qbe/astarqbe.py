from queue import PriorityQueue
import qbe.util as util

class Graph(object):
    def __init__(self, dataframe):
        self.predicates = util.enumerate_predicates(dataframe)

    def get_cost(self, from_node, to_node):
        return 0.1

    def get_neighbors(self, current):
        neighbor_list = []
        for pred in self.predicates:
            if current == '':
                neighbor_list.append(pred)
            elif not pred in current:
                for bop in [' AND ', ' OR ']:
                    neighbor_list.append(current + bop + pred)

        return neighbor_list

class AstarQBE(object):
    def __init__(self, heuristic):
        self.heuristic = heuristic
    
    def search_best_predicate(self, dataframe, max_iterations=100, threshold=0.001):
        graph = Graph(dataframe)
        
        start = ''
        count = 0
        iteration = 0

        frontier = PriorityQueue()
        frontier.put((0, count, start))
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0

        best_fitness = float('inf') # Initialize with max value
        best_predicate = start

        hall_of_fame = []

        print('------------------------------')
        print('Iteration \t Best fitness')

        while not frontier.empty() and iteration < max_iterations:
            current = frontier.get()[2]
            fitness = self.heuristic(current)

            if fitness < best_fitness:
                best_fitness = fitness
                best_predicate = current
                
            print('{0}\t\t{1:02.4f}'.format(iteration + 1, best_fitness))

            if fitness <= threshold:
                break

            if best_fitness not in hall_of_fame:
                hall_of_fame.append(best_fitness)
            else:
                break

            for neighbor in graph.get_neighbors(current):
                count += 1
                new_cost = cost_so_far[current] + graph.get_cost(current, neighbor)

                if (neighbor not in cost_so_far) or (new_cost < cost_so_far[neighbor]):
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor)
                    frontier.put((priority, count, neighbor))
                    came_from[neighbor] = current

            iteration += 1
        
        print('-------- Search ended --------')
        return best_predicate #return came_from, cost_so_far