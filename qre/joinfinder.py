class AbstractJoinFinder(object):
    def __init__(self, schema_graph):
        self.schema_graph = schema_graph

    def traverse_graph(self, graph, start, path, visited):
        visited.add(start)
        next_nodes = [n for n in graph[start] if not n in visited]
        if len(next_nodes) <= 0:
            path.append('#%#')
            return None
        else:
            for n in next_nodes:
                if n in visited:
                    continue
                else:
                    path.append(n)
                    self.traverse_graph(graph, n, path, visited)

class PKFKJoinFinder(AbstractJoinFinder):
    def enumerate_candidate_joins(self, starting_table):
        start = starting_table
        joins = self.get_fk_join_paths(self.schema_graph, start)
        candidate_joins_and_tables = []
        for j in joins:
            condition = str.format('{0}.id{0} = {1}.id{0}', starting_table, j[0])
            for i in range(len(j)-1):
                condition += str.format(' AND {0}.id{1} = {1}.id{1}', j[i], j[i+1])
            candidate_joins_and_tables.append((condition, j))
        return candidate_joins_and_tables
    
    def get_fk_join_paths(self, graph, starting_table):
        path = []
        paths = []
        self.traverse_graph(graph, starting_table, path, set())
        temp = []
        for t in path:
            if not t == '#%#':
                temp.append(t)
            else:
                temp2 = [x for x in temp]
                paths.append(temp2)
                temp = []
        return paths

class NatualJoinFinder(AbstractJoinFinder):
    def natural_join_tables(self, tables):
        candidate_tables = dict()
        for itab, idata in tables.items():
            candidate_tables[itab] = dict()
            for jtab, jdata in tables.items():
                if itab != jtab:
                    for col in idata['name'].values:
                        if col in jdata['name'].values:
                            if not jtab in candidate_tables[itab].keys():
                                candidate_tables[itab][jtab] = []
                            candidate_tables[itab][jtab].append(col)

        return candidate_tables