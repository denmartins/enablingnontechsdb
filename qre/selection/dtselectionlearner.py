import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from selectionlearner import AbstractSelectionLearner

or_separator = '#or#'
and_separator = '#and#'

class DTSelectionLearner(AbstractSelectionLearner):
    def create_training_data(self, example, sketch_result, positive_indicies, convert_categorical_values=False, verbose=False):
        return super().create_training_data(example, sketch_result, positive_indicies, True, verbose)

    def tree_2_classification_rule(self, tree):
        left      = tree.tree_.children_left
        right     = tree.tree_.children_right
        threshold = tree.tree_.threshold
        features  = [self.feature_columns[i] for i in tree.tree_.feature]
        value = tree.tree_.value

        rules = []

        n_nodes = tree.tree_.node_count
        is_leaf = np.zeros(shape=n_nodes, dtype=bool)

        node_parenting = {}

        root = (0, [])
        stack = [root]

        # Trasverse tree to get an appropriate representation
        while len(stack) > 0:
            node_id, parents = stack.pop()
            node_parenting[node_id] = parents

            # If we have a test node
            if (left[node_id] != right[node_id]):
                l_heritage = parents[:]
                l_heritage.append(('l', node_id))

                r_heritage = parents[:]
                r_heritage.append(('r', node_id))

                stack.append((left[node_id], l_heritage))
                stack.append((right[node_id], r_heritage))
            else:
                is_leaf[node_id] = True

        # Get rules by finding paths from leaf to root
        for node, parents in node_parenting.items():
            if is_leaf[node] and value[node][0][1] > 0:
                current_rule = []
                for direction, p_node in parents:
                    if direction == 'l': # Left direction indicates that the parent condition is true
                        current_rule.append(features[p_node] + " <= " + str(threshold[p_node]))
                    else: # Change the parent condition is parent direction is right (False)
                        current_rule.append(features[p_node] + " > " + str(threshold[p_node]))
                    current_rule.append(' ' + and_separator + ' ')
                    
                rules.append(''.join(current_rule[:-1])) # Remove last AND
                rules.append( ' ' + or_separator + ' ')
                
                # Stop the loop if all relevant items were classified correctly in a leaf
                if value[node][0][1] == self.num_tuples_in_example:
                    break

        rules = rules[:-1] # Remove last OR

        predicate = ''.join(rules)
        return predicate

    def transform_condition(self, condition):
        transformed_condition = ''

        if not '_' in condition:
            transformed_condition = condition
        else:
            column_value_pair = condition.strip().split('_')
            transformed_condition = ''
            if '>' in column_value_pair[1]:
                if 'ISNULL' in column_value_pair[1]:
                    transformed_condition = str.format("{0} IS NULL", column_value_pair[0])
                else:
                    transformed_condition = str.format("{0} = '{1}'", column_value_pair[0], column_value_pair[1].split(' >')[0])
            else:
                transformed_condition = str.format("{0} != '{1}'", column_value_pair[0], column_value_pair[1].split(' <')[0])
        
        return transformed_condition
    
    def get_selection_predicate(self, tree_classification_rule):
        processed_selection_criteria = ''

        selpredicates = tree_classification_rule.split(or_separator)
        inner_predicates = []
        for pred in selpredicates:
            if and_separator in pred:
                conditions = []
                and_conditions = pred.strip().split(and_separator)
                for c in and_conditions:
                    conditions.append(self.transform_condition(c))
                inner_predicates.append('(' + ' AND '.join(conditions) + ')')
            else:
                inner_predicates.append(self.transform_condition(pred))

        processed_selection_criteria = ' OR '.join(inner_predicates)    
        return processed_selection_criteria.strip()

    def search(self):
        tree_classifier = DecisionTreeClassifier()
        tree_classifier = tree_classifier.fit(self.X, self.y)
        classification_rule = self.tree_2_classification_rule(tree_classifier)
        return str.format('({0})', self.get_selection_predicate(classification_rule))