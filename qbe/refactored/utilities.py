def replace_feature_names(query, feature_names):
    for i in range(len(feature_names)):
        variable = 'ARG' + str(i)
        query = query.replace(variable, feature_names[i])
    return query

def adjust_expression_for_convertion(expression):
    return expression.replace("'", '').replace(' ', '').replace('(',',').replace(')', '').split(',')

def rename_operators(expression):
    # Operators have the form **op** to avoid conflict with column names
    operators = {'**gt**': '>', '**lt**': '<', '**eq**': '=',  '**and_**': 'AND', '**or_**': 'OR', '**ge**': '>=', '**le**': '<=', '**ne**': '<>'}

    text = str(expression)
    for key, value in operators.items():
        text = text.replace(key, value)

    return text

def pref2inf(expression):
    adjusted = adjust_expression_for_convertion(expression)
    stack = []
    l = adjusted[::-1]
    for e in l:
        if e in ['gt', 'eq', 'lt', 'and_', 'or_', 'add', 'ge', 'le', 'ne']:
            op1 = stack.pop()
            op2 = stack.pop()
            stack.append('(%s **%s** %s)' % (op1, e, op2)) # Operator's form **op**
        else:
            if e == 'notapplied':
                op = stack.pop()
                if type(op) == bool and op == True: # To handle the True statement of DEAP GP
                    pass
                else:
                    stack.append(op)
            else:
                stack.append(e)
    return rename_operators(stack.pop())

def classification_rule_2_sql(classification_rule):
    select_statement_query = "SELECT * FROM dataframe WHERE {}".format(classification_rule)
    return select_statement_query

def genetic_2_predicate(genetic_individual):
    return rename_operators(pref2inf(str(genetic_individual)))

# Convert a NumPy array into a set to facilitate fitness calculation
def convert_nparray_to_set(nparray):
    converted_set = set([tuple(i) for i in nparray.values[0:nparray.size,:]])
    return converted_set

def get_precision(actual_set, relevant_set):
    try:
        return len(relevant_set & actual_set)/len(actual_set)
    except ZeroDivisionError:
        return 0

def get_recall(actual_set, relevant_set):
    try:
        return len(relevant_set & actual_set)/len(relevant_set)
    except ZeroDivisionError:
        return 0

def get_specificity(actual_set, relevant_set, irrelevant_set):
    false_positive = irrelevant_set & actual_set
    true_negative = irrelevant_set - false_positive
    try:
        return len(true_negative)/len(true_negative | false_positive)
    except ZeroDivisionError:
        return 0

def get_fscore(actual_set, relevant_set, factor=1):
    precision = get_precision(actual_set, relevant_set)
    recall = get_recall(actual_set, relevant_set)
    if precision + recall > 0:
        return (1+factor**2)*precision * recall / ((factor**2)*precision + recall)
    else:
        return 0