import operator

class DiadicClause(object):
    def __init__(self, left_token, operation, right_token):
        self.left_token = left_token
        self.operation = operation
        self.right_token = right_token

    def to_string(self):
        return self.left_token, self.operation, self.right_token

class Operation(object):
    EQUALS = operator.eq
    GREATER_THAN = operator.gt
    GREATER_THAN_EQUALS = operator.ge
    LESS_THAN = operator.lt
    LESS_THAN_EQUALS = operator.le