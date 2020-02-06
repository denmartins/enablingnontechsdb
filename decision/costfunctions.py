def top_limit_cost_function(limit, value):
    cost = value - limit
    if(cost < 0): 
        cost = 0
    return cost

def bottom_limit_cost_function(limit, value):
    cost = limit - value
    if(cost < 0): 
        cost = 0
    return cost

def categorical_cost_function(desired, value):
    return 1 - int(desired == value)