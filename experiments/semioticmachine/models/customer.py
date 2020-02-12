class Customer:
    BUDGET_LIMIT = 'budget_limit'
    AGE = 'age'
    KIDS = 'num_of_kids'
    
    CIVIL_STATE = 'civil_state'
    MARRIED_CIVIL_STATE = 'married'
    SINGLE_CIVIL_STATE = 'single'
    
    SEX = 'sex'
    MALE_SEX = 'male'
    FEMALE_SEX = 'female'
    
    def __init__(self):
        self.preferences = []
        self.profile = []
        self.criteria = []