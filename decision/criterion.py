class Criterion(object):
    def __init__(self, index, maximize, weight):
        self.index = index
        self.maximize = maximize
        self.weight = weight