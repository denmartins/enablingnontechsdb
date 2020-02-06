class BaseSADA(object):
    def __init__(self, dataset, selector, evaluator, discriminator, adjuster):
        self.dataset = dataset
        self.selector = selector
        self.evaluator = evaluator
        self.discriminator = discriminator
        self.adjuster = adjuster

    def select(self, query, num_of_selected_candidates):
        raise NotImplementedError('You have to implement the select method')
    
    def assess(self, candidates, criteria):
        raise NotImplementedError('You have to implement the assess method')

    def discriminate(self, candidates_to_discriminate):
        raise NotImplementedError('You have to implement the discriminate method')
    
    def adjust(self, query):
        raise NotImplementedError('You have to implement the adjust method')