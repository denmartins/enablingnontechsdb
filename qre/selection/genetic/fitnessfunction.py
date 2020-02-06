class AbstractFitnessFunction(object):
    def __init__(self, default_value, cut_value):
        self.default_value = default_value
        self.cut_value = cut_value
    
    def evaluate(self, individual):
        raise NotImplementedError()

class AbstractFitnessFunctionQRE(AbstractFitnessFunction):
    def evaluate(self, actual_table, output_table, verbose=False):
        raise NotImplementedError()

class ClassificationFitness(AbstractFitnessFunctionQRE):
    def evaluate(self, actual_table, output_table, verbose=False):
        if actual_table.empty:
            return self.default_value
        
        relevant = set(output_table[output_table > 0].index)
        retrieved = set(actual_table.index)

        precision = len(relevant & retrieved)/len(retrieved)
        recall = len(relevant & retrieved)/len(relevant)

        if recall < 1:
            return 0
            
        if precision + recall > 0:
            fscore = 2*precision*recall/(precision + recall)
        else:
            fscore = 0.0

        return fscore

class InOutFitnessFunction(AbstractFitnessFunctionQRE):
    def contained(self, candidate, container):
        temp = container[:]
        try:
            for v in candidate:
                temp.remove(v)
            return True
        except ValueError:
            return False
            
    def evaluate(self, actual_table, output_table, verbose=False):
        fit = self.default_value
        if actual_table.shape[0] < 1 or actual_table.shape[1] < 1:
            return fit
            
        matched_columns = 0
        for actcol in actual_table.columns:
            for outcol in output_table.columns:
                if self.contained(output_table[outcol].values.tolist(), actual_table[actcol].values.tolist()):
                    matched_columns += 1

        if matched_columns <= 0:
            return fit
        else:
            matched_columns = min(output_table.shape[1], matched_columns)
        
        matched_rows = 0
        for outrow in output_table.values:
            for actrow in actual_table.values:
                if self.contained(list(outrow), list(actrow)):
                    matched_rows += 1
        
        if matched_rows > 0:
            matched_rows = min(output_table.shape[0], matched_rows)

        row_fscore = 0
        row_recall = matched_rows/output_table.shape[0]
        row_precision = matched_rows/actual_table.shape[0]
        if row_precision != 0 or row_recall != 0:
            row_fscore = 2*(row_precision * row_recall)/(row_precision + row_recall)

        fit = matched_columns/output_table.shape[1] + row_fscore
        
        return fit