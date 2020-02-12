import pylab as plt, minisom
from scipy.spatial import distance as spd
from models.car import Car
import operator

class SomSuggestionStrategy():
    def __init__(self, candidates, preprocessed_data):
        self.data = preprocessed_data.as_matrix()
        self.som = minisom.MiniSom(9, 9, len(preprocessed_data.columns), sigma=1.0, learning_rate=0.8)
        self.som.train_random(self.data, 1000)
        #print('Quantization error: %.3f' % self.som.quantization_error(self.data))

    def get_suggestions(self, num_retrieved, optimal_candidates):
        similar = []

        for cand in optimal_candidates.index.values:
            # Adjusting index from pandas
            candidate = cand - 1
            elements = {}
            winner = self.som.winner(self.data[candidate])
        
            for index in range(len(self.data)):
                if index == candidate:
                    continue
                w = self.som.winner(self.data[index])
                distance = spd.cityblock(list(winner), list(w))
                elements[index] = distance
        
            sorted_candidates = sorted(elements.items(), key=operator.itemgetter(1))
            # +1 to adjust index for pandas
            best = [ind[0] + 1 for ind in sorted_candidates[:num_retrieved]]
            similar.extend([x for x in best if (not x in similar) and (not x in optimal_candidates.index.values)])

        return similar
    
    def get_suggestions_via_example(self, example):
        elements = {}        
        winner = self.som.winner(example)
        for index in range(len(self.data)):
            w = self.som.winner(self.data[index])
            distance = spd.cityblock(list(winner), list(w))
            elements[index] = distance
        
        sorted_candidates = sorted(elements.items(), key=operator.itemgetter(1))
        best = [x[0] + 1 for x in sorted_candidates[:5]]
        return best
        
    def print_optimal(self, selected_candidates, optimal_candidates, total_candidates, complete_print = True):
        plt.bone()
        plt.pcolor(self.som.distance_map().T, cmap='Greys', edgecolors='#1F442A', linewidths=1)
        plt.colorbar()

        winners = {}

        offset = 0.5

        for index in total_candidates.index.values:
            if index in optimal_candidates:
                marker = 'o'
                color = 'red'
            elif index in selected_candidates:
                marker = 's'
                color = 'blue'
            else:
                marker = 'x'
                color = 'green'
            
            w = self.som.winner(self.data[index - 1])

            if not w in winners:
                winners[w] = marker + ': ' + str(total_candidates.loc[index][Car.TYPE])
            else:
                winners[w] = winners[w] + '\n' + marker + ': ' + str(total_candidates.loc[index][Car.TYPE])
            
            plt.plot(w[0] + offset, w[1] + offset, marker, markerfacecolor='None', 
                markeredgecolor=color, markersize=12, markeredgewidth=2)

        for win, text in winners.items():
            plt.text(win[0] + 0.2, win[1] + 0.5, str(text), fontsize=8, 
                     bbox=dict(facecolor='white', alpha=0.8, pad = 0.2))

        plt.axis([0,self.som.weights.shape[0],0,self.som.weights.shape[1]])
        plt.show()
