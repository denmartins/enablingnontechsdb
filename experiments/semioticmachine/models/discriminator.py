import pylab as plt, seaborn as sns, pickle, os.path, pandas as pd
from sklearn.cluster import KMeans
from models.car import Car

class NullObjectDiscriminator():
    def discriminate(self, candidates_to_discriminate, candidates):
        return None

class KmeansDiscriminator():
    def __init__(self, preprocessed_dataset):
        self.kmeans = None
        self.preprocessed_data = preprocessed_dataset.copy(True)
        self.columns = preprocessed_dataset.columns

    def discriminate(self, candidates_to_discriminate, candidates):
        filename = 'database\\kmeans.sav'
        if os.path.isfile(filename):
            self.kmeans = pickle.load(open(filename, 'rb'))
            clusters = self.kmeans.predict(self.preprocessed_data[self.columns].as_matrix())
        else:
            self.kmeans = KMeans(n_clusters=4)
            clusters = self.kmeans.fit_predict(self.preprocessed_data[self.columns].as_matrix())
            pickle.dump(self.kmeans, open(filename, 'wb'))
        
        self.preprocessed_data['cluster'] = clusters
        cluster_means = self.preprocessed_data.groupby(['cluster'], as_index=False).mean()
        cluster_columns = [Car.PRICE, Car.HORSEPOWER, Car.MPG, Car.FUEL_TANK_CAPACITY, 
                           Car.PASSENGER_CAPACITY, Car.LENGTH, Car.ORIGIN]

        candidates['cluster'] = clusters
        candidates.loc[self.preprocessed_data[self.preprocessed_data['cluster'] == 0].index.values]['cluster'] = 'Expensive, high potency, midsized cars'
        candidates.loc[self.preprocessed_data[self.preprocessed_data['cluster'] == 1].index.values]['cluster'] = 'Large, high potency, american cars'
        candidates.loc[self.preprocessed_data[self.preprocessed_data['cluster'] == 2].index.values]['cluster'] = 'Sportive, midsized, balanced, american cars'
        candidates.loc[self.preprocessed_data[self.preprocessed_data['cluster'] == 3].index.values]['cluster'] = 'Small, high fuel economy, cheap cars'

        self.print_heatmap(cluster_means, cluster_columns)
    
    def print_heatmap(self, cluster_means, cluster_columns):
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(cluster_means[cluster_columns], annot=True)

        plt.show()

    def print_items_per_cluster(self, candidates):
        x = 0
        y = 0
        for i in range(4):
            elements = self.preprocessed_data[self.preprocessed_data['cluster'] == i].index.values
            for j in range(len(elements)):
                if (j + 1) % 8 == 0:
                    x = 0
                    y = y + 0.5

                plt.text(x, y, str(candidates.loc[elements[j]][Car.MAKE]))
                x = x + 2.5
            x = 0
            y = y + 2.0
        
        markers=['o', 'x', '^', 'h']

        x = 0.2
        y = 0
        for i in range(len(self.preprocessed_data)):
            if i % 10 == 0:
               x = 0.2
               y = y + 0.8
            
            plt.plot(x, y, markers[int(self.preprocessed_data.iloc[i]['cluster'])], markerfacecolor='None', markersize=12, markeredgewidth=2)
            plt.text(x, y, str(candidates.iloc[i][Car.MAKE]), fontsize=8, bbox=dict(facecolor='white', alpha=0.8, pad = 0.2))
            x = x + 1.0
        
        plt.axis([0,20,0,15])