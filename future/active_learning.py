# SEE: https://github.com/modAL-python/modAL#installation
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from modAL.models import ActiveLearner
import matplotlib as mpl
import matplotlib.pyplot as plt

def tutorial():
    # Set our RNG seed for reproducibility.
    RANDOM_STATE_SEED = 123
    np.random.seed(RANDOM_STATE_SEED)

    iris = load_iris()
    X_raw = iris['data']
    y_raw = iris['target']

    # Isolate our examples for our labeled dataset.
    n_labeled_examples = X_raw.shape[0]
    training_indices = np.random.randint(low=0, high=n_labeled_examples + 1, size=3)

    X_train = X_raw[training_indices]
    y_train = y_raw[training_indices]

    # Isolate the non-training examples we'll be querying.
    X_pool = np.delete(X_raw, training_indices, axis=0)
    y_pool = np.delete(y_raw, training_indices, axis=0)

    # Specify our core estimator along with it's active learning model.
    knn = KNeighborsClassifier(n_neighbors=3)
    learner = ActiveLearner(estimator=knn, X_training=X_train, y_training=y_train)

    # Isolate the data we'll need for plotting.
    predictions = learner.predict(X_raw)
    is_correct = (predictions == y_raw)

    # Record our learner's score on the raw data.
    unqueried_score = learner.score(X_raw, y_raw)
    print("ActiveLearner class predictions (Accuracy: {score:.3f})".format(score=unqueried_score))

    N_QUERIES = 20
    performance_history = [unqueried_score]

    # Allow our model to query our unlabeled dataset for the most
    # informative points according to our query strategy (uncertainty sampling).
    for index in range(N_QUERIES):
        query_index, query_instance = learner.query(X_pool)

        # Teach our ActiveLearner model the record it has requested.
        X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
        learner.teach(X=X, y=y)

        # Remove the queried instance from the unlabeled pool.
        X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)

        # Calculate and report our model's accuracy.
        model_accuracy = learner.score(X_raw, y_raw)
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))

        # Save our model's performance for plotting.
        performance_history.append(model_accuracy)

    # Plot our performance over time.
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

    ax.plot(performance_history)
    ax.scatter(range(len(performance_history)), performance_history, s=13)

    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

    ax.set_ylim(bottom=0, top=1)
    ax.grid(True)

    ax.set_title('Incremental classification accuracy')
    ax.set_xlabel('Query iteration')
    ax.set_ylabel('Classification Accuracy')

    plt.show()

def active_learning(X_train, y_train, X_pool, y_pool, X_raw, y_raw, estimator, number_queries):
    learner = ActiveLearner(estimator=estimator, X_training=X_train, y_training=y_train)
    for index in range(number_queries):
        query_index, query_instance = learner.query(X_pool)

        # Teach our ActiveLearner model the record it has requested.
        X = X_pool[query_index]
        print('Instance: {0}, {1}'.format(X, y_pool[query_index]))
        y = np.array(int(input('Is positive? (y/n)? ') in ['Y', 'y'])).reshape(1, )

        learner.teach(X=X, y=y)

        # Remove the queried instance from the unlabeled pool.
        X_pool = np.delete(X_pool, query_index, axis=0)

    predictions = learner.predict(X_raw)
    is_correct = (predictions == y_raw)
    print(is_correct)
    model_accuracy = learner.score(X_raw, y_raw)
    print('Accuracy: {acc:0.4f}'.format(acc=model_accuracy))

def iris():
    iris = load_iris()
    X_raw = iris['data']
    y_raw = np.array([ int(c == 2) for c in iris['target']])

    n_labeled_examples = X_raw.shape[0]
    
    # Isolate our examples for our labeled dataset.
    training_indices = np.random.randint(low=0, high=n_labeled_examples + 1, size=10)

    X_train = X_raw[training_indices]
    y_train = y_raw[training_indices]

    # Isolate the non-training examples we'll be querying.
    X_pool = np.delete(X_raw, training_indices, axis=0)
    y_pool = np.delete(y_raw, training_indices, axis=0)

    #estimator = KNeighborsClassifier(n_neighbors=3)
    estimator = GaussianNB()

    active_learning(X_train, y_train, X_pool, y_pool, X_raw, y_raw, estimator, number_queries=20)

if __name__ == "__main__":
    iris()
    #tutorial()
