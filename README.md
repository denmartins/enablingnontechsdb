# Enabling non-technical users to query and purchase data

Repository to store the implementation details of the Intelligent Semiotic Machine described in the Ph.D. dissertation "Enabling Non-technical Users to Query and Purchase Data" of Denis Mayr Lima Martins.

## Experimental Setup
All implementations have been developed in the Python programming language. The code requires Python 3.4 or later, and does not run in Python 2.

### Python libraries used in this project
Next, we described all libraries that have been used to develop the techniques described in the thesis.

**Self-Organizing Map (SOM)**: The library [MiniSom](https://github.com/JustGlowing/minisom) has been used. MiniSom offers a simple API for training and visualizing SOMs that facilitates the development of custom functionalities. MiniSom has been employed in several [publications] (https://github.com/JustGlowing/minisom#who-uses-minisom). In the thesis, MiniSom has been employed througout Chapters 3 and 4. We highlight that a custom implementation of the SOM-based indexing technique described in the thesis has been conducted. This technique has been used for finding similar data items by analyzing the topological map created by the SOM.

**Genetic Programming (GP)**: The library [Distributed Evolutionary Algorithms in Python (DEAP)](https://github.com/DEAP/deap#projects-using-deap) has been employed. DEAP offers a comprehensive API that facilitates the construction of custom GP algorithms, such as representation of individual, results visualization, and genetic operators. A custom implementation of the GP technique based on the DEAP framework has been employed in Chapter 4 of the thesis.

**Genetic Algorithm (GA)**: The library [Inspyred](https://pythonhosted.org/inspyred/) has been employed. Inspyred offers diverse implementations of genetic operators, such as recombination, mutation, and reproduction. In particular, we have used the GA implementation offered by Inspyred to find solutions to the [Knapsack Problem](https://pythonhosted.org/inspyred/examples.html#the-knapsack-problem) described in Chapter 5 of the thesis.

**Decision Trees (DT)**: The library [Scikit-Learn](https://scikit-learn.org/) has been employed. Scikit-Learn implements a [CART decision tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier) algorithm that uses the Gini Index for measuring the goodness of an attribute split. In all experiments described in Chapter 4, we have employed the DT implementation of Scikit-Learn using the default parameters offered by the API.

**k-Means**: The [Scikit-Learn](https://scikit-learn.org/) implementation of [k-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) has been used in the experiments of Chapter 3 of the thesis.

**TOPSIS technique**: The library [Scikit-Criteria](https://scikit-criteria.readthedocs.io) has been employed. Scikit-Criteria implements a collection of multi-criteria decision analysis methods in Python. It offers an out-of-the-box implementation of the TOPSIS technique that has been used in the experiments provided in Chapter 5.

## Employed data sets
...

# License
...

# Citation
...

# Acknowledgments
...