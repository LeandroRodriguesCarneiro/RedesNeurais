from sklearn.neural_network import MLPClassifier
from sklearn import datasets

iris = datasets.load_iris()
inputs = iris.data
outputs = iris.target

network = MLPClassifier(verbose=True, max_iter=100000, tol=0.00001, activation='logistic', learning_rate_init=0.000001)
network.fit(inputs, outputs)