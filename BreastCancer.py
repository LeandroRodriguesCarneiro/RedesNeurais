import numpy as np
from sklearn import datasets

from multiLayerPerceptron import MultiLayerPerceptron


base = datasets.load_breast_cancer()
inputs = base.data
exits = base.target
expected_results = np.empty([len(exits),1], dtype=int)

for i in range(len(exits)):
    expected_results[i] = exits[i]

mlp = MultiLayerPerceptron(epochs=100000, learning_rate=0.1, momentum=1, hidden_neurons=15, inputs=inputs, expected_results=expected_results)
mlp.train()

print(f'Acuracia: {(mlp.accuracy() * 100):.2f}%')