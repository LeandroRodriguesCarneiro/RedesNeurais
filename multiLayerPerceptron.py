import numpy as np

class MultiLayerPerceptron:
    def __init__(self, epochs, learning_rate, momentum, hidden_neurons, inputs, expected_results, weights_0=None, weights_1=None):
        self.inputs = inputs
        self.expected_results = expected_results
        input_neurons = inputs.shape[1]  # Número de neurônios na camada de entrada

        if weights_0 is not None:
            self.weights_0 = weights_0
        else:
            self.weights_0 = 2 * np.random.rand(input_neurons, hidden_neurons) - 1
        if weights_1 is not None:
            self.weights_1 = weights_1
        else:
            self.weights_1 = 2 * np.random.rand(hidden_neurons, 1) - 1
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum



    def sum_values(self, inputs, weights):
        return np.dot(inputs, weights)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, x):
        return x * (1 - x)

    def mse(self, predicted, target):
        return np.mean((predicted - target) ** 2)

    def rmse(self, predicted, target):
        return np.sqrt(self.mse(predicted, target))

    def accuracy(self):
        return 1 - self.error

    def train(self):
        for _ in range(self.epochs):
            hidden_layer = self.sigmoid(self.sum_values(self.inputs, self.weights_0))
            output_layer = self.sigmoid(self.sum_values(hidden_layer, self.weights_1))

            output_error = self.expected_results - output_layer
            #self.error = self.rmse(output_layer, self.expected_results) #calculo de erro rmse
            #self.error = self.mse(output_layer, self.expected_results) #calculo de mse
            self.error = np.mean(np.abs(output_error)) #calculo de media absoluta
            print(f'Erro: {self.error}')

            d_output_layer = self.d_sigmoid(output_layer)
            delta_output = output_error * d_output_layer

            delta_output_x_weight = delta_output.dot(self.weights_1.T)
            delta_hidden_layer = delta_output_x_weight * self.d_sigmoid(hidden_layer)

            new_weights_1 = hidden_layer.T.dot(delta_output)
            self.weights_1 = (self.weights_1 * self.momentum) + (new_weights_1 * self.learning_rate)

            new_weights_0 = self.inputs.T.dot(delta_hidden_layer)
            self.weights_0 = (self.weights_0 * self.momentum) + (new_weights_0 * self.learning_rate)

    def predict(self, inputs):
        hidden_layer = self.sigmoid(self.sum_values(inputs, self.weights_0))
        output_layer = self.sigmoid(self.sum_values(hidden_layer, self.weights_1))
        return output_layer
    
# Definindo os dados de entrada, resultados esperados e pesos iniciais
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_results = np.array([[0], [1], [1], [0]])
weights_0 = np.array([[-0.424, -0.740, -0.961], [0.358, -0.577, -0.469]])
weights_1 = np.array([[-0.017], [-0.893], [0.148]])

# Criando o modelo MLP
mlp = MultiLayerPerceptron(inputs=inputs, expected_results=expected_results, hidden_neurons=3, epochs=1000000, learning_rate=0.6, momentum=1)

# Treinando o modelo
mlp.train()

print(f'Acuracia: {(mlp.accuracy() * 100):.2f}%')
print(f'Pesos Primeira Camada: \n{mlp.weights_0}')
print(f'Pesos Camada Oculta: \n{mlp.weights_1}')

# Realizando previsões para as entradas fornecidas
for i in range(inputs.shape[0]):
    print(f'XOR {inputs[i]}: {mlp.predict(inputs[i])}')