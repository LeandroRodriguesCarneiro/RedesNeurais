import numpy as np

class Perceptron:

    def __init__(self, array_input, array_output, array_weights=None):
        self.inputs_matrix = array_input
        self.weights_matrix = array_weights if array_weights is not None else np.random.rand(2)
        self.expected_results = array_output
        self.learning_rate = 0.1

    def sumValues(self):
        return self.inputs_matrix.dot(self.weights_matrix)

    def stepFunction(self, sum):
        return np.where(sum >= 1, 1, 0)

    def CalcExit(self, inputs):
        return self.stepFunction(self.sumValues()) if inputs is None else self.stepFunction(inputs.dot(self.weights_matrix))

    def learning(self):
        total_error = 1
        while total_error != 0:
            total_error = 0
            for i in range(len(self.expected_results)):
                exitCalc = self.CalcExit(self.inputs_matrix[i])  
                error = abs(self.expected_results[i] - exitCalc)  
                total_error += error
                for j in range(len(self.weights_matrix)):
                    self.weights_matrix[j] = self.weights_matrix[j] + (self.learning_rate * self.inputs_matrix[i][j] * error)
                    print('Peso atualizado: '+str(self.weights_matrix[j]))
            print('Total de erros: '+str(total_error))
        return 0

# Teste do algoritmo de aprendizado
perceptronE = Perceptron(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 0, 1]))
perceptronE.learning()
perceptronOU = Perceptron(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 1]))
perceptronOU.learning()

# Previsão para um novo conjunto de dados
#inputs_matrix = np.random.randint(2, size=(4, 2))
inputs_matrix = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for i in range(inputs_matrix.shape[0]):
    print(f'E {inputs_matrix[i]}: {perceptronE.CalcExit(inputs_matrix[i])}')
    print(f'OU {inputs_matrix[i]}: {perceptronOU.CalcExit(inputs_matrix[i])}')
