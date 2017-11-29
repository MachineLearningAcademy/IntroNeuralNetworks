import numpy as np

class NeuralNetwork():
    def __init__(self, layers=[2, 3, 2], output='linear'):
        self.weights = []
        self.output  = 'linear'

        # Create layer weight matrices
        for i in range(0, len(layers)-1):
            weight_matrix = np.random.random((layers[i], layers[i+1]))
            self.weights.append(weight_matrix)


    def fit(self, X, y, iterations=10, learning_rate=0.01):
        X = np.array(X, dtype='float64')
        y = np.array(y, dtype='float64')

        for i in range(iterations):
            derivative_matrixes = self._backprop(X, y)

            for l in range(len(self.weights)):
                self.weights[l] = self.weights[l] + learning_rate*derivative_matrixes[l]

    def predict(self, X):
        return self._forward(X)[-1]


    def _forward(self, X):
        forward_values = []

        forward_values.append(X)
        for weight_matrix in self.weights:
            values = np.dot(forward_values[-1], weight_matrix)
            forward_values.append(values)

        return forward_values


    def _backward(self, y_hat):
        backward_values = []

        backward_values.append(y_hat.T)
        for weight_matrix in reversed(self.weights):
            values = np.dot(weight_matrix, backward_values[-1])
            backward_values.append(values)

        backward_values = list(map(np.transpose, backward_values))
        return list(reversed(backward_values))


    def _loss(self, y, y_hat):
        return np.square(y - y_hat, axis=1)

    def _loss_derivative(self, y, y_hat):
        return 2*(y - y_hat)

    def _backprop(self, X, y):
        X = np.array(X, dtype='float64')
        y = np.array(y, dtype='float64')

        forward_values  = self._forward(X)

        y_hat = forward_values[-1]
        loss_derivative = self._loss_derivative(y, y_hat)

        backward_values = self._backward(loss_derivative)

        derivatives = []
        # Iterate over all layers
        n_layers = len(self.weights)
        for l in range(n_layers):
            weight_matrix = self.weights[l]

            derivative_matrix = np.zeros(weight_matrix.shape)

            # Iterate over all weights in the layer
            for i in range(weight_matrix.shape[0]):
                for j in range(weight_matrix.shape[1]):
                    weight_derivative = np.multiply(forward_values[l][:, i],backward_values[l+1][:, j])
                    derivative_matrix[i, j] = np.mean(weight_derivative)

            derivatives.append(derivative_matrix)

        return derivatives
