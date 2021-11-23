import numpy as np
import random


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[:-1])]  # тут поменял

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j} - val: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")
            print(f"Epoch {j} - train: {self.evaluate(training_data)} / {n}\n")

    def update_mini_batch(self, mini_batch, eta):
        n = len(mini_batch)
        x = np.asarray([_x.ravel() for _x, _y in mini_batch]).T
        y = np.asarray([_y.ravel() for _x, _y in mini_batch]).T
        nabla_b, nabla_w = self.backprop(x, y)

        self.weights = [w - (eta / n) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / n) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [0 for b in self.biases]
        nabla_w = [0 for w in self.weights]

        # feed forward
        activation = x
        activations = [x]

        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        # backpropagation
        delta = self.cost_derivative_mse(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta.sum(1).reshape([len(delta), 1])
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = (np.dot(self.weights[-l + 1].T, delta)) * sp
            nabla_b[-l] = delta.sum(1).reshape([len(delta), 1])
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_result = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_result)

    @staticmethod
    def cost_derivative_mse(output_activations, y):
        return output_activations - y

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        return Network.sigmoid(z) * (1 - Network.sigmoid(z))