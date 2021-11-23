import numpy as np
import random


class Activations(object):
    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        return Activations.sigmoid(z) * (1 - Activations.sigmoid(z))


class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a-y) ** 2

    @staticmethod
    def delta(a, y, z):
        return a-y * Activations.sigmoid_prime(z)


class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1-y) * np.log(1 - a)))

    @staticmethod
    def delta(a, y, z=0):
        return a - y


class Network(object):
    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[:-1])]  # тут поменял
        self.cost = cost

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = Activations.sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, validation_data=None, verbose=False,
            monitor_validation_cost=False, monitor_validation_accuracy=False,
            monitor_training_cost=False, monitor_training_accuracy=False):

        n = len(training_data)

        if validation_data:
            n_test = len(validation_data)

        validation_cost, validation_accuracy = [], []
        training_cost, training_accuracy = [], []

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            print(f'Epoch {j}: ', end='')
            if monitor_training_cost:
                cost = self.total_cost(training_data)
                training_cost.append(cost)
                print(f'Cost_train[{cost:.6f}]', end='..')
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data)
                training_accuracy.append(accuracy)
                print(f'Acc_train[{accuracy:.6f}]', end='..')
            if monitor_validation_cost:
                cost = self.total_cost(validation_data)
                validation_cost.append(cost)
                print(f'Cost_val[{cost:.6f}]', end='..')
            if monitor_validation_accuracy:
                accuracy = self.accuracy(validation_data)
                validation_accuracy.append(accuracy)
                print(f'Acc_val[{accuracy:.6f}]')

        return validation_cost, validation_accuracy, training_cost, training_accuracy

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
            activation = Activations.sigmoid(z)
            activations.append(activation)

        # backpropagation
        delta = self.cost.delta(activations[-1], y, zs[-1])
        nabla_b[-1] = delta.sum(1).reshape([len(delta), 1])
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = Activations.sigmoid_prime(z)
            delta = (np.dot(self.weights[-l + 1].T, delta)) * sp
            nabla_b[-l] = delta.sum(1).reshape([len(delta), 1])
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)
        return nabla_b, nabla_w

    def total_cost(self, data, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            cost += self.cost.fn(a, y) / len(data)
        return cost

    def accuracy(self, data):
        results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results) / len(data)

    def evaluate(self, test_data):
        test_result = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_result)
