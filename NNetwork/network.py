import numpy as np
import random
import sys
import json


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def vectorized_result(j):
    res = np.zeros((10, 1))
    res[j] = 1.0
    return res


class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a-y) ** 2

    @staticmethod
    def delta(a, y, z):
        return (a - y) * sigmoid_prime(z)


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
        self.biases, self.weights = None, None
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(x, y) / np.sqrt(y) for x, y in zip(self.sizes[1:], self.sizes[:-1])]

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(self.sizes[1:], self.sizes[:-1])]


    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda=0.0,
            validation_data=None, verbose=True,
            monitor_validation_cost=False, monitor_validation_accuracy=False,
            monitor_training_cost=False, monitor_training_accuracy=False):

        n = len(training_data)
        if validation_data:
            n_val = len(validation_data)

        # Metrics
        validation_cost, validation_accuracy = [], []
        training_cost, training_accuracy = [], []

        # train
        for j in range(epochs):
            np.random.shuffle(training_data)
            # optimize with numpy
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)
            if verbose and \
                    (monitor_training_cost or monitor_validation_cost or
                     monitor_training_accuracy or monitor_validation_accuracy):
                print(f'Epoch {j}: ', end='')
            if monitor_training_cost:
                cost = self.total_cost(training_data)
                training_cost.append(cost)
                if verbose:
                    print(f'Cost_train[{cost:.6f}]', end='..')
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                if verbose:
                    print(f'Acc_train[{accuracy:.6f}]', end='..')
            if monitor_validation_cost:
                cost = self.total_cost(validation_data)
                validation_cost.append(cost)
                if verbose:
                    print(f'Cost_val[{cost:.6f}]', end='..')
            if monitor_validation_accuracy:
                accuracy = self.accuracy(validation_data, convert=True)
                validation_accuracy.append(accuracy)
                if verbose:
                    print(f'Acc_val[{accuracy:.6f}]')
        return validation_cost, validation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        n_batch = len(mini_batch)
        x = np.asarray([_x.ravel() for _x, _y in mini_batch]).T
        y = np.asarray([_y.ravel() for _x, _y in mini_batch]).T
        nabla_b, nabla_w = self.backprop(x, y)

        # update params
        self.weights = [(1-eta*(lmbda/n)) * w - (eta/n_batch) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/n_batch) * nb for b, nb in zip(self.biases, nabla_b)]

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
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost.delta(activations[-1], y, zs[-1])
        nabla_b[-1] = delta.sum(1).reshape([len(delta), 1])
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta.sum(1).reshape([len(delta), 1])
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
        return nabla_b, nabla_w

    def total_cost(self, data, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        return cost

    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results) / len(data)

    def save(self, filename):
        data = {
            "sizes": list(self.sizes),
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.weights],
            "cost": str(self.cost.__name__)
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def load(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        cost = getattr(sys.modules[__name__], data['cost'])
        sizes = data['sizes']
        net = Network(sizes, cost=cost)
        net.weights = [np.array(w) for w in data['weights']]
        net.biases = [np.array(b) for b in data['biases']]
        return net

    @staticmethod
    def get_metrics(data, start=-10, end=0):
        val_cost, val_acc, train_cost, train_acc = data[0], data[1], data[2], data[3]
        n = len(val_cost)
        if abs(start) > n:
            start = -n
        print(f'\tMetrics [{n+start}:{n+end}]:')
        for i in range(start, end):
            print(f'Epoch {n+i}: ', end='')
            print(f'Cost_train[{train_cost[i]:.6f}]', end='..')
            print(f'Acc_train[{train_acc[i]:.6f}]', end='..')
            print(f'Cost_val[{val_cost[i]:.6f}]', end='..')
            print(f'Acc_val[{val_acc[i]:.6f}]')
