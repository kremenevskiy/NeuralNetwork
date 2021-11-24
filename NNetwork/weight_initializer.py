import json
import random

from mnist_loader import mnist_loader
import network

import matplotlib.pyplot as plt
import numpy as np


def main(filename='weights_init.json', n=100, eta=0.1):
    run_network(filename, n, eta)
    make_plot(filename)


def run_network(filename, n, eta):
    random.seed(12345678)
    np.random.seed(12345678)

    training_data, test_data = mnist_loader
    net = network.Network([784, n, 10], cost=network.CrossEntropyCost)
    print('Train network using default starting weights')
    def_val_cost, def_val_acc, def_train_cost, def_train_acc = net.SGD(training_data, 30, 10, eta, lmbda=5.0,
                                                                       verbose=False,
                                                                       validation_data=test_data,
                                                                       monitor_validation_accuracy=True)
    print('Train network using large starting weights')
    net.large_weight_initializer()
    large_val_cost, large_val_acc, large_train_cost, large_train_acc = net.SGD(training_data, 30, 10, eta, lmbda=5.0,
                                                                               verbose=False,
                                                                               validation_data=test_data,
                                                                               monitor_validation_accuracy=True)
    with open(filename, 'w') as f:
        data = {'default_weight_initializer': [def_val_cost, def_val_acc, def_train_cost, def_train_acc],
                'large_weight_initializer': [large_val_cost, large_val_acc, large_train_cost, large_train_acc]}

        json.dump(data, f)


def make_plot(filename):
    with open(filename, 'r') as f:
        results = json.load(f)

    def_val_cost, def_val_acc, def_train_cost, def_train_acc = results['default_weight_initializer']
    large_val_cost, large_val_acc, large_train_cost, large_train_acc = results['large_weight_initializer']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0, 30, 1), large_val_acc, color='blue', label='Large weights init')
    ax.plot(np.arange(0, 30, 1), def_val_acc, color='green', label='Optimized weights init')
    ax.set_xlim([0, 30])
    ax.set_xlabel('Epoch')
    ax.set_ylim([0.85, 1])
    ax.set_title('Classification accuracy')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    main()
