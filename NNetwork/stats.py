import random
import json
import numpy as np
import matplotlib.pyplot as plt
from mnist_loader import mnist_loader
import network


SIZES = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]


def run_networks():
    random.seed(12345678)
    np.random.seed(12345678)
    train_data, validation_data = mnist_loader
    net = network.Network([784, 30, 10], cost=network.CrossEntropyCost)
    accuracies = []
    for size in SIZES:
        print(f'Train network with data set size {size}')
        num_epochs = 1500000 // size
        net.SGD(train_data[:size], num_epochs, 10, 0.5, lmbda=size*0.0001, verbose=False)
        accuracy = net.accuracy(validation_data)
        print(f'\tAccuracy: {accuracy:.4f}')
        accuracies.append(accuracy)
    with open('stats.json', 'w') as f:
        json.dump(accuracies, f)


def make_linear_plot(accuracies):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(SIZES, accuracies, color='orange')
    ax.plot(SIZES, accuracies, "o", color='blue')
    ax.set_xlim(0, 50000)
    ax.set_ylim(0.6, 1)
    ax.grid(True)
    ax.set_xlabel('Training set size')
    ax.set_title('Accuracy on the validation data')
    plt.show()


def make_log_plot(accuracies):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(SIZES, accuracies, color='blue')
    ax.plot(SIZES, accuracies, "o", color='orange')
    ax.set_xlim(100, 50000)
    ax.set_ylim(0.6, 1)
    ax.set_xscale('log')
    ax.grid(True)
    ax.set_xlabel('Training set size')
    ax.set_title('Log Accuracy on the validation data')
    plt.show()


def make_plots():
    with open("stats.json", "r") as f:
        accuracies = json.load(f)

    make_linear_plot(accuracies)
    make_log_plot(accuracies)


def main():
    run_networks()
    make_plots()


if __name__ == '__main__':
    main()
