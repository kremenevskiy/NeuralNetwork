import json
import random
import sys

import matplotlib.pyplot as plt
import numpy as np


def plot_training_cost(training_cost, num_epochs, training_cost_xmin=0):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_cost_xmin, num_epochs), training_cost[training_cost_xmin:num_epochs], color='red')
    ax.set_xlim([training_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Cost')
    ax.set_title('Cost on the training data')
    plt.show()


def plot_val_cost(validation_cost, num_epochs, validation_cost_xmin=0):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(validation_cost_xmin, num_epochs), validation_cost[validation_cost_xmin:num_epochs], color='red')
    ax.set_xlim([validation_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Cost')
    ax.set_title('Cost on the validation data')
    plt.show()


def plot_training_accuracy(training_accuracy, num_epochs, training_accuracy_xmin=0):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_accuracy_xmin, num_epochs),
            training_accuracy[training_accuracy_xmin:num_epochs], color='red')
    ax.set_xlim([training_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy on train')
    ax.set_title('Accuracy on the training data')
    plt.show()


def plot_validation_accuracy(validation_accuracy, num_epochs, test_accuracy_xmin=0):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(test_accuracy_xmin, num_epochs),
            validation_accuracy[test_accuracy_xmin:num_epochs], color='red')
    ax.set_xlim([test_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy on validation')
    ax.set_title('Accuracy on the validation data')
    plt.show()


def plot_overlay(validation_accuracy, training_accuracy, num_epochs, ylim=0.9, xmin=0):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(xmin, num_epochs), training_accuracy, color='orange', label='Accuracy on the train data')
    ax.plot(np.arange(xmin, num_epochs), validation_accuracy, color='blue', label='Accuracy on the validation data')
    ax.grid(True)
    ax.set_xlim([xmin, num_epochs])
    ax.set_ylim([ylim, 1])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Compare accuracy on test and validation')
    plt.legend(loc="lower right")
    plt.show()


def make_plots(data, num_epochs, training_cost_xmin=0,
               train_accuracy_xmin=0,
               validation_cost_xmin=0,
               validation_accuracy_xmin=0,
               overlay_ylim=0.9):
    val_cost, val_acc, train_cost, train_acc = data[0], data[1], data[2], data[3]
    plot_training_cost(train_cost, num_epochs, training_cost_xmin)
    plot_training_accuracy(train_acc, num_epochs, train_accuracy_xmin)
    plot_val_cost(val_cost, num_epochs, validation_cost_xmin)
    plot_validation_accuracy(val_acc, num_epochs, validation_accuracy_xmin)
    plot_overlay(val_acc, train_acc, num_epochs, ylim=overlay_ylim)
