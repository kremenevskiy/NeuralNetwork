import time
from network import Network as Net
from mnist_loader import mnist_loader
import numpy as np
import network
from plotting import make_plots

training_data, validation_data, test_data = mnist_loader

net = Net([784, 30, 10], cost=network.CrossEntropyCost)
epochs = 3
data = \
    net.SGD(training_data=training_data, epochs=epochs, mini_batch_size=10, eta=0.5, lmbda=5,
            validation_data=validation_data, monitor_validation_cost=True, monitor_validation_accuracy=True,
            monitor_training_cost=True, monitor_training_accuracy=True)

make_plots(data, epochs, training_cost_xmin=0, validation_cost_xmin=0,
           train_accuracy_xmin=0, validation_accuracy_xmin=0, overlay_ylim=0.9)

# net.get_metrics(data)

# if need to save network weights
# net.save('neural.json')
