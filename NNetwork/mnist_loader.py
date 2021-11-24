import numpy as np
from keras.datasets import mnist
data_mnist = mnist.load_data()


def vectorized_result(j):
    res = np.zeros((10, 1))
    res[j] = 1
    return res


train, test = data_mnist

x_train = [np.reshape(x / 255, (784, 1)) for x in train[0]]
y_train = [vectorized_result(j) for j in train[1]]
x_test = [np.reshape(x / 255, (784, 1)) for x in test[0]]
y_test = [j for j in test[1]]

training_data = list(zip(x_train, y_train))
test_data = list(zip(x_test, y_test))

mnist_loader = (training_data, test_data)

