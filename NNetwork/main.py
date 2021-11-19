import time
from network import Network as Net
from mnist_loader import data_mnist
import numpy as np


def vectorized_res(j):
    res = np.zeros((10, 1))
    res[j] = 1
    return res


train, test = data_mnist

x_train = [np.reshape(x / 255, (784, 1)) for x in train[0]]
y_train = [vectorized_res(j) for j in train[1]]
x_test = [np.reshape(x / 255, (784, 1)) for x in test[0]]
y_test = [vectorized_res(j) for j in test[1]]

training_data = list(zip(x_train, y_train))
test_data = list(zip(x_test, y_test))


# net = Net([784, 10])
# net.SGD(training_data=training_data, epochs=30, mini_batch_size=10, eta=1, test_data=test_data)

x = [0, 1, 1, 3, 1]
y = [0, 1, 5, 10, 1]
print(y)

# test = [(0, 1), (1, 1)]
# # print()
# # res = sum(int(x == y) for x, y, in test)
# # print(res)
# print(net.evaluate(test))


