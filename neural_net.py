import numpy as np
from math import exp as e
from math import sqrt
from random import randint
import csv

# Let there be L layers in total
# when layer 1 is the input layer and layer L is the output layer
# so there are L-2 hidden layers

hidden_layer_sizes = [10, 10] # number of neurons for each of the hidden layers
step_size = 0.005
batch_size = 1000 # however many data points we train on before testing validation
threshold = 0.005

L = len(hidden_layer_sizes) + 2 # total number of layers

def format_y_data(data):
    y_data = np.zeros((len(data), K))
    for i in range(len(data)):
        index = max(0, int(data[i][D]))
        y_data[i][index] = 1 # y_data has to be formatted specially
    return y_data

def load_basic_data():
    data = np.loadtxt('data/data_3class.csv')
    global D, K
    D = len(data[0]) - 1
    K = int(np.amax(data[:,D])) + 1
    x_data = data[:,0:D]
    y_data = format_y_data(data)
    return x_data[:400], y_data[:400], x_data[400:600], y_data[400:600], x_data[600:], y_data[600:]

def load_2d_data(number):
    train_data = np.loadtxt('data/data' + str(number) + '_train.csv')
    val_data = np.loadtxt('data/data' + str(number) + '_validate.csv')
    test_data = np.loadtxt('data/data' + str(number) + '_test.csv')
    global D, K
    D = len(train_data[0]) - 1
    K = int(np.amax(train_data[:,D])) + 1
    return train_data[:,0:D], format_y_data(train_data), val_data[:,0:D], format_y_data(val_data), test_data[:,0:D], format_y_data(test_data)

# For the training set, use the first 200 samples per class
# For the validation set, use the next 150 samples
# For the test set, use the next 150 samples
def load_mnist_data():
    num_train, num_val, num_test = 200, 150, 150
    train_data, val_data, test_data = [], [], []
    for i in range(10):
        with open('data/mnist_digit_' + str(i) + '.csv', 'r') as fin:
            reader=csv.reader(fin)
            data = []
            rowNum = 0
            for row in reader:
                data.append([float(s) * 2/255 - 1 for s in row[0].split()] + [i])
                rowNum += 1
                if (rowNum >= num_train + num_val + num_test):
                    break
        train_data.extend(data[:num_train])
        val_data.extend(data[num_train:num_train + num_val])
        test_data.extend(data[num_train + num_val:num_train + num_val + num_train])
    train_data = np.array(train_data)
    val_data = np.array(val_data)
    test_data = np.array(test_data)
    global D,K
    D = len(train_data[0]) - 1
    K = int(np.amax(train_data[:,D])) + 1
    return train_data[:,0:D], format_y_data(train_data), val_data[:,0:D], format_y_data(val_data), test_data[:,0:D], format_y_data(test_data)

def relu(z):
    return np.array([max(0,x) for x in z])
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
def relu_derivative(z):
    return np.array([(1 if x > 0 else 0) for x in z])

def test_accuracy(x_val, y_val, w, b):
    # These don't really matter, just need placeholders
    z = [0] * (L+1)
    a = [0] * (L+1)

    correct_count = 0
    total_count = 0
    for i in range(len(x_val)):
        # Input - put the input data into the first d elements of a[1]
        a[1] = x_val[i]

        # Feedforward
        for l in range(2, L+1):
            z[l] = w[l].T.dot(a[l-1]) + b[l]
            if l < L:
                a[l] = relu(z[l])
            else:
                a[l] = softmax(z[l])

        # Testing
        predicted_class = np.argmax(a[L])
        actual_class = np.argmax(y_val[i])
        if predicted_class == actual_class:
            correct_count += 1
        total_count += 1
    return 1. * correct_count / total_count

def train_model(x_train, y_train, x_val, y_val):
    # Randomly initialize data
    # the 0th entry of any of these are never used, I just did this for consistency with the lecture notes
    layer_sizes = [1, D] + hidden_layer_sizes + [K]
    z = [0] * (L+1)
    a = [0] * (L+1)
    delta = [0] * (L+1)
    b = [0]
    w = [0]
    for l in range(1, L+1):
        std_dev = 1 / sqrt(layer_sizes[l-1])
        b.append(np.zeros(layer_sizes[l]))
        w.append(np.random.normal(0, std_dev, (layer_sizes[l-1], layer_sizes[l])))

    # Array of validation accuracies
    validation_accuracies = [0] * 10

    # Actually train it
    while (True):
        for i in range(batch_size):
            # To be truly stochastic, randomly choose an index from the training set
            i = randint(0, len(x_train) - 1)
            # Input
            a[1] = x_train[i]

            # Feedforward
            for l in range(2, L+1):
                z[l] = w[l].T.dot(a[l-1]) + b[l]
                if l < L:
                    a[l] = relu(z[l])
                else:
                    a[l] = softmax(z[l])

            # Output error - based on https://piazza.com/class/isc0gszez6g165?cid=743
            delta[L] = a[L] - y_train[i]

            # Backpropogation
            for l in range(L-1, 1, -1):
                delta[l] = np.diag(relu_derivative(z[l])).dot(w[l+1]).dot(delta[l+1])

            # Gradient
            for l in range(2, L+1):
                w[l] = w[l] - step_size * np.outer(a[l-1], delta[l])
                b[l] = b[l] - step_size * delta[l]

        # Test on validation set
        # Stop the training once the current accuracy is super close to the average of the past ten's accuracies
        accuracy = test_accuracy(x_val, y_val, w, b)
        validation_accuracies.append(accuracy)
        validation_accuracies.pop(0)
        average = sum(validation_accuracies) / len(validation_accuracies)
        if abs(accuracy - average) < threshold:
            print "Final validation accuracy average is ", average
            return w, b
        print accuracy

if __name__ == '__main__':
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist_data()
    w, b = train_model(x_train, y_train, x_val, y_val)
    print "Final test accuracy is", test_accuracy(x_test, y_test, w, b)
