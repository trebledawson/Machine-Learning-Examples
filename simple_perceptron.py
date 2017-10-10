###############################################################################
# Simple Perceptron                                                           #
# Glenn Dawson                                                                #
# 2017-10-10                                                                  #
###############################################################################

#####################
#  List of Imports  #
#####################
import numpy as np
from numpy.random import uniform as uniform
from numpy.random import multivariate_normal as mvnrnd
from math import sqrt as sqrt
import matplotlib.pyplot as plt
import time

##########
#  Main  #
##########

def main():
    start = time.time()
    print('Program start.')

    # Generate training data
    print('Generating training data...')
    cases = 1000
    mu1 = [1.6, 1]
    mu2 = [3.3, 2.7]
    mu3 = [1, 3]
    sigma1 = [[0.1, 0], [0, 0.1]]
    sigma2 = [[0.1, 0], [0, 0.1]]
    sigma3 = [[0.1, 0], [0, 0.1]]
    Q1 = mvnrnd(mu1, sigma1, cases)
    Q2 = mvnrnd(mu2, sigma2, cases)
    Q3 = mvnrnd(mu3, sigma3, cases)
    Q = np.vstack((Q1, Q2, Q3))
    Y = np.vstack(
        (np.zeros((cases, 1)), np.ones((cases, 1)), np.full((cases, 1), 2)))

    # Train perceptron classifier
    print('Training perceptron classifier...')
    sp = SimplePerceptron(Q, Y)

    # Classification test
    print('Generating test data...')
    mu_test = [2.1, 2]
    sigma_test = [[0.7, 0], [0, 0.7]]
    Q_test = mvnrnd(mu_test, sigma_test, 10000)

    print('Evaluating test data using perceptron model...')
    P = sp.predict(Q_test)

    # Plot training data and classifications of test data
    print('Plotting data...')
    plt.subplot(121) # Training data
    plt.plot(Q1[:, 0], Q1[:, 1], 'b.')
    plt.plot(Q2[:, 0], Q2[:, 1], 'r.')
    plt.plot(Q3[:, 0], Q3[:, 1], 'g.')
    plt.title('Plot of randomly generated data')
    plt.xlim((-0.5, 4.5))
    plt.ylim((-0.5, 4.5))

    plt.subplot(122) # Test data
    plt.plot(Q_test[P == 0, 0], Q_test[P == 0, 1], 'b.')
    plt.plot(Q_test[P == 1, 0], Q_test[P == 1, 1], 'r.')
    plt.plot(Q_test[P == 2, 0], Q_test[P == 2, 1], 'g.')
    plt.title('Simple perceptron classification of data')
    plt.xlim((-0.5,4.5))
    plt.ylim((-0.5,4.5))

    print('Time elapsed:', time.time()-start, 'seconds.')
    plt.show()

##################################
#  Simple Perceptron Classifier  #
##################################

class SimplePerceptron:

    def __init__(self, train, labels):
        np.array(train)
        np.array(labels)

        # Extract number of classes and features
        self.classes = np.unique(labels)
        self.num_classes = len(self.classes)
        self.num_features = train.shape[1]

        # Sort instances by classes
        self.weights = np.ndarray((self.num_classes, self.num_features + 1))
        for i in range(self.num_classes):
            class_data = train[np.nonzero(labels == i)[0]]
            nonclass_data = train[np.nonzero(labels != i)[0]]
            weight = self.gradient_descent(class_data, nonclass_data)
            self.weights[i] = weight


        print('The perceptron was successfully trained.')

    def gradient_descent(self, class_data, nonclass_data, eta = 0.03):
        # Create single array of data and set x_0 to 1
        data = np.vstack((class_data, nonclass_data))
        data = np.hstack((np.ones((data.shape[0], 1)), data))

        # Create single array of data labels (-1,1)
        labels = np.append(np.ones((1, class_data.shape[0])),
                           np.full((1, nonclass_data.shape[0]), -1))

        # Initialize weights by [-1/sqrt(d),1/sqrt(d)]
        w = uniform(-1/sqrt(self.num_features), 1/sqrt(self.num_features),
                                                      self.num_features + 1)

        # Gradient descent algorithm
        count = 0
        dW = w
        while (np.linalg.norm(dW) > 0.1):
            miss_test = np.inner(w, data) * labels
            miss_indices = miss_test < 0
            dJ = data[miss_indices] * labels[miss_indices].reshape(-1, 1)
            sum = np.sum(dJ, axis=0)
            dW = eta * sum
            w = w + dW
            count += 1

        return w

    def node(self, data, weights):
        return np.inner(weights,data)

    def predict(self, test):
        np.array(test)
        classifications = []
        for instance in test:
            node_outputs = self.node(np.hstack((1, instance)), self.weights)
            selection = np.argmax(node_outputs)
            classifications.append(selection)

        return np.array(classifications)

if __name__ == '__main__':
    main()