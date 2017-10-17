###############################################################################
# Support Vector Machine                                                      #
# Glenn Dawson                                                                #
# 2017-10-10                                                                  #
###############################################################################

#############
#  Imports  #
#############

import numpy as np
from numpy.random import multivariate_normal as mvnrnd
import matplotlib.pyplot as plt
import time
from quadprog import solve_qp

##########
#  Main  #
##########

def main():
    test_2d()

######################################
#  Support Vector Machine Classifier #
######################################
class SupportVectorMachine:
    def __init__(self, data1, data2):
        training_start = time.time()
        np.array(data1)
        np.array(data2)
        data = np.vstack((data1, data2))
        labels1 = np.ones((data1.shape[0], 1))
        labels2 = np.full((data2.shape[0], 1), -1)
        labels = np.vstack((labels1, labels2))

        # Extract dimensions
        self.classes = np.unique(labels)
        self.num_classes = len(self.classes)
        self.num_instances = data.shape[0]
        self.num_features = data.shape[1]

        # Add x_0 = 1 to data
        data = np.insert(data, 0, 1, axis=1)

        # quadprog parameters
        H = np.eye(self.num_features + 1)
        H[0][0] = 0.00001
        f = np.ravel(np.zeros((1, self.num_features + 1)))
        A = labels * data
        b = np.ravel(np.full((1, self.num_instances), -1))

        self.params = self.quadprog_solve_qp(H, f, A, b)

        print('The support vector has been obtained.')
        print('Training time:', time.time() - training_start, 'seconds.')

    def quadprog_solve_qp(self, P, q, G=None, h=None, A=None, b=None):
        # Author: Stephane Caron
        qp_G = .5 * (P + P.T) # make sure P is symmetric
        qp_a = -q
        if A is not None:
            qp_C = -np.vstack([A, G]).T
            qp_b = -np.hstack([b, h])
            meq = A.shape[0]
        else:  # no equality constraint
            qp_C = -G.T
            qp_b = (-h).astype(np.float)
            meq = 0

        return solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

    def predict(self, test_data):
        print('Classifying data...')
        data = np.insert(np.array(test_data), 0, 1, axis=1)
        w = self.params[:self.num_features + 1]
        classification = np.inner(w, data)

        return np.greater(classification, 0)


##################
#  Test scripts  #
##################

def test_2d():
    print('Program start.')
    start = time.time()

    # Generate training data
    print('Generating training data...')
    mu1 = [0.7, 1]
    mu2 = [3.7, 3.3]
    sigma1 = [[0.2, 0], [0, 0.3]]
    sigma2 = [[0.2, 0], [0, 0.3]]
    Q1 = mvnrnd(mu1, sigma1, 1000)
    Q2 = mvnrnd(mu2, sigma2, 1000)

    # Train SVM
    print('Training support vector machine...')
    svm = SupportVectorMachine(Q1, Q2)

    # Generate test data
    print('Generating test data...')
    mu = [2, 2]
    sigma = [[0.7, 0], [0, 0.7]]
    Q = mvnrnd(mu, sigma, 10000)

    # Classify test data
    print('Classifying test data...')
    P = svm.predict(Q)

    # Plot data
    print('Plotting data...')
    plt.subplot(121)
    plt.title('Randomly generated data')
    plt.plot(Q1[:, 0], Q1[:, 1], 'b.')
    plt.plot(Q2[:, 0], Q2[:, 1], 'r.')

    X = np.linspace(0, 4, 500)
    w = svm.params[:svm.num_features + 1]
    plt.plot(X,(-(w[1]*X+w[0]))/(w[2]))
    plt.xlim(xmin=-1, xmax=5)
    plt.ylim(ymin=-1, ymax=5)

    plt.subplot(122)
    plt.title('Classified test data')
    plt.plot(Q[P == 0, 0], Q[P == 0, 1], 'b.')
    plt.plot(Q[P == 1, 0], Q[P == 1, 1], 'r.')
    plt.xlim(xmin=-1, xmax=5)
    plt.ylim(ymin=-1, ymax=5)

    print('Total elapsed time is', time.time() - start, 'seconds.')
    plt.show()


if __name__ == '__main__':
    main()