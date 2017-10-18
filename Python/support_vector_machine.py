###############################################################################
# Support Vector Machine                                                      #
# Glenn Dawson                                                                #
# 2017-10-18                                                                  #
###############################################################################

#############
#  Imports  #
#############

import numpy as np
from numpy.random import multivariate_normal as mvnrnd
import matplotlib.pyplot as plt
import time
from quadprog import solve_qp
from scipy.spatial.distance import pdist, squareform

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
        labels1 = np.ones((1, data1.shape[0]))
        labels2 = np.full((1, data2.shape[0]), -1)
        labels = np.ravel(np.hstack((labels1, labels2)))

        # Extract dimensions
        self.classes = np.unique(labels)
        self.num_classes = len(self.classes)
        self.num_instances = data.shape[0]
        self.num_features = data.shape[1]

        # Add x_0 = 1 to data
        data = np.insert(data, 0, 1, axis=1)

        # Margin coefficient
        C = 0.5

        # Kernel definitions
        kernel = 'gaussian'
        if kernel == 'gaussian':
            sigma = 1
            squared_dists = squareform(pdist(data, 'sqeuclidean'))
            K = np.exp(-squared_dists / (2 * sigma ** 2))
        elif kernel == 'polynomial':
            d = 3
            gamma = 1
            K = (1 + gamma * np.inner(data, data.T)) ** d

        # quadprog parameters to solve for alpha
        H = np.outer(labels, labels.T) * K
        f = np.ravel(np.full((1, self.num_instances), -1)).astype(np.float)
        A = np.zeros((self.num_instances, self.num_instances))
        A = A.astype(np.float)
        b = np.ravel(np.zeros((1, self.num_instances))).astype(np.float)
        Aeq = labels.astype(np.float)
        beq = np.ravel(np.zeros((1, self.num_instances))).astype(np.float)
        lb = np.ravel(np.zeros((1, self.num_instances))).astype(np.float)
        ub = np.ravel(np.full((1, self.num_instances), C)).astype(np.float)

        alpha = self.quadprog_solve_qp(H, f, A, b, Aeq, beq, lb, ub)

        self.w = np.dot(data.T, alpha * labels)

        print('The support vector has been obtained.')
        print('Training time:', time.time() - training_start, 'seconds.')

    def quadprog_solve_qp(self, P, q, G=None, h=None, A=None, b=None,
                          lb=None, ub=None):
        # Author: Stephane Caron
        qp_G = .5 * (P + P.T) # make sure P is symmetric
        qp_a = -q
        if A is not None:
            if lb is not None:
                if ub is not None:
                    qp_C = np.vstack([G, -np.eye(self.num_instances),
                                      np.eye(self.num_instances)])
                    qp_C = np.transpose(qp_C).astype(np.float)
                    qp_b = np.hstack([h, -lb, ub]).astype(np.float)
                    meq = 0
            else:
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
        w = self.w[:self.num_features + 1]
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
    class_1_instances = 100
    class_2_instances = 100
    Q1 = mvnrnd(mu1, sigma1, class_1_instances)
    Q2 = mvnrnd(mu2, sigma2, class_2_instances)

    # Train SVM
    print('Training support vector machine...')
    svm = SupportVectorMachine(Q1, Q2)

    # Generate test data
    print('Generating test data...')
    mu = [2, 2]
    sigma = [[0.7, 0], [0, 0.7]]
    test_instances = 10000
    Q = mvnrnd(mu, sigma, test_instances)

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
    w = svm.w[:svm.num_features + 1]
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