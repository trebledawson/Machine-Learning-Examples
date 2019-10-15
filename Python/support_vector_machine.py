###############################################################################
# Support Vector Machine                                                      #
# Glenn Dawson                                                                #
# 2019-10-10                                                                  #
###############################################################################

import numpy as np
from numpy.random import multivariate_normal as mvnrand
import qpsolvers
import matplotlib.pyplot as plt
import time

C = 10
KERNEL = 'gaussian'
SIGMA = 1
GAMMA = 2
POLY_ORDER = 2
INTERCEPT = 3
N_SAMPLES = 100

def main():
    synthetic_test()


def synthetic_test():
    print('Program start.')
    start = time.time()

    # Generate training data
    print('Generating training data...')
    mu1 = [1, 1]
    mu2 = [2, 2]
    sigma1 = [[0.15, 0], [0, 0.15]]
    sigma2 = [[0.15, 0], [0, 0.15]]
    data1 = mvnrand(mu1, sigma1, N_SAMPLES)
    data2 = mvnrand(mu2, sigma2, N_SAMPLES)

    # Train SVM
    print('Training support vector machine...')
    SVM = SupportVectorMachine(data1, data2, C=C, kernel=KERNEL)

    # Plot training data and decision boundary
    fig, ax = plt.subplots()
    SVM.plot_decision_boundary(ax)
    plt.xlim(left=0, right=3)
    plt.ylim(bottom=0, top=3)
    plt.grid()

    # Generate test data
    print('Generating test data...')
    mu = [1.4, 1.4]
    sigma = [[0.7, 0], [0, 0.7]]
    test_instances = 10000
    qtest = mvnrand(mu, sigma, test_instances)

    # Classify test data
    print('Classifying test data...')
    pred = SVM.predict(qtest)

    plt.figure()
    plt.title('Classified test data')
    plt.plot(qtest[pred == 0, 0], qtest[pred == 0, 1], 'y.')
    plt.plot(qtest[pred == 1, 0], qtest[pred == 1, 1], 'm.')
    plt.xlim(left=0, right=3)
    plt.ylim(bottom=0, top=3)
    plt.grid()

    print('Total elapsed time is', time.time() - start, 'seconds.')
    plt.show()


class SupportVectorMachine:
    def __init__(self, data1, data2, C=1000, kernel='gaussian', b=0.0, tol=0.01,
                 epsilon=0.01):
        self.data = np.vstack((data1, data2))
        self.labels = np.ravel(np.hstack((np.ones((1, data1.shape[0])),
                                          -np.ones((1, data2.shape[0])))))
        self.C = C
        self.kernel = kernel
        self.n_samples = self.data.shape[0]
        self.alphas = np.zeros(self.n_samples)
        self.b = b
        self.objective = []
        self.error_tol = tol
        self.epsilon = epsilon

        # Initialize errors
        self.errors = self.decision_func(self.alphas, self.labels,
                                         self.kernel, self.data,
                                         self.data, self.b) - self.labels

        # Train
        self.train()
        self.weights = np.sum(self.alphas * self.labels * self.data.T, axis=0)

    def decision_func(self, alphas, target, kernel, X_train, X_test, b):
        """Applies the SVM decision function to the input feature vectors in
        X_test.

        Parameters
        ----------
        alphas : Array-like, shape = ()
            alphas

        target : Array-like, shape = ()
            labels

        kernel : string
            SVM kernel function.

        X_train : Array-like, shape = ()
            Training data.

        X_test : array-like, shape = ()
            Testing data.

        b : float
            b

        Returns
        -------
        decisions : Array-like, shape = ()
            decisions
        """
        k = self.kernel_func(X_train, X_test, kernel=kernel)
        return (alphas * target) @ k - b

    @staticmethod
    def kernel_func(data_a, data_b, kernel='gaussian'):
        """Applies the specified kernel function to the data matrices a and b.

        Parameters
        ----------
        data_a : Array-like, shape = ()
            First data array.

        data_b : Array-like, shape = ()
            Second data array.

        kernel: string
            Kernel specification. Must be 'gaussian', 'polynomial', or 'inner'.
        """
        if kernel == 'gaussian':
            sigma = SIGMA
            if np.ndim(data_a) == 1 and np.ndim(data_b) == 1:
                norm = np.linalg.norm(data_a - data_b)
                return np.exp(- norm / (2 * sigma ** 2))
            elif (np.ndim(data_a) > 1 and np.ndim(data_b) == 1) or (
                    np.ndim(data_a) == 1 and np.ndim(data_b) > 1):
                norm = np.linalg.norm(data_a - data_b, axis=1)
                return np.exp(- norm / (2 * sigma ** 2))
            elif np.ndim(data_a) > 1 and np.ndim(data_b) > 1:
                norm = np.linalg.norm(
                    data_a[:, np.newaxis] - data_b[np.newaxis, :], axis=2)
                return np.exp(- norm / (2 * sigma ** 2))

        elif kernel == 'polynomial':
            gamma = GAMMA
            d = POLY_ORDER
            return (1 + gamma * (data_a @ data_b.T)) ** d

        elif kernel == 'inner':
            b = INTERCEPT
            return data_a @ data_b.T + b

    def train(self):
        """Produces the alpha weights for SVM.
        
        P : yTy * xTx, shape = (n_samples, n_samples)
        q : array of -1 for negative summation, shape = (n_samples,)
        G : negative eye matrix for 1st inequality constraint, eye matrix for 2nd inequality constraint, shape = (2 * n_samples, n_samples)
        h : array of 0 for 1st inequality constraint, C for 2nd inequality constraint, shape = (2 * n_samples,)
        A : sparse matrix with first row the labels for equality constraint, shape = (n_samples, n_samples)
        b : array of 0 for equality constraint, shape = (n_samples,)
        """
        q = -np.ones(self.n_samples)
        G = np.vstack((-np.eye(self.n_samples), np.eye(self.n_samples)))
        h = np.concatenate((np.zeros(self.n_samples),
                            self.C * np.ones(self.n_samples)))
        b = np.zeros(self.n_samples)
        A = np.zeros((self.n_samples, self.n_samples))
        A[0] = self.labels
        P = np.multiply(np.outer(self.labels, self.labels),
                        self.kernel_func(self.data, self.data,
                                         kernel=self.kernel))
        P = P + 1e-8 * np.eye(self.n_samples)  # For stability
        self.alphas = qpsolvers.solve_qp(P, q, G, h, A, b)

    def plot_decision_boundary(self, ax, resolution=100,
                               colors=('b', 'k', 'r')):
        """
        Plots the decision boundary on the input axes object.
        Range of decision boundary grid is determined by the training data.
        Returns decision boundary grid and axes object (`grid`, `ax`).
        """

        # Generate coordinate grid of shape [resolution x resolution]
        # and evaluate over the entire space
        xrange = np.linspace(self.data[:, 0].min(), self.data[:, 0].max(),
                             resolution)
        yrange = np.linspace(self.data[:, 1].min(), self.data[:, 1].max(),
                             resolution)
        grid = [[self.decision_func(self.alphas, self.labels,
                                    self.kernel, self.data,
                                    np.array([xr, yr]), self.b)
                 for xr in xrange] for yr in yrange]
        grid = np.array(grid).reshape(len(xrange), len(yrange))

        # Plot decision contours using grid and
        # make a scatter plot of training data
        ax.contour(xrange, yrange, grid, (-1, 0, 1), linewidths=(1, 1, 1),
                   linestyles=('--', '-', '--'), colors=colors)
        ax.scatter(self.data[:, 0], self.data[:, 1],
                   c=self.labels, lw=0, alpha=0.5)

        # Plot support vectors (non-zero alphas)
        # as circled points (linewidth > 0)
        mask = self.alphas != 0.0
        ax.scatter(self.data[:, 0][mask], self.data[:, 1][mask],
                   c=self.labels[mask])

        return grid, ax

    def predict(self, test_data):
        test_data = np.array(test_data)
        classifications = self.decision_func(self.alphas, self.labels,
                                             self.kernel, self.data,
                                             test_data, self.b)

        return np.less_equal(classifications, 0)

if __name__ == '__main__':
    main()
