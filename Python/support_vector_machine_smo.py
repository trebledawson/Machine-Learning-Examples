###############################################################################
# Support Vector Machine                                                      #
# Glenn Dawson                                                                #
# 2017-10-20                                                                  #
# SMO algorithm taken from http://jonchar.net/notebooks/SVM/                  #
###############################################################################

#############
#  Imports  #
#############

import numpy as np
from numpy.random import multivariate_normal as mvnrnd
import matplotlib.pyplot as plt
import time

##########
#  Main  #
##########

def main():
    test_2d()

######################################
#  Support Vector Machine Classifier #
######################################
class SupportVectorMachineSMO:

    def __init__(self, data1, data2, C=1000, kernel='gaussian',
                 b=0):
        print('Initializing SVM attributes...')
        data1 = np.array(data1)
        data2 = np.array(data2)
        self.data = np.vstack((data1, data2))                   # Training data

        labels1 = np.ones((1, data1.shape[0]))
        labels2 = -np.ones((1, data2.shape[0]))
        self.labels = np.ravel(np.hstack((labels1, labels2)))   # Data labels

        self.C = C                              # Regularization parameter
        self.kernel = kernel                    # Kernel definition
        self.alphas = np.zeros(len(self.data))  # Lagrange multipliers
        self.b = b                              # Bias term
        self._obj = []                          # Onjective function cache
        self.num_inst = len(self.data)          # Size of training data

        # Hyperparameters
        self.tol = 0.01         # Error tolerance
        self.epsilon = 0.01     # Alpha tolerance



        print('Training SVM...')
        train_start = time.time()
        # Initialize errors
        self.errors = self.decision_function(self.alphas, self.labels,
                                             self.kernel, self.data,
                                             self.data, self.b) - self.labels

        # Train SVM
        self.train()

        # Extract prediction weights
        self.w = np.sum(self.alphas * self.labels * self.data.T, axis=0)

        print('Support vector machine successfully trained.')
        print('Training time:', time.time() - train_start, 'seconds.')

    def kernelFunc(self, x, y, kernel=None):
        if kernel == 'gaussian':
            # Hyperparameters
            sigma = 1

            if np.ndim(x) == 1 and np.ndim(y) == 1:
                norm = np.linalg.norm(x - y)
                return np.exp(- norm / (2 * sigma ** 2))
            elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (
                    np.ndim(x) == 1 and np.ndim(y) > 1):
                norm = np.linalg.norm(x - y, axis=1)
                return np.exp(- norm / (2 * sigma ** 2))
            elif np.ndim(x) > 1 and np.ndim(y) > 1:
                norm = np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :],
                                      axis=2)
                return np.exp(- norm / (2 * sigma ** 2))

        elif kernel == 'polynomial':
            # Hyperparameters
            gamma = 1
            d = 3

            dot = x @ y.T
            return (1 + gamma*dot) ** d

        elif kernel == 'inner':
            # Hyperparameters
            b = 1

            return x @ y.T + b

    def objective_dual(self, alphas, target, kernel, X_train):
        """
        Returns the SVM objective function based in the input self
        defined by:
        `alphas`: vector of Lagrange multipliers
        `target`: vector of class labels (-1 or 1) for training data
        `kernel`: kernel function
        `X_train`: training data for self.
        """

        targets_prod = target * target
        alphas_prod = alphas * alphas
        K = self.kernelFunc(X_train, X_train, kernel = kernel)
        half_sum = 0.5 * np.sum(targets_prod * K * alphas_prod)
        return np.sum(alphas) - half_sum

    def decision_function(self, alphas, target, kernel, X_train, x_test, b):
        """
        Applies the SVM decision function to the input feature vectors
        in `x_test`.
        """
        K = self.kernelFunc(X_train, x_test, kernel = kernel)

        return (alphas * target) @ K - b

    def train(self):

        num_changed = 0
        examine_all = 1

        while (num_changed > 0) or (examine_all):
            num_changed = 0
            if examine_all:
                # Loop over all training examples
                for i in range(self.alphas.shape[0]):
                    examine_result = self.examine_example(i)
                    num_changed += examine_result
                    if examine_result:
                        obj_result = self.objective_dual(self.alphas,
                                                         self.labels,
                                                         self.kernel,
                                                         self.data)
                        self._obj.append(obj_result)
            else:
                # Loop over examples where alphas are not already at their
                # limits
                for i in \
                np.where((self.alphas != 0) & (self.alphas != self.C))[0]:
                    examine_result = self.examine_example(i)
                    num_changed += examine_result
                    if examine_result:
                        obj_result = self.objective_dual(self.alphas,
                                                         self.labels,
                                                         self.kernel,
                                                         self.data)
                        self._obj.append(obj_result)

            if examine_all == 1:
                examine_all = 0
            elif num_changed == 0:
                examine_all = 1

        return

    def examine_example(self, j):
        tol = self.tol
        y2 = self.labels[j]
        alpha_j = self.alphas[j]
        E2 = self.errors[j]
        r2 = E2 * y2

        # Proceed if error is within specified tolerance (tol)
        if ((r2 < -tol and alpha_j < self.C) or (r2 > tol and alpha_j > 0)):

            if len(self.alphas[(self.alphas != 0) & (
                self.alphas != self.C)]) > 1:

                # Use 2nd choice heuristic is choose max difference in error
                if self.errors[j] > 0:
                    i = np.argmin(self.errors)
                elif self.errors[j] <= 0:
                    i = np.argmax(self.errors)

                step_result = self.take_step(i, j)

                if step_result:
                    return 1

            # Loop through non-zero and non-C alphas, starting at a random point
            for i in np.roll(
                    np.where((self.alphas != 0) & (self.alphas != self.C))[
                        0],
                    np.random.choice(np.arange(self.num_inst))):

                step_result = self.take_step(i, j)

                if step_result:
                    return 1

            # Loop through all alphas, starting at a random point
            for i in np.roll(np.arange(self.num_inst),
                              np.random.choice(np.arange(self.num_inst))):
                step_result = self.take_step(i, j)
                if step_result:
                    return 1

        return 0
    
    def take_step(self, i, j):

        # Skip if chosen alphas are the same
        if i == j:
            return 0

        alpha_i = self.alphas[i]
        alpha_j = self.alphas[j]
        y1 = self.labels[i]
        y2 = self.labels[j]
        E1 = self.errors[i]
        E2 = self.errors[j]
        s = y1 * y2

        # Compute lb & ub, the bounds on new possible alpha values
        if (y1 != y2):
            lb = max(0, alpha_j - alpha_i)
            ub = min(self.C, self.C + alpha_j - alpha_i)
        elif (y1 == y2):
            lb = max(0, alpha_i + alpha_j - self.C)
            ub = min(self.C, alpha_i + alpha_j)
        if (lb == ub):
            return 0

        # Compute kernel & 2nd derivative eta
        k11 = self.kernelFunc(self.data[i], self.data[i], self.kernel)
        k12 = self.kernelFunc(self.data[i], self.data[j], self.kernel)
        k22 = self.kernelFunc(self.data[j], self.data[j], self.kernel)
        eta = 2 * k12 - k11 - k22

        # Compute new alpha 2 (a2) if eta is negative
        if (eta < 0):
            a2 = alpha_j - y2 * (E1 - E2) / eta

            # Clip a2 based on bounds lb & ub
            if lb < a2 < ub:
                a2 = a2
            elif (a2 <= lb):
                a2 = lb
            elif (a2 >= ub):
                a2 = ub

        # If eta is non-negative, move new a2 to bound with greater
        # objective function value
        else:
            alphas_adj = self.alphas.copy()
            alphas_adj[j] = lb
            # objective function output with a2 = L
            Lobj = self.objective_dual(alphas_adj, self.labels, self.kernel,
                                      self.data)
            alphas_adj[j] = ub
            # objective function output with a2 = H
            Hobj = self.objective_dual(alphas_adj, self.labels, self.kernel,
                                      self.data)
            if Lobj > (Hobj + self.epsilon):
                a2 = lb
            elif Lobj < (Hobj - self.epsilon):
                a2 = ub
            else:
                a2 = alpha_j

        # Push a2 to 0 or C if very close
        if a2 < 1e-8:
            a2 = 0.0
        elif a2 > (self.C - 1e-8):
            a2 = self.C

        # If examples can't be optimized within epsilon, skip this pair
        if (np.abs(a2 - alpha_j) < self.epsilon * (
                        a2 + alpha_j + self.epsilon)):
            return 0

        # Calculate new alpha 1 (a1)
        a1 = alpha_i + s * (alpha_j - a2)

        # Update threshold b to reflect newly calculated alphas
        # Calculate both possible thresholds
        b1 = E1 + y1 * (a1 - alpha_i) * k11 + y2 * (a2 - alpha_j) * k12 + self.b
        b2 = E2 + y1 * (a1 - alpha_i) * k12 + y2 * (a2 - alpha_j) * k22 + self.b

        # Set new threshold based on if a1 or a2 is bound by L and/or H
        if 0 < a1 and a1 < self.C:
            b_new = b1
        elif 0 < a2 and a2 < self.C:
            b_new = b2
        # Average thresholds if both are bound
        else:
            b_new = (b1 + b2) * 0.5

        # Update self object with new alphas & threshold
        self.alphas[i] = a1
        self.alphas[j] = a2

        # Update error cache
        # Error cache for optimized alphas is set to 0 if they're unbound
        for index, alph in zip([i, j], [a1, a2]):
            if 0.0 < alph < self.C:
                self.errors[index] = 0.0

        # Set non-optimized errors based on equation 12.11 in Platt's book
        non_opt = [n for n in range(self.num_inst) if (n != i and n != j)]
        self.errors[non_opt] = self.errors[non_opt] + \
                                y1 * (a1 - alpha_i) * self.kernelFunc(
                                    self.data[i], self.data[non_opt],
                                    self.kernel) + \
                                y2 * (a2 - alpha_j) * self.kernelFunc(
                                    self.data[j], self.data[non_opt],
                                    self.kernel) + self.b - b_new

        # Update self threshold
        self.b = b_new

        return 1

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
        grid = [[self.decision_function(self.alphas, self.labels,
                                   self.kernel, self.data,
                                   np.array([xr, yr]), self.b) for xr in
                 xrange] for yr in yrange]
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
        classifications = self.decision_function(self.alphas, self.labels,
                                                 self.kernel, self.data,
                                                 test_data, self.b)

        return np.less_equal(classifications, 0)

##################
#  Test scripts  #
##################

def test_2d():
    print('Program start.')
    start = time.time()

    # Generate training data
    print('Generating training data...')
    mu1 = [1, 1]
    mu2 = [2, 2]
    sigma1 = [[0.15, 0], [0, 0.15]]
    sigma2 = [[0.15, 0], [0, 0.15]]
    class_1_instances = 100
    class_2_instances = 100
    Q1 = mvnrnd(mu1, sigma1, class_1_instances)
    Q2 = mvnrnd(mu2, sigma2, class_2_instances)

    # Train SVM
    print('Training support vector machine...')
    svm = SupportVectorMachineSMO(Q1, Q2)

    # Plot training data and decision boundary
    fig, ax = plt.subplots()
    svm.plot_decision_boundary(ax)
    plt.xlim(xmin = 0, xmax = 3)
    plt.ylim(ymin = 0, ymax = 3)

    # Generate test data
    print('Generating test data...')
    mu = [1.4, 1.4]
    sigma = [[0.7, 0], [0, 0.7]]
    test_instances = 10000
    Q = mvnrnd(mu, sigma, test_instances)

    # Classify test data
    print('Classifying test data...')
    P = svm.predict(Q)

    plt.figure()
    plt.title('Classified test data')
    plt.plot(Q[P == 0, 0], Q[P == 0, 1], 'y.')
    plt.plot(Q[P == 1, 0], Q[P == 1, 1], 'm.')
    plt.xlim(xmin = 0, xmax = 3)
    plt.ylim(ymin = 0, ymax = 3)

    print('Total elapsed time is', time.time() - start, 'seconds.')
    plt.show()



if __name__ == '__main__':
    main()
