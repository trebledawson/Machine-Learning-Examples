# ############################################################################ #
# Logistic Regression using Gradient Descent                                   #
# Glenn Dawson                                                                 #
# 2017-10-06                                                                   #
# ############################################################################ #

import numpy as np
from numpy.random import multivariate_normal as mvnrnd
from numpy.random import rand
import matplotlib.pyplot as plt
import time


def main():
    # Generate data
    mu1 = [1, 1]
    mu2 = [3, 3]
    sigma1 = [[0.2, 0], [0, 0.1]]
    sigma2 = [[0.1, 0], [0, 0.2]]
    q1 = mvnrnd(mu1, sigma1, 1000)
    q2 = mvnrnd(mu2, sigma2, 1000)

    # Perform gradient descent
    start = time.time()
    theta, costs = gradient_descent(q1, q2, alpha=1e-4, epsilon=1e-3)
    print('Algorithm runtime: {:.4f} seconds'.format(time.time()-start))
    
    # Plot results
    plt.subplot(121)
    plt.plot(q1[:, 0], q1[:, 1], '.')
    plt.plot(q2[:, 0], q2[:, 1], '.')
    plt.grid()
    plt.title('Plot of randomly generated data')

    log_reg_clf = np.linspace(-.25, 4.25, 500)
    plt.plot(log_reg_clf, (-(theta[1] * log_reg_clf + theta[0])) / (theta[2]))

    plt.subplot(122)
    plt.plot(costs)
    plt.grid()
    plt.title('Cost function values over gradient descent iterations')
    plt.show()


def gradient_descent(data_a, data_b, alpha=1e-3, epsilon=1e-2, theta=rand(3)):
    """
    Gradient descent for logistic regression.

    Parameters
    ----------
    data_a : array-like, shape = (n_samples, n_features)
        Data corresponding to class A.

    data_b : array-like, shape = (n_samples, n_features)
        Data corresponding to class B.

    alpha : float
        Learning rate.

    epsilon : float
        Stopping criterion.

    theta : array-like, shape = (3)
        Initial guess for weights.

    Returns
    -------
    theta : array, shape = (3)
        Final weights for logistic regression classification.

    costs : list, shape = (n_steps)
        History of costs over algorithm runtime.
    """
    # Prepend 0 to data vectors
    data_a = np.insert(data_a, 0, 1, axis=1)
    data_b = np.insert(data_b, 0, 1, axis=1)
    data = np.vstack((data_a, data_b))

    # Create labels
    targets = np.concatenate((np.zeros(data_a.shape[0]),
                              np.ones(data_b.shape[0])))

    # Gradient descent algorithm using logistic function
    dCost = 1
    costs = []
    first = True
    while dCost > epsilon:
        # Calculate error
        logits = 1 / (1 + np.exp(-(data @ theta)))

        # Update theta
        grad = (data.T * (logits - targets)).T
        theta = theta - alpha * np.sum(grad, axis=0)

        # Calculate cost
        cost = np.sum(-targets * np.log(logits)
                      - (1 - targets) * np.log(1 - logits))

        if first:
            dCost = abs(cost)
            first = False
        else:
            dCost = abs(costs[-1] - cost)
            
        costs.append(cost)

    return theta, costs


if __name__ == '__main__':
    main()
