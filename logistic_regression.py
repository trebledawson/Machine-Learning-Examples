###############################################################################
# Logistic Regression using Gradient Descent                                  #
# Glenn Dawson                                                                #
# 2017-10-06                                                                  #
###############################################################################

#####################
#  List of Imports  #
#####################
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import time

##########
#  Main  #
##########

def main():
    start = time.time()
    # Generate test data
    mu1 = [1,1]
    mu2 = [3,3]
    sigma1 = [[0.1,0],[0,0.1]]
    sigma2 = [[0.1,0],[0,0.1]]
    Q1 = rand.multivariate_normal(mu1,sigma1,1000)
    Q2 = rand.multivariate_normal(mu2, sigma2, 1000)

    # Perform gradient descent
    (theta,costs) = gradient_descent(Q1,Q2)

    print(time.time()-start)
    # Plot results
    plt.subplot(121)
    plt.plot(Q1[:, 0], Q1[:, 1], '.')
    plt.plot(Q2[:, 0], Q2[:, 1], '.')
    plt.title('Plot of randomly generated data')

    X = np.linspace(0,5,500)
    plt.plot(X,(-(theta[1]*X+theta[0]))/(theta[2]))

    plt.subplot(122)
    plt.plot(costs)
    plt.title('Cost function values over gradient descent iterations')
    plt.show()

###############
#  Functions  #
###############

#################################################################
#  Gradient descent algorithm for logistic regression           #
#  -----------                                                  #
#  Arguments:                                                   #
#  X1 and X2 are arrays of 2D data                              #
#  theta is the initial fit value                               #
#  alpha is the learning rate                                   #
#################################################################

def gradient_descent(X1, X2, theta=np.array([0.0,0.0,0.0]), alpha=0.0001):
    # Add X_0 = 1 to X1, X2 and concatenate
    X1 = np.hstack((np.ones(X1.shape[0]).reshape(-1, 1), np.array(X1)))
    X2 = np.hstack((np.ones(X2.shape[0]).reshape(-1, 1), np.array(X2)))
    X = np.vstack((X1,X2))

    # Create labels
    Y = np.concatenate((np.zeros(X1.shape[0]),np.ones(X2.shape[0])))

    # Gradient descent algorithm using logistic function
    dCost = 1
    costs = []
    count = 0
    while dCost > 0.001:
        # Calculate error
        H = 1/(1+np.exp(-(np.inner(theta,X))))

        # Update theta
        dJ = np.zeros((X.shape[0],X.shape[1]))
        for j in range(X.shape[0]):
            dJ[j,:] = ((H[j]-Y[j])*X[j,:])
        theta = theta - alpha*np.sum(dJ,axis=0)

        # Calculate cost
        J = np.sum(-Y * np.log(H) - (1 - Y) * np.log(1 - H))

        if count > 0:
            dCost = abs(costs[-1] - J)
        else:
            dCost = abs(J)

        costs.append(J)

        # Increase counter
        count += 1

    return theta, costs


if __name__ == '__main__':
    main()