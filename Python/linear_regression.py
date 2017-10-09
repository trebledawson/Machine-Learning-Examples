###############################################################################
# Linear Regression using Gradient Descent                                    #
# Glenn Dawson                                                                #
# 2017-10-06                                                                  #
###############################################################################

#####################
#  List of Imports  #
#####################
import numpy as np
from numpy.random import rand as rand
import matplotlib.pyplot as plt
import time

##########
#  Main  #
##########
def main():
    start = time.time()
    b = 7
    m = 2
    n = 100
    noise = rand(n,1)
    X = rand(n,1)*10
    Y = m*X + b + noise

    (theta,costs) = gradient_descent(X,Y)

    print(time.time() - start)

    plt.figure(1)
    plt.subplot(121)
    plt.plot(X,Y,'.')
    plt.title('Plot of randomly generated data')
    plt.plot(X,theta[1]*X + theta[0])

    plt.subplot(122)
    plt.plot(costs)
    plt.title('Cost function values over gradient descent iterations')
    plt.show()

###############
#  Functions  #
###############

###########################################################
#  Gradient descent algorithm for linear regression       #
#  -----------                                            #
#  Arguments:                                             #
#  X is a set of independent variables                    #
#  Y is the set of dependent variables according to Y(X)  #
#  theta is the initial fit value                         #
#  alpha is the learning rate                             #
###########################################################

def gradient_descent(X, Y, theta = np.array([0.0,0.0]),alpha = 0.001):
    X = np.hstack((np.ones(len(X)).reshape(-1,1),X))
    delta_cost = 1
    costs = []
    count = 0
    while delta_cost > 0.000000001:
        # Calculating error
        H = np.inner(theta,X) - Y.T

        # Updating theta
        for i in range(len(theta)):
            J = np.multiply(H, X[:,i])
            theta[i] = theta[i] - alpha * np.sum(J)/(2*len(X))

        # Computing LMS cost function
        cost = np.sum(np.power(H,2)) / (2*len(X))

        # Checking delta_cost
        if count > 0:
            delta_cost = abs(costs[-1] - cost)
        else:
            delta_cost = abs(cost)

        costs.append(cost)
        count += 1
    return theta,costs


if __name__ == '__main__':
    main()