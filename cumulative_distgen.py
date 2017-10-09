############################################################################
# Cumulative Distribution Function for Arbitrary Probability Distributions #
# Glenn Dawson                                                             #
# 2017-10-06                                                               #
############################################################################

###################
# List of Imports #
###################
import random
import numpy as np
import matplotlib.pyplot as plt

########
# Main #
########
def main():
    random_discrete_distribution = np.random.rand(400,1)
    random_discrete_distribution = random_discrete_distribution / np.sum(random_discrete_distribution)
    sample = cumulative_dist_gen(random_discrete_distribution,10000)
    plt.hist(sample,100)
    plt.show()

##################################################################
# Cumulative Distribution Function                               #
# -------------                                                  #
# Arguments:                                                     #
# X is a vector containing normalized probabilities              #
# Y is an integer representing the number of samples to be drawn #
##################################################################
def cumulative_dist_gen(X,Y):
    sample = []
    for i in range (0,Y):
        r = random.uniform(0,1)
        cumdist = 0
        j=0
        while (r >= cumdist):
            cumdist = cumdist + X[j]
            j += 1
        sample.append(j)
    return sample

if __name__ == '__main__':
    main()