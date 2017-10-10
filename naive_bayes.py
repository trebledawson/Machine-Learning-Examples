###############################################################################
# Naive Bayes Classifier                                                      #
# Glenn Dawson                                                                #
# 2017-10-08                                                                  #
###############################################################################

#####################
#  List of Imports  #
#####################
import numpy as np
import scipy.stats as spstats
import numpy.random as rand
import matplotlib.pyplot as plt

##########
#  Main  #
##########

def main():
    cases = 1000
    mu1 = [1,1]
    mu2 = [3,3]
    mu3 = [1,3]
    sigma1 = [[0.1,0],[0,0.1]]
    sigma2 = [[0.1,0],[0,0.1]]
    sigma3 = [[0.1,0],[0,0.1]]
    Q1 = rand.multivariate_normal(mu1, sigma1, cases)
    Q2 = rand.multivariate_normal(mu2, sigma2, cases)
    Q3 = rand.multivariate_normal(mu3, sigma3, cases)
    Q = np.vstack((Q1,Q2,Q3))
    Y = np.vstack((np.zeros((cases,1)),np.ones((cases,1)),np.full((cases,1),2)))

    nb = NaiveBayes(Q,Y)

    plt.subplot(121)
    plt.subplot(121)
    plt.plot(Q1[:,0], Q1[:,1], 'b.')
    plt.plot(Q2[:,0], Q2[:,1], 'r.')
    plt.plot(Q3[:,0], Q3[:,1], 'g.')
    plt.title('Plot of randomly generated data')

    # Classification test
    mu_test = [2,2]
    sigma_test = [[1,0],[0,1]]
    Q_test = rand.multivariate_normal(mu_test, sigma_test, cases)
    P = nb.predict(Q_test)

    # Plotting data
    plt.subplot(122)
    plt.plot(Q_test[P == 0,0], Q_test[P == 0,1], 'b.')
    plt.plot(Q_test[P == 1,0], Q_test[P == 1,1], 'r.')
    plt.plot(Q_test[P == 2,0], Q_test[P == 2,1], 'g.')
    plt.title('Naive Bayes classification of data')

    plt.show()

############################
#  Naive Bayes Classifier  #
############################

class NaiveBayes:

    def __init__(self, train, labels):
        train = np.array(train)
        labels = np.array(labels)

        # Extract number of classes, instances, and features in training data
        self.num_classes = len(np.unique(labels))
        num_instances = train.shape[0]
        self.num_features = train.shape[1]

        # Compute mu, sigma, and prior probabilities for each class
        self.mu = np.zeros((self.num_classes, self.num_features))
        self.sigma = np.zeros((self.num_classes, self.num_features))
        self.prior = np.zeros(self.num_classes)

        for i in range(self.num_classes):
            self.train_class = train[np.nonzero(labels==i)[0]]
            self.mu[i,:] = np.mean(self.train_class, axis=0)
            self.sigma[i,:] = np.std(self.train_class, axis=0)
            self.prior[i] = len(self.train_class) / num_instances

        print('The Naive Bayes classifier has been successfully trained.\n')

    def predict(self, test):
        test = np.array(test)
        num_instances = test.shape[0]
        classifications = []

        # For each instance...
        for i in range(num_instances):
            posterior = np.zeros(self.num_classes)

            # ...compute class-conditional likelihoods...
            cc_likelihood = np.zeros((self.num_classes,self.num_features))
            for j in range(self.num_classes):
                cc_likelihood[j,:] = spstats.norm.pdf(test[i,:],
                                                           self.mu[j,:],
                                                           self.sigma[j,:])
            # ...and posterior probabilities...
            for j in range(self.num_classes):
                Q = self.prior[j]
                R = np.prod(cc_likelihood[j, :])
                posterior[j] = self.prior[j] * np.prod(cc_likelihood[j,:],
                                                       axis=0)

            # ...and make classification decision.
            classifications.append(np.argmax(posterior))

        return classifications


if __name__ == '__main__':
    main()