# ############################################################################# #
# Naive Bayes Classifier                                                        #
# Glenn Dawson                                                                  #
# 2017-10-08 (updated 2019)                                                     #
# ############################################################################# #

import numpy as np
import scipy.stats as spstats
from numpy.random import multivariate_normal as mvnrand
import matplotlib.pyplot as plt
from sklearn import datasets as ds
from sklearn.model_selection import KFold


def main():
    tests = ['iris', 'wine', 'cancer', False]
    for test in tests:
        print('------')
        if test:
            uci_test(test)
        else:
            synthetic_test()


# ######################## #
#  Naive Bayes Classifier  #
# ######################## #

class NaiveBayes:
    """
    Naive Bayes object. Assumes Gaussian prior.

    Parameters
    ----------
    train : array-like, shape = (n_samples, n_features)
        Training data on which to fit the classifier.

    labels : array-like, shape = (n_samples)
        Target values.

    Attributes
    ----------
    n_classes_ : int
        Number of classes.

    n_features_ : int
        Number of features.

    mu_ : array, shape = (n_classes_, n_features_)
        Mean of each feature per class.

    sigma_ : array, shape = (n_classes_, n_features_)
        Standard deviation of each feature per class.

    prior_ : array, shape = (n_classes_)
        Prior probability of each class.
    """
    def __init__(self, train, labels):
        train = np.array(train)
        labels = np.array(labels)

        # Extract number of classes, instances, and features in training data
        self.n_classes_ = len(np.unique(labels))
        self.n_features_ = train.shape[1]

        # Compute mu, sigma, and prior probabilities for each class
        self.mu_ = np.zeros((self.n_classes_, self.n_features_))
        self.sigma_ = np.zeros((self.n_classes_, self.n_features_))
        self.prior_ = np.zeros(self.n_classes_)

        for i in range(self.n_classes_):
            train_class = train[np.nonzero(labels == i)[0]]
            self.mu_[i, :] = np.mean(train_class, axis=0)
            self.sigma_[i, :] = np.std(train_class, axis=0)
            self.prior_[i] = len(train_class) / train.shape[0]

    def predict(self, data):
        """Make classification predictions on testing data.

         Parameters
         ----------
         data : array-like, shape = (n_samples, n_features)
            Array of data to classify.

         Returns
         -------
         classifications : array, shape = (n_samples)
            Classification predictions for testing data.
        """
        data = np.array(data)
        classifications = []

        # For each instance...
        for i in range(len(data)):
            # ...compute class-conditional likelihoods...
            cc_likelihood = spstats.norm.pdf(data[i, :], self.mu_, self.sigma_)

            # ...and posterior probabilities...
            posterior = self.prior_ * np.prod(cc_likelihood, axis=1)

            # ...and make classification decision.
            classifications.append(np.argmax(posterior))

        return np.array(classifications)


##################
#  Test scripts  #
##################

def run_naive_bayes(x_train, x_test, y_train, y_test):
    """Train a Naive Bayes classifier and evaluate on test data.

    Parameters
    ----------
    x_train : array-like, shape = (n_samples, n_features)
        Training data.

    x_test : array-like, shape = (n_samples, n_features)
        Testing data.

    y_train : array-like, shape = (n_samples)
        Training targets.

    y_test : arraylike, shape = (n_samples)
        Testing targets.

    Returns
    -------
    correct : int
        Number of correctly classified test samples.

    n_test : int
        Number of samples in the testing dataset.
    """
    # Train Naive Bayes classifier
    nb = NaiveBayes(x_train, y_train)

    # Classification of test data
    pred = nb.predict(x_test)

    # Evaluation of results
    evaluation = np.logical_and(y_test.T, pred)
    correct = np.count_nonzero(evaluation)

    return correct, len(x_test)


def synthetic_test():
    n_samples = 1000
    mu1 = [2, 1]
    mu2 = [3, 3]
    mu3 = [1, 3]
    sigma1 = [[0.1, 0],[0, 0.1]]
    sigma2 = [[0.1, 0],[0, 0.1]]
    sigma3 = [[0.1, 0],[0, 0.1]]
    q1 = mvnrand(mu1, sigma1, n_samples)
    q2 = mvnrand(mu2, sigma2, n_samples)
    q3 = mvnrand(mu3, sigma3, n_samples)
    data = np.vstack((q1, q2, q3))
    labels = np.vstack((np.zeros((n_samples, 1)),
                        np.ones((n_samples, 1)),
                        np.full((n_samples, 1), 2)))

    nb = NaiveBayes(data, labels)

    plt.subplot(121)
    plt.plot(q1[:, 0], q1[:, 1], 'b.')
    plt.plot(q2[:, 0], q2[:, 1], 'r.')
    plt.plot(q3[:, 0], q3[:, 1], 'g.')
    plt.title('Plot of randomly generated training data')
    plt.grid()
    plt.xlim(-.25, 4.25)
    plt.ylim(-.25, 4.25)

    # Classification test
    mu_test = [2, 2]
    sigma_test = [[1, 0], [0, 1]]
    q_test = mvnrand(mu_test, sigma_test, n_samples)
    pred = nb.predict(q_test)

    # Plotting data
    plt.subplot(122)
    plt.plot(q_test[pred == 0, 0], q_test[pred == 0, 1], 'b.')
    plt.plot(q_test[pred == 1, 0], q_test[pred == 1, 1], 'r.')
    plt.plot(q_test[pred == 2, 0], q_test[pred == 2, 1], 'g.')
    plt.title('Naive Bayes classification of testing data')
    plt.grid()
    plt.xlim(-.25, 4.25)
    plt.ylim(-.25, 4.25)

    plt.show()


def uci_test(uci_data):
    data, target = load_data(uci_data)

    kf = KFold(n_splits=10, shuffle=True)
    results = []
    totals = []
    fold = 1
    for train_idx, test_idx in kf.split(data):
        x_train = data[train_idx]
        x_test = data[test_idx]
        y_train = target[train_idx]
        y_test = target[test_idx]

        result, total = run_naive_bayes(x_train, x_test, y_train, y_test)
        percentage = 100 * result / total

        # Print results
        print('Fold {0} accuracy is {1:.8f} percent.'.format(fold, percentage))

        results.append(result)
        totals.append(total)
        fold += 1

    result = sum(results)
    total = sum(totals)

    print('Overall 10-fold accuracy on', uci_data, 'dataset is {:.8f} '
          'percent'.format(100 * result / total))


def load_data(data):
    if data == 'iris':
        # Load Fisher iris data
        print('Loading Fisher iris data...')
        data, targets = ds.load_iris(return_X_y=True)
    elif data == 'wine':
        # Load wine data
        print('Loading UCI wine dataset...')
        data, targets = ds.load_wine(return_X_y=True)
    elif data == 'cancer':
        # Load breast cancer Wisconsin (diagnostic) data
        print('Loading UCI breast cancer Wisconsin dataset...')
        data, targets = ds.load_breast_cancer(return_X_y=True)

    return data, targets

if __name__ == '__main__':
    main()
