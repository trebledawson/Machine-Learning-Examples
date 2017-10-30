###############################################################################
# Naive Bayes Classifier                                                      #
# Glenn Dawson                                                                #
# 2017-10-08                                                                  #
###############################################################################

#####################
#  List of Imports  #
#####################
import numpy as np
import pandas as pd
import string
import scipy.stats as spstats
import numpy.random as rand
import matplotlib.pyplot as plt
from sklearn import datasets as ds
from sklearn.model_selection import KFold

##########
#  Main  #
##########

def main():
    UCI_data = 1

    if UCI_data == 1:
        iris()
    elif UCI_data == 2:
        wine()
    elif UCI_data == 3:
        cancer()
    elif UCI_data == 4:
        letters()
    elif UCI_data == 5:
        drive()

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
                posterior[j] = self.prior[j] * np.prod(cc_likelihood[j,:],
                                                       axis=0)

            # ...and make classification decision.
            classifications.append(np.argmax(posterior))

        return np.array(classifications)

##################
#  Test scripts  #
##################

def run_naive_bayes(dtrain, dtest, ltrain, ltest):
    # Train Naive Bayes classifier
    nb = NaiveBayes(dtrain, ltrain)

    # Classification of test data
    pred = nb.predict(dtest)

    # Evaluation of results
    evaluation = np.logical_and(ltest.T, pred)
    correct = np.count_nonzero(evaluation)

    return correct, len(dtest)

def test_2d():
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

def iris():
    print('Program start.')

    # Load Fisher iris data
    print('Loading Fisher iris data...')
    iris = ds.load_iris()

    kf = KFold(n_splits=10, shuffle=True)
    results = []
    totals = []
    count = 1
    for train_idx, test_idx in kf.split(iris.data):
        dtrain = iris.data[train_idx]
        dtest = iris.data[test_idx]
        ltrain = iris.target[train_idx]
        ltest = iris.target[test_idx]

        result, total = run_naive_bayes(dtrain, dtest, ltrain, ltest)
        percentage = 100 * result / total

        # Print results
        print('Fold', count, 'accuracy is', percentage, 'percent.')

        results.append(result)
        totals.append(total)
        count += 1

    result = sum(results)
    total = sum(totals)

    print('Overall K-fold accuracy is', 100 * result / total, 'percent.')

def wine():
    print('Program start.')

    # Load wine data
    print('Loading UCI wine dataset...')
    wine = ds.load_wine()

    kf = KFold(n_splits=10, shuffle=True)
    results = []
    totals = []
    count = 1
    for train_idx, test_idx in kf.split(wine.data):
        dtrain = wine.data[train_idx]
        dtest = wine.data[test_idx]
        ltrain = wine.target[train_idx]
        ltest = wine.target[test_idx]

        result, total = run_naive_bayes(dtrain, dtest, ltrain, ltest)
        percentage = 100 * result / total

        # Print results
        print('Fold', count, 'accuracy is', percentage, 'percent.')

        results.append(result)
        totals.append(total)
        count += 1

    result = sum(results)
    total = sum(totals)

    print('Overall K-fold accuracy is', 100 * result / total, 'percent.')

def cancer():
    print('Program start.')

    # Load breast cancer Wisconsin (diagnostic) data
    print('Loading UCI breast cancer Wisconsin dataset...')
    data, target = ds.load_breast_cancer(return_X_y=True)

    kf = KFold(n_splits=10, shuffle=True)
    results = []
    totals = []
    count = 1

    for train_idx, test_idx in kf.split(data):
        dtrain = data[train_idx]
        dtest = data[test_idx]
        ltrain = target[train_idx]
        ltest = target[test_idx]

        result, total = run_naive_bayes(dtrain, dtest, ltrain, ltest)
        percentage = 100 * result / total

        # Print results
        print('Fold', count, 'accuracy is', percentage, 'percent.')

        results.append(result)
        totals.append(total)
        count += 1

    result = sum(results)
    total = sum(totals)

    print('Overall K-fold accuracy is', 100 * result / total, 'percent.')

def letters():
    print('Program start.')

    # Load letter recognition data
    print('Loading UCI letter recognition dataset...')
    file = "G:\Glenn\Misc\Machine Learning\Datasets\\UCI Datasets\Letter " \
           "Recognition Data\letter-recognition.data"

    names = ['letter', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    data = pd.read_csv(file, names=names)
    labels = np.zeros((data.shape[0], 1))

    for letter, index in zip(list(string.ascii_uppercase), list(range(26))):
        labels[data.index[data['letter'] == letter]] = index

    data = np.array(data.drop('letter', 1))

    kf = KFold(n_splits=10, shuffle=True)
    results = []
    totals = []
    count = 1

    for train_idx, test_idx in kf.split(data):
        dtrain = data[train_idx]
        dtest = data[test_idx]
        ltrain = labels[train_idx]
        ltest = labels[test_idx]

        print('Fold evaluation...')
        result, total = run_naive_bayes(dtrain, dtest, ltrain, ltest)
        percentage = 100 * result / total

        # Print results
        print('Fold', count, 'accuracy is', percentage, 'percent.')

        results.append(result)
        totals.append(total)
        count += 1

    result = sum(results)
    total = sum(totals)

    print('Overall K-fold accuracy is', 100 * result / total, 'percent.')

def drive():

    # Load sensorless drive data
    print('Loading UCI sensorless drive diagnosis dataset...')
    file = "G:\Glenn\Misc\Machine Learning\Datasets\\UCI " \
           "Datasets\Sensorless Drive Diagnosis\Sensorless_drive_diagnosis.txt"

    names = list(range(49))

    data = pd.read_csv(file, delim_whitespace=True, names=names)
    labels = np.array(data[48]).reshape(-1, 1)
    labels -= 1

    data = np.array(data.drop(48, 1))

    kf = KFold(n_splits=10, shuffle=True)
    results = []
    totals = []
    count = 1

    for train_idx, test_idx in kf.split(data):
        dtrain = data[train_idx]
        dtest = data[test_idx]
        ltrain = labels[train_idx]
        ltest = labels[test_idx]

        print('Fold evaluation...')
        result, total = run_naive_bayes(dtrain, dtest, ltrain, ltest)
        percentage = 100 * result / total

        # Print results
        print('Fold', count, 'accuracy is', percentage, 'percent.')

        results.append(result)
        totals.append(total)
        count += 1

    result = sum(results)
    total = sum(totals)

    print('Overall K-fold accuracy is', 100 * result / total, 'percent.')


if __name__ == '__main__':
    main()