###############################################################################
#  Feedforward Neural Network                                                 #
#  Glenn Dawson                                                               #
#  2017-10-10                                                                 #
###############################################################################

#####################
#  List of Imports  #
#####################
import numpy as np
from numpy.random import uniform
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import normalize
##########
#  Main  #
##########



######################################
#  Feedforward Neural Network Class  #
######################################

class FeedforwardNeuralNetwork:
    def __init__(self, data, labels, hidden_nodes=10):
        np.array(normalize(data))
        np.array(labels)

        self.classes = np.unique(labels)
        self.num_classes = len(self.classes)
        self.num_features = data.shape[1]
        self.hidden_nodes = hidden_nodes
        self.weights_hidden = np.ndarray((self.hidden_nodes, self.num_features))
        self.weights_output = np.ndarray((self.num_classes, self.hidden_nodes))

        self.train(data, labels)


    def train(self, data, labels):
        # Split data into train, test, and validation data (0.7, 0.15, 0.15)
        data_train, data_test, labels_train, labels_test = tts(data, labels,
                                                               test_size=0.3)

        data_test, data_val, labels_test, labels_val = tts (data_test,
                                                            labels_test,
                                                            test_size=0.5)
        while


    def activation(self, data, weights):
        # From http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
        x = (2/3) * np.inner(data, weights)
        return 1.7159*np.tanh(x)

    def backpropagation(self, eta=0.03):
        # dWjk = eta*delta_k*y_j
        # delta_k = (target - actual) * (1-output(k)^2)
        # y_j = f(sum[1,d]{ w_ji, x_i }

        # dWij = eta*delta_j*x*i
        # delta_j = sum[1,c]{ delta_k * w_kj } * (1-output(j)*2)
        # x_i = d(net_j) / d(w_ji)














if __name__ == '__main__':
    main()