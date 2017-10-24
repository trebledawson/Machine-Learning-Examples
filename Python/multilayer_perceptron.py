###############################################################################
#  Multilayer Perceptron                                                      #
#  Glenn Dawson                                                               #
#  2017-10-24                                                                 #
###############################################################################

#############
#  Imports  #
#############
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from numpy.random import uniform as uniform
from numpy.random import multivariate_normal as mvnrnd
from math import sqrt
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
import time

##########
#  Main  #
##########
def main():
    test()

###########################
#  Multilayer Perceptron  #
###########################

class MultilayerPerceptron:
    def __init__(self, data, labels):
        # Initialize data, labels
        self.data = np.array(data)
        self.labels = np.array(labels)

        # Format data and labels
        self.data = np.insert(self.data, 0, 1, axis=1)
        encoder = OneHotEncoder(sparse=False)
        self.labelsOH = encoder.fit_transform(self.labels)

        # Split data into training, testing, validation data
        self.data_train, data_other, self.labels_train, labels_other = tts(
            self.data, self.labelsOH, test_size=0.3)

        self.data_test, self.data_val, self.labels_test, self.labels_val = tts(
            data_other, labels_other, test_size=0.5)

        # Extract data dimensions
        self.num_inst = self.data_train.shape[0]
        self.num_feat = self.data_train.shape[1] - 1
        self.num_classes = len(np.unique(self.labels))

        # Hyperparameters
        self.activation_function = 'sigmoid'    # Hidden layer activation
                                                ## function
        self.output_function = 'sigmoid'        # Output layer activation
                                                ## function
        self.cost_function = 'crossentropy'     # Objective function
        self.eta = 0.1                          # Initial learning rate
        self.beta_1 = 0.7                       # Learning rate update coeff 1
        self.beta_2 = 1.05                      # Learning rate update coeff 2
        self.alpha = 0.9                        # Momentum term
        self.max_fail = 15                      # Max validation failures
        self.max_epochs = 1000                  # Max epochs to train
        self.num_layers = 1                     # Number of hidden layers
        self.num_nodes = [self.num_feat, 30, self.num_classes]
        ''' ^^^ Note: Also includes input and output nodes ^^^ '''

        # Initialize weights
        self.weights = np.empty(self.num_layers + 1, dtype=object)
        for l in range(self.num_layers + 1):
            self.weights[l] = uniform(-1 / sqrt(self.num_nodes[l] + 1),
                                      1 / sqrt(self.num_nodes[l] + 1),
                                      size=(self.num_nodes[l + 1],
                                            self.num_nodes[l] + 1))

        # Initialize delta_weights_old for momentum term
        self.delta_weights_old_out = np.zeros(self.weights[-1].shape)
        self.delta_weights_old_hid = np.zeros(self.weights[-2].shape)

        # Initialize costs trackers
        self.train_costs = [0.1]
        self.test_costs = [0.1]
        self.val_costs = [0.1]

        # Train feedforward neural network
        self.train()

    def forward_prop(self, data):
        num_inst = data.shape[0]
        node_outputs = np.empty(self.num_layers + 2, dtype=object)
        node_outputs[0] = data
        weighted_inputs = np.empty(self.num_layers + 2, dtype=object)
        weighted_inputs[0] = np.delete(data, 0, axis=1)

        for l in range(self.num_layers + 1):
            node_outputs[l+1] = np.ones((num_inst, self.num_nodes[l+1] + 1))
            weighted_inputs[l+1] = np.ones((num_inst, self.num_nodes[l+1]))

        for d in range(num_inst):
            # Calculate hidden layers
            for l in range(self.num_layers):
                for n in range(self.num_nodes[l+1]):
                    weighted_input = self.weights[l][n].T @ node_outputs[l][d]
                    node_outputs[l + 1][d][n+1] = self.activation(
                        weighted_input)
                    weighted_inputs[l+1][d][n] = np.delete(weighted_input, 0)

            # Calculate output layer
            for n in range(self.num_nodes[-1]):
                weighted_input = self.weights[-1][n].T @ node_outputs[-2][d]
                node_outputs[-1][d][n+1] = self.output_activation(
                    weighted_input)
                weighted_inputs[-1][d][n] = np.delete(weighted_input, 0)

        # Slice bias from output
        node_outputs[-1] = np.delete(node_outputs[-1], 0, axis=1)

        return node_outputs, weighted_inputs

    def cost(self, node_outputs, labels):
        J = 0
        m = len(labels)

        if self.cost_function == 'crossentropy':
            for i in range(m):
                cost_term_one = -labels * np.log(node_outputs[-1])
                cost_term_two = (1 - labels) * np.log(1 - node_outputs[-1])
                J += np.sum(cost_term_one - cost_term_two)
        elif self.cost_function == 'mse':
            norm = np.linalg.norm(labels - node_outputs[-1], axis=1)
            J = 0.5 * np.sum(np.power(norm, 2))

        J = J

        # Cost regularization
        reg_sum = 0
        for weight in self.weights:
            reg_sum += np.sum(np.power(weight, 2))

        J += (self.eta / 2) * reg_sum

        return J/m

    def backprop(self):
        # Forward pass
        node_outputs, weighted_inputs = self.forward_prop(self.data_train)

        # Extract variables
        z = node_outputs[-1]
        y = node_outputs[-2]
        x = node_outputs[-3]
        net_k = weighted_inputs[-1]
        net_j = weighted_inputs[-2]

        # Calculate output layer weight updates
        grad_net_k = self.gradient(net_k)
        perf_k = self.labels_train - z
        delta_k = perf_k * grad_net_k
        delta_w_kj = self.eta * (delta_k.T @ y)

        # Calculate hidden layer weight updates
        inner_sum = np.sum(delta_k @ self.weights[-1], axis=1).reshape(-1, 1)
        grad_net_j = self.gradient(net_j)
        delta_j = inner_sum * grad_net_j
        delta_w_ji = self.eta * (delta_j.T @ x)

        # Calculate costs and update learning rate
        J = self.cost(node_outputs, self.labels_train)
        self.learning_rate(J)
        self.train_costs.append(J)

        # Update weights with momentum
        momentum_term_one_kj = (1 - self.alpha) * delta_w_kj
        momentum_term_one_ji = (1 - self.alpha) * delta_w_ji
        momentum_term_two_kj = self.alpha * self.delta_weights_old_out
        momentum_term_two_ji = self.alpha * self.delta_weights_old_hid

        self.weights[-1] += momentum_term_one_kj + momentum_term_two_kj
        self.weights[-2] += momentum_term_one_ji + momentum_term_two_ji

        self.delta_weights_old_out = delta_w_kj
        self.delta_weights_old_hid = delta_w_ji

    def train(self):
        # Train feedforward neural network
        training_start = time.time()
        val_fail = 0
        val_equal = 0
        epochs = 0
        while val_fail < self.max_fail and epochs < self.max_epochs and \
                val_equal < self.max_fail:
            self.backprop()

            test_output, _ = self.forward_prop(self.data_test)
            val_output, _ = self.forward_prop(self.data_val)
            self.test_costs.append(self.cost(test_output, self.labels_test))
            self.val_costs.append(self.cost(val_output, self.labels_val))

            if self.val_costs[-1] > self.val_costs[-2]:
                val_fail += 1
            elif val_fail > 0:
                val_fail -= 1

            if self.val_costs[-1] == self.val_costs[-2]:
                val_equal += 1

            epochs += 1

        if epochs < self.max_epochs:
            if val_equal < self.max_fail:
                print('The feedforward neural network has been successfully '
                  'trained.')
            else:
                print('Training diverged; neural network training failed.')
        else:
            print('Maximum epochs reached; neural network training failed.')

        print('Training time:', time.time() - training_start, 'seconds.')

        # Plot performance during training
        plt.title('Training Performance of Feedforward Neural Network')
        plt.plot(self.train_costs, 'b', label='Train')
        plt.plot(self.test_costs, 'r', label='Test')
        plt.plot(self.val_costs, 'g', label='Validation')
        plt.xlabel('Training epochs')
        plt.ylabel('Training error')
        plt.legend()

        print('Best performance is', self.val_costs[-1], 'at epoch', epochs,
              '.')

        plt.show()

    def learning_rate(self, J):
        if J > self.train_costs[-1]:
            self.eta = self.beta_1 * self.eta
        elif J < self.train_costs[-1]:
            self.eta = self.beta_2 * self.eta

    # Prediction
    def predict(self, data):
        data = np.insert(data, 0, 1, axis=1)
        node_outputs, _ = self.forward_prop(data)
        return np.argmax(np.array(node_outputs[-1]), axis=1)

    # Activation function definitions
    def activation(self, input_data):
        if self.activation_function == 'relu':
            return np.maximum(0, input_data)

        elif self.activation_function == 'sigmoid':
            out = 1 / (1 + np.exp(-input_data))
            if out >= 0.99:
                return 0.99
            elif out <= 0.01:
                return 0.01
            else:
                return out

        elif self.activation_function == 'tanh':
            return 1.7159 * np.tanh((2 / 3) * input_data)

        elif self.activation_function == 'linear':
            return input_data

    def gradient(self, input_data):
        if self.activation_function == 'relu':
            output = np.zeros(input_data.shape)
            output[input_data > 0] = 1
            return output

        elif self.activation_function == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-input_data))
            sigmoid[np.nonzero(sigmoid > 0.99)] = 0.99
            sigmoid[np.nonzero(sigmoid < 0.01)] = 0.01
            return sigmoid * (1 - sigmoid)

        elif self.activation_function == 'tanh':
            return 1.14393 * (1 - (np.tanh((2 / 3) * input_data)) ** 2)

        elif self.activation_function == 'linear':
            return np.ones(len(input_data))

    def output_activation(self, input_data):
        if self.output_function == 'sigmoid':
            out = 1 / (1 + np.exp(-input_data))
            if out >= 0.99:
                return 0.99
            elif out <= 0.01:
                return 0.01
            else:
                return out

        elif self.output_function == 'tanh':
            return 1.7159 * np.tanh((2 / 3) * input_data)

        elif self.output_function == 'linear':
            return input_data

    def output_gradient(self, input_data):
        if self.output_function == 'sigmoid':
            sigmoid = self.activation(input_data)
            return np.multiply(sigmoid, 1 - sigmoid)

        elif self.output_function == 'tanh':
            return 1.14393 * (1 - (np.tanh((2 / 3) * input_data)) ** 2)

        elif self.output_function == 'linear':
            return np.ones(len(input_data))

#############
#  Scripts  #
#############
def test():
    mu1 = [0.8, 0.8]
    mu2 = [2.1, 2.1]
    mu3 = [3, 0.4]
    sigma1 = [[0.2, 0], [0, 0.2]]
    sigma2 = [[0.2, 0], [0, 0.2]]
    sigma3 = [[0.2, 0], [0, 0.2]]
    class1_inst = 150
    class2_inst = 150
    class3_inst = 150
    Q1 = np.array(mvnrnd(mu1, sigma1, class1_inst))
    Q2 = np.array(mvnrnd(mu2, sigma2, class2_inst))
    Q3 = np.array(mvnrnd(mu3, sigma3, class3_inst))
    Q = np.vstack((Q1, Q2, Q3))

    labels1 = np.zeros(len(Q1))
    labels2 = np.ones(len(Q2))
    labels3 = np.ones(len(Q3)) + 1
    labels = np.hstack((labels1, labels2, labels3)).reshape(-1, 1)

    net = MultilayerPerceptron(Q, labels)

    plt.figure()
    mu_t = [1.6, 1.6]
    sigma_t = [[0.35, 0], [0, 0.35]]
    test_inst = 10000
    test = np.array(mvnrnd(mu_t, sigma_t, test_inst))

    P = net.predict(test)

    plt.subplot(121)
    plt.plot(Q1[:, 0], Q1[:, 1], 'b.')
    plt.plot(Q2[:, 0], Q2[:, 1], 'r.')
    plt.plot(Q3[:, 0], Q3[:, 1], 'g.')
    plt.title('Randomly generated data')
    plt.xlim(xmin=-0.5, xmax=4.5)
    plt.ylim(ymin=-0.5, ymax=3.5)

    plt.subplot(122)
    plt.plot(test[P == 0, 0], test[P == 0, 1], 'b.')
    plt.plot(test[P == 1, 0], test[P == 1, 1], 'r.')
    plt.plot(test[P == 2, 0], test[P == 2, 1], 'g.')
    plt.title('Classified test data')
    plt.xlim(xmin=-0.5, xmax=3.5)
    plt.ylim(ymin=-0.5, ymax=3.5)

    plt.show()

if __name__ == '__main__':
    main()