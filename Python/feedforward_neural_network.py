###############################################################################
#  Feedforward Neural Network                                                 #
#  Glenn Dawson                                                               #
#  2017-10-21                                                                 #
#  Adapted from Andrew Ng                                                     #
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
from scipy.optimize import minimize


##########
#  Main  #
##########

def main():
    test_2d()


################################
#  Feedforward Neural Network  #
################################

class NeuralNetworkFF:
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
        self.activation_function = 'relu'   # Hidden layer activation function
        self.output_function = 'sigmoid'    # Output layer activation function
        self.eta = 0.01                     # Learning rate
        self.alpha = 0.75                   # Momentum term
        self.max_fail = 20                  # Max validation failures
        self.max_epochs = 1000              # Max epochs to train
        self.num_layers = 2                 # Number of hidden layers
        self.num_nodes = [self.num_feat, 12, 10, self.num_classes]
        ''' ^^^ Note: Also includes input and output nodes ^^^ '''

        # Initialize weights
        self.weights = np.empty(self.num_layers + 1, dtype=object)
        for l in range(self.num_layers + 1):
            self.weights[l] = uniform(-1 / sqrt(self.num_nodes[l] + 1),
                                      1 / sqrt(self.num_nodes[l] + 1),
                                      size=(self.num_nodes[l+1],
                                            self.num_nodes[l] + 1))

        # Initialize delta_weights_old for momentum term
        self.delta_weights_old = np.zeros(self.weights.shape)

        # Initialize costs trackers
        self.train_costs = [1]
        self.test_costs = [1]
        self.val_costs = [1]

        # Train feedforward neural network
        self.train()

    def forward_prop(self, data):
        num_inst = data.shape[0]
        node_outputs = np.empty(self.num_layers + 2, dtype=object)
        node_outputs[0] = data
        weighted_inputs = np.empty(self.num_layers + 2, dtype=object)
        weighted_inputs[0] = data

        for l in range(self.num_layers + 1):
            node_outputs[l+1] = np.ones((num_inst, self.num_nodes[l+1] + 1))
            weighted_inputs[l+1] = np.ones((num_inst, self.num_nodes[l+1] + 1))

        for d in range(num_inst):
            # Calculate hidden layers
            for l in range(self.num_layers):
                for n in range(self.num_nodes[l+1]):
                    weighted_input = node_outputs[l][d] @ self.weights[l][n].T
                    node_outputs[l + 1][d][n+1] = self.activation(
                        weighted_input)
                    weighted_inputs[l+1][d][n+1] = weighted_input

            # Calculate output layer
            for n in range(self.num_nodes[-1]):
                weighted_input = node_outputs[-2][d] @ self.weights[-1][n].T
                node_outputs[-1][d][n+1] = self.output_activation(
                    weighted_input)
                weighted_inputs[-1][d][n+1] = weighted_input

        # Slice bias from output
        node_outputs[-1] = np.delete(node_outputs[-1], 0, axis=1)

        return node_outputs, weighted_inputs

    def cost(self, node_outputs, labels):
        J = 0
        for i in range(self.num_inst):
            cost_term_one = -labels * np.log(node_outputs[-1])
            cost_term_two = (1 - labels) * np.log(1 - node_outputs[-1])
            J += np.sum(cost_term_one - cost_term_two)

        J = J / self.num_inst

        # Cost regularization
        reg_sum = 0
        for weight in self.weights:
            reg_sum += np.sum(np.power(weight, 2))

        J += self.eta / (2 * self.num_inst) * reg_sum

        return J

    # Backpropagation algorithm derived from Andrew Ng's Coursera notes
    def backprop(self, weights):
        # Update weights
        w = np.empty(self.num_layers + 1, dtype=object)
        for l in range(self.num_layers + 1):
            w[l] = weights[:self.num_nodes[l+1] * (self.num_nodes[l] + 1)]
            self.weights[l] = np.reshape(w[l], (self.num_nodes[l + 1],
                                                self.num_nodes[l] + 1))
            weights = weights[self.num_nodes[l+1] * (self.num_nodes[l] + 1):]
        m = self.data_train.shape[0]

        # Forward pass
        node_outputs, weighted_inputs = self.forward_prop(self.data_train)

        # Cost function
        J = self.cost(node_outputs, self.labels_train)

        # Backpropagation
        cost_grad = np.empty(self.num_layers + 1, dtype=object)
        deltas = np.empty(self.num_layers + 2, dtype=object)
        deltas[-1] = node_outputs[-1] - self.labels_train

        for l in range(2, self.num_layers + 3):
            gradient = self.gradient(weighted_inputs[-l])
            d = (deltas[-(l-1)] @ self.weights[-(l-1)])
            deltas[-l] = np.delete(d * gradient, 0, axis=1)

            cost_grad[-(l-1)] = deltas[-(l-1)].T @ node_outputs[-l] / m

        # Gradient regularization
        reg_weights = self.eta * self.weights / m
        for weight in reg_weights:
            weight[:, 0] = 0

        grad = cost_grad + reg_weights

        D = np.array([])
        for grad in grad:
            D = np.concatenate((D, np.ravel(grad)))

        return J, D

    def train(self):
        weights = np.array([])
        for weight in self.weights:
            weights = np.concatenate((weights, np.ravel(weight)))

        fmin = minimize(fun=self.backprop, x0=weights, args=(), method='TNC',
                        jac=True, options={'maxiter': self.max_epochs})
        print('Minimized cost value:', fmin.fun)
        print('Epochs:', fmin.nfev)
        print(fmin.success)
        print(fmin.message)

        # Update weights
        fit = fmin.x
        w = np.empty(self.num_layers + 1, dtype=object)
        for l in range(self.num_layers + 1):
            w[l] = fit[:self.num_nodes[l+1] * (self.num_nodes[l] + 1)]
            self.weights[l] = np.reshape(w[l], (self.num_nodes[l+1],
                                                self.num_nodes[l] + 1))
            fit = fit[self.num_nodes[l+1] * (self.num_nodes[l] + 1):]

    '''
    # Backpropagation algorithm derived from Dr. Polikar's lecture notes   
    def backprop(self):
        # Run feedforward pass through the network
        node_outputs, weighted_inputs = self.forward_prop(self.data_train)

        # Initialize backprop variables
        hidden_sum = np.empty(self.num_layers + 1, dtype=object)
        delta_j = np.empty(self.num_layers + 1, dtype=object)
        delta_weights = np.empty(self.num_layers + 1, dtype=object)

        # Determine change in output layer weights
        output_gradient = self.gradient(weighted_inputs[-1])
        delta_j[-1] = (node_outputs[-1] - self.labels_train) * output_gradient
        delta_weights[-1] = self.eta * (delta_j[-1].T @ (node_outputs[-2]))

        # Determine change in hidden layer weights
        for l in range(2, node_outputs.shape[0]):
            hidden_gradient = self.gradient(weighted_inputs[-l])
            hidden_sum[-l] = np.delete(delta_j[-(l-1)] @ self.weights[-(l-1)],
                                       0, axis=1)
            delta_j[-l] = hidden_sum[-l] * hidden_gradient
            delta_weights[-l] = self.eta * (delta_j[-l].T @ node_outputs[-(
                l+1)])

        # Update weights with momentum
        momentum_term_one = (1-self.alpha) * delta_weights
        momentum_term_two = self.alpha * self.delta_weights_old
        self.weights -= momentum_term_one + momentum_term_two
        self.delta_weights_old = delta_weights

        # Cost function
        self.train_costs.append(self.cost(node_outputs, self.labels_train))
    
    def train(self):
        # Train feedforward neural network
        val_fail = 0
        epochs = 0
        while val_fail < self.max_fail and epochs < self.max_epochs:
            self.backprop()

            test_output, _ = self.forward_prop(self.data_test)
            val_output, _ = self.forward_prop(self.data_val)
            self.test_costs.append(self.cost(test_output, self.labels_test))
            self.val_costs.append(self.cost(val_output, self.labels_val))

            if self.val_costs[-1] > self.val_costs[-2]:
                val_fail += 1
            elif val_fail > 0:
                val_fail -= 1

            epochs += 1

        # Plot performance during training
        plt.title('Training Performance of Feedforward Neural Network')
        plt.plot(self.train_costs, 'b', label='Train')
        plt.plot(self.test_costs, 'r', label='Test')
        plt.plot(self.val_costs, 'g', label='Validation')
        plt.legend()

        print('Best performance is', self.val_costs[-1], 'at epoch', epochs,
              '.')

        plt.show()
    '''

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
            sigmoid = self.activation(input_data)
            return np.multiply(sigmoid, 1 - sigmoid)

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

##################
#  Test Scripts  #
##################
def test_2d():
    mu1 = [0.7, 0.7]
    mu2 = [2.2, 2.2]
    sigma1 = [[0.2, 0], [0, 0.2]]
    sigma2 = [[0.2, 0], [0, 0.2]]
    class_1_inst = 200
    class_2_inst = 200
    Q1 = np.array(mvnrnd(mu1, sigma1, class_1_inst))
    Q2 = np.array(mvnrnd(mu2, sigma2, class_2_inst))
    Q = np.vstack((Q1, Q2))

    labels1 = np.zeros(len(Q1))
    labels2 = np.ones(len(Q2))
    labels = np.hstack((labels1, labels2)).reshape(-1, 1)

    net = NeuralNetworkFF(Q, labels)

    mu_t = [1.4, 1.4]
    sigma_t = [[0.25, 0],[0, 0.25]]
    test_inst = 10000
    test = np.array(mvnrnd(mu_t, sigma_t, test_inst))

    P = net.predict(test)

    plt.subplot(121)
    plt.plot(Q1[:, 0], Q1[:, 1], 'b.')
    plt.plot(Q2[:, 0], Q2[:, 1], 'r.')
    plt.title('Randomly generated data')
    plt.xlim(xmin=-0.5, xmax=3.5)
    plt.ylim(ymin=-0.5, ymax=3.5)


    plt.subplot(122)
    plt.plot(test[P == 0, 0], test[P == 0, 1], 'b.')
    plt.plot(test[P == 1, 0], test[P == 1, 1], 'r.')
    plt.title('Classified test data')
    plt.xlim(xmin=-0.5, xmax=3.5)
    plt.ylim(ymin=-0.5, ymax=3.5)

    plt.show()

if __name__ == '__main__':
    main()