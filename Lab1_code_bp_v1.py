import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Sigmoid function.
    This function accepts any shape of np.ndarray object as input and perform sigmoid operation.
    """
    return 1 / (1 + np.exp(-x))


def der_sigmoid(y):
    """ First derivative of Sigmoid function.
    The input to this function should be the value that output from sigmoid function.
    """
    return y * (1 - y)


def compute_cost(A3, Y):
    n = Y.shape[1]  # number of examples

    # first cost
    cost = (A3 - Y)
    error_sum = np.sum(abs(cost[:]))
    cost_value = error_sum / n

    # second cost
    # for i in range(A3.shape[1]):
    #     if A3[:, i] < 1e-4:
    #         A3[:, i] = 1e-4
    #     elif A3[:, i] > 0.9999:
    #         A3[:, i] = 0.9999

    #cost = - np.sum(np.multiply(np.log(A3), Y) + np.multiply(1-Y, np.log(1-A3))) / m

    # third cost
    #cost = (1. / m) * (-np.dot(Y, np.log(A3).T) - np.dot(1 - Y, np.log(1 - A3).T))

    cost_value = np.squeeze(cost_value)  # makes sure cost is the dimension we expect.
    return cost, cost_value

def update_parameters(parameters, grads, learning_rate = 1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    dW3 = grads["dW3"]
    db3 = grads["db3"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

    return parameters

class GenData:
    @staticmethod
    def _gen_linear(n=100):
        """ Data generation (Linear)

        Args:
            n (int):    the number of data points generated in total.

        Returns:
            data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
                a data point in 2d space.
            labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
                Each row represents a corresponding label (0 or 1).
        """
        data = np.random.uniform(0, 1, (n, 2))

        inputs = []
        labels = []

        for point in data:
            inputs.append([point[0], point[1]])

            if point[0] > point[1]:
                labels.append(0)
            else:
                labels.append(1)

        return np.array(inputs), np.array(labels).reshape((-1, 1))

    @staticmethod
    def _gen_xor(n=100):
        """ Data generation (XOR)

        Args:
            n (int):    the number of data points generated in total.

        Returns:
            data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
                a data point in 2d space.
            labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
                Each row represents a corresponding label (0 or 1).
        """
        data_x = np.linspace(0, 1, n // 2)

        inputs = []
        labels = []

        for x in data_x:
            inputs.append([x, x])
            labels.append(0)

            if x == 1 - x:
                continue

            inputs.append([x, 1 - x])
            labels.append(1)

        return np.array(inputs), np.array(labels).reshape((-1, 1))

    @staticmethod
    def fetch_data(mode, n):
        """ Data gather interface

        Args:
            mode (str): 'Linear' or 'XOR', indicate which generator is used.
            n (int):    the number of data points generated in total.
        """
        assert mode == 'Linear' or mode == 'XOR'

        data_gen_func = {
            'Linear': GenData._gen_linear,
            'XOR': GenData._gen_xor
        }[mode]

        return data_gen_func(n)


class SimpleNet:
    def __init__(self, hidden_size, num_step=2000, print_interval=100):
        """ A hand-crafted implementation of simple network.

        Args:
            hidden_size:    the number of hidden neurons used in this model.
            num_step (optional):    the total number of training steps.
            print_interval (optional):  the number of steps between each reported number.
        """
        self.num_step = num_step
        self.print_interval = print_interval
        self.learning_rate = 0.1
        self.inputs = [[0, 0]]
        self.labels = [[0]]
        self.output = [0.0]
        self.cost = 0.0
        self.cache = {}

        # Model parameters initialization
        # Please initiate your network parameters here.
        '''
            n_x -- size of the input layer
            n_h1 -- size of the hidden layer 1
            n_h2 -- size of the hidden layer 2
            n_y -- size of the output layer
            
            W1 -- weight matrix of shape (n_h1, n_x)
            b1 -- bias vector of shape (n_h1, 1)
            W2 -- weight matrix of shape (n_h2, n_h1)
            b2 -- bias vector of shape (n_h2, 1)
            W3 -- weight matrix of shape (n_y, n_h2)
            b3 -- bias vector of shape (n_y, 1)
        '''
        n_x = 2
        n_h1 = hidden_size
        n_h2 = hidden_size
        n_y = 1
        np.random.seed(1)

        #Xavier Initialization
        # W1 = np.random.randn(n_h1, n_x) * np.sqrt(1 / n_x)
        # b1 = np.zeros([n_h1, 1])
        # W2 = np.random.randn(n_h2, n_h1) * np.sqrt(1 / n_h1)
        # b2 = np.zeros([n_h2, 1])
        # W3 = np.random.randn(n_y, n_h2) * np.sqrt(1 / n_h2)
        # b3 = np.zeros([n_y, 1])

        # Normal distribution
        W1 = np.random.randn(n_h1, n_x)
        b1 = np.zeros([n_h1, 1])
        W2 = np.random.randn(n_h2, n_h1)
        b2 = np.zeros([n_h2, 1])
        W3 = np.random.randn(n_y, n_h2)
        b3 = np.zeros([n_y, 1])

        self.parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}


    @staticmethod
    def plot_result(data, gt_y, pred_y):
        """ Data visualization with ground truth and predicted data comparison. There are two plots
        for them and each of them use different colors to differentiate the data with different labels.

        Args:
            data:   the input data
            gt_y:   ground truth to the data
            pred_y: predicted results to the data
        """
        assert data.shape[0] == gt_y.shape[0]
        assert data.shape[0] == pred_y.shape[0]

        plt.figure()

        plt.subplot(1, 2, 1)
        plt.title('Ground Truth', fontsize=18)

        for idx in range(data.shape[0]):
            if gt_y[idx] == 0:
                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.subplot(1, 2, 2)
        plt.title('Prediction', fontsize=18)

        for idx in range(data.shape[0]):
            if pred_y[idx] == 0:
                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.show()

    def forward(self, inputs):
        """ Implementation of the forward pass.
        Arg:
            inputs : the input data with shape (2, BatchSize)
        """
        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]
        W3 = self.parameters["W3"]
        b3 = self.parameters["b3"]

        Z1 = np.dot(W1, inputs) + b1
        A1 = sigmoid(Z1)  # Sigmoid
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)  # Sigmoid
        Z3 = np.dot(W3, A2) + b3
        A3 = sigmoid(Z3)  # Sigmoid

        self.cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}
        return A3

    def backward(self):
        """ Implementation of the backward pass.
        It should utilize the saved loss to compute gradients and update the network all the way to the front.
        """
        n = self.inputs.shape[0] # number of examples
        X = self.inputs
        Y = self.labels

        W1 = self.parameters["W1"]
        W2 = self.parameters["W2"]
        W3 = self.parameters["W3"]
        A1 = self.cache["A1"]
        A2 = self.cache["A2"]
        A3 = self.cache["A3"]
        Z1 = self.cache["Z1"]
        Z2 = self.cache["Z2"]
        Z3 = self.cache["Z3"]

        # for i in range(A3.shape[1]):
        #     if A3[:, i] < 1e-4:
        #         A3[:, i] = 1e-4
        #     elif A3[:, i] > 0.9999:
        #         A3[:, i] = 0.9999

        # L = - (np.divide(Y, A3) - np.divide(1 - Y, 1 - A3))
        L = self.cost # L = (A3 - Y)
        temp_s = sigmoid(Z3)
        dZ3 = L * der_sigmoid(temp_s)  # Sigmoid (back propagation)

        dW3 = 1 / n * np.dot(dZ3, A2.T)
        db3 = 1 / n * np.sum(dZ3, axis=1, keepdims=True)

        dA2 = np.dot(W3.T, dZ3)
        temp_s = sigmoid(Z2)
        dZ2 = dA2 * der_sigmoid(temp_s)  # Sigmoid (back propagation)

        dW2 = 1 / n * np.dot(dZ2, A1.T)
        db2 = 1 / n * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(W2.T, dZ2)
        temp_s = sigmoid(Z1)
        dZ1 = dA1 * der_sigmoid(temp_s)  # Sigmoid (back propagation)

        dW1 = 1 / n * np.dot(dZ1, X.T)
        db1 = 1 / n * np.sum(dZ1, axis=1, keepdims=True)

        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}
        return grads

    def train(self, inputs, labels):
        """ The training routine that runs and update the model.

        Args:
            inputs: the training data used in the model.
                    The shape is expected to be [2, BatchSize].
            labels: the ground truth of correspond to input data.
        """
        # make sure that the amount of data and label is match
        assert inputs.shape[1] == labels.shape[1]
        self.inputs = inputs.copy()
        self.labels = labels.copy()

        for epochs in range(self.num_step):
            # operation in each training step:
            #   1. forward passing
            #   2. compute loss
            #   3. propagate gradient backward to the front
            #   4. update weights
            A3 = self.forward(inputs)
            self.cost, cost_value = compute_cost(A3, labels)
            grads = self.backward()
            self.parameters = update_parameters(self.parameters, grads, self.learning_rate)

            if epochs % self.print_interval == 0:
                print('Epoch {} loss : {}'.format(epochs, cost_value))

        print('Training finished')
        # self.test(inputs, labels)

    def test(self, inputs, labels):
        """ The testing routine that run forward pass and report the accuracy.

        Args:
            inputs: the testing data. One or several data samples are both okay.
                The shape is expected to be [2, BatchSize].
            labels: the ground truth correspond to the inputs.
        """
        n = inputs.shape[1] # number of examples
        result = self.forward(inputs)
        # set print precision
        with np.printoptions(precision=8, suppress=True):
            print(np.reshape(np.squeeze(result),(n,1)))

        # Calculate test error
        error = abs(result - labels)
        error_sum = np.sum(error[:])
        error_avg = error_sum / n

        # Calculate test accuracy and print it.
        print('accuracy: %.2f' % ((1 - error_avg)*100) + '%')
        print('')

        return (1 - error_avg)*100

if __name__ == '__main__':
    data, label = GenData.fetch_data('Linear', 70)
    net = SimpleNet(50, num_step=5000)

    # data, label = GenData.fetch_data('XOR', 70)
    # net = SimpleNet(50, num_step=5000)

    net.train(data.T, label.T)
    net.test(data.T, label.T)

    pred_result = np.round(net.forward(data.T))
    SimpleNet.plot_result(data, label, pred_result.T)