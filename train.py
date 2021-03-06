#!python
########################################################
# Train neural network file
# Author: Muhammed El-Yamani
# muhammedelyamani92@gmail.com
# September 2020
########################################################
import numpy as np
from data_prep import features_train, targets_train, features_validation, targets_validation, features_test, targets_test
from sys import exit
from scipy import stats

import matplotlib.pyplot as plt


def main():
    nn = NeuralNetwork()

    nn.train(features_train, targets_train,
             features_validation, targets_validation)

    nn.test(features_test, targets_test)

    nn.plot_boundary(features_test, targets_test, 'Test')

    nn.plot_boundary(features_train, targets_train, 'Train')

    plt.show()

    exit(0)


class NeuralNetwork:

    def __init__(self, activate_hidden_layers=True, hidden_layers=(2,),
                 epochs=100, batch_size=32, learning_rate=0.95, beta=0.9, activate_early_stopping=True,
                 activate_regularization=True, regularization_type='L2', reg_factor=0.01, enhance_weights=False,
                 dropout_activate=False, dropout_bias=True, prob_dropout_input=0.2, prob_dropout_hidden=0.5,
                 bias=True, jumps=1, type_loss_function="CE",
                 type_activation_hidden="relu", ELU_factor=1,
                 debug=True, graph=True, random_seed=42, display_weights=False, display_stat_layers=False):
        """
        [describe]: initialize hyper parameters

        @param [activate_hidden_layers]: boolean, The architecture of NN has hidden layers or not

        @param [hidden_layers]: tuple of nurons in each hidden layer in order

        @param [epochs]: number of iterations for NN to learn

        @param [batch_size]: 
            1  -> SGD (Update weights over each training example)
           -1  -> BGD (Update weights At end each epoch)
            N  -> MBGD (Update weights after each N training examples)

        @param [learning_rate]: scaling the gradient descent to improve the stability of learning

        @param [beta]: hyper parameter beta for momentum between (0,1) recommended 0.9

        @param [activate_early_stopping]: if activated the algorithm early stopping
        will run to tune(stop) the epochs and prevent the trainning from overfitting

        @param [activate_regularization]: To penalize high weights

        @param [regularization_type]: Two types-> L1 / L2
        L1: Feature selection
        L2: Training the Model

        @param [reg_factor]: How much you want to penalize large weights
        if this factor is large you want to penalize so much and vice versa

        @param [enhance_weights]: if it is true, instead of weight decay to decrease the weights,
        the weigts will increase, but it will increase the model for overfitting. 
        it is active if  activate_regularization is true

        @param [prob_dropout_input]: Probabilty that randomly ignore(dropout) nodes in the input layer

        @param [prob_dropout_hidden]: Probabilty that randomly ignore(dropout) nodes in the hidden layer

        @param [bias]: Add bias (shift the boundary descition) or Not

        @param [jumps]: jumps on epochs as sensetive tunning for epochs

        @param [type_loss_function]: the type of loss function that used as measure of the error
        and used in updating the weights

        @param [type_activation_hidden]: The activation function on the hidden layers

        @param [ELU_factor]: hyper parameter >= 0 used in ELU activation function
        -> Notice if ELU_factor is 0 then it becomes Relu activation fuction

        @param [debug]: debug or not

        @param [graph]: To graph the graphs

        @param [random_seed]: initalize random
        """

        if debug:
            np.random.seed(random_seed)
        self.__activate_hidden_layers = activate_hidden_layers
        self.__hidden_layers = hidden_layers
        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__learning_rate = learning_rate
        self.__activate_early_stopping = activate_early_stopping  # to tune the epochs
        self.__activate_regularization = activate_regularization  # To penalize high weights
        self.__regularization_type = regularization_type  # L1 / L2
        self.__reg_factor = reg_factor
        self.__enhance_weights = enhance_weights
        #######################################################################################
        # Gradient Accumulation by Momentum
        self.__gradMomentum = []  # intial all items by zero
        # hyper parameter beta for momentum between (0,1)
        self.__beta = beta
        #######################################################################################
        # To decrease the overfitting and enhance the training
        self.__dropout_activate = dropout_activate
        self.__dropout_bias = dropout_bias
        self.__prob_dropout_input = prob_dropout_input
        self.__prob_dropout_hidden = prob_dropout_hidden
        #######################################################################################
        self.__bias = bias  # add bias or not
        self.__jumps = jumps  # jumps on epochs as sensetive tunning for epochs
        self.__type_loss_function = type_loss_function  # 'CE' / 'SE'
        self.__type_activation_hidden = type_activation_hidden  # "sigmoid"
        self.__ELU_factor = ELU_factor
        self.__graph = graph
        self.__debug = debug
        self.__display_weights = display_weights
        self.__display_stat_layers = display_stat_layers
        self.__n_records = None
        self.__n_features = None
        self.__n_classes = None
        #######################################################################################
        # Initialize flag to warn you if the error is increasing
        self.__last_loss_train = None
        self.__last_loss_validation = None
        #######################################################################################
        self.__weights = []
        self.__del_weights = []
        self.__in_layers = []
        self.__out_layers = []
        self.__error_terms = []
        self.__dropout_layers = []  # Add dropout layers

    def __extract_metadata(self, features, targets):
        self.__n_records, self.__n_features = features.shape
        if len(targets.shape) == 1:
            self.__n_classes = 1
        else:
            _, self.__n_classes = targets.shape

    def __addWeights(self):
        # intialize number of the hidden layers if there is no any hidden layer
        n_hidden_layers = 0
        # If there are hidden layers
        if(self.__activate_hidden_layers):
            n_hidden_layers = len(self.__hidden_layers)

        self.__weights = [None for _ in range(n_hidden_layers + 1)]
        self.__del_weights = [None for _ in range(n_hidden_layers + 1)]

        n_rows = self.__n_features + self.__bias
        # It will iterate if there are hidden layers
        # if self.__activate_hidden_layers is true
        for i in range(len(self.__weights) - 1):
            # The weights connect the (input layer or hidden layer) with other hidden layers
            self.__weights[i] = np.random.normal(loc=0, scale=np.power(self.__n_features, -0.5),
                                                 size=(n_rows, self.__hidden_layers[i]))
            self.__del_weights[i] = np.zeros(self.__weights[i].shape)
            # update number of rows of the matrix weight
            n_rows = self.__hidden_layers[i] + self.__bias
        # The weights connect the layer(input or hidden) with output layer
        self.__weights[-1] = np.random.normal(loc=0, scale=np.power(self.__n_features, -0.5),
                                              size=(n_rows, self.__n_classes))
        self.__del_weights[-1] = np.zeros(self.__weights[-1].shape)

    def __construct_NN(self):
        # Add the weights for the NN
        self.__addWeights()
        self.__gradMomentum = self.__del_weights.copy()

    def __decision_dropout(self, probability, num):
        return np.random.choice([0, 1], size=num, p=[probability, 1 - probability])

    def __add_dropout_layers(self):
        if not self.__dropout_activate:
            return

        # initialize self.__dropout_layers
        n_weights = len(self.__weights)
        self.__dropout_layers = [None for _ in range(n_weights)]

        # rows of weights
        for i, weight in enumerate(self.__weights):
            # number of nodes in each layer
            number_nodes = weight.shape[0]
            if self.__bias and not self.__dropout_bias:
                number_nodes -= 1
            # assign the values of self.__dropout_layers
            if i == 0:
                # the input layer
                dropout_layer = self.__decision_dropout(
                    self.__prob_dropout_input, number_nodes)
            else:
                # the hidden layers
                dropout_layer = self.__decision_dropout(
                    self.__prob_dropout_hidden, number_nodes)
            if self.__bias and not self.__dropout_bias:
                dropout_layer = np.r_[dropout_layer, np.ones(1)]

            self.__dropout_layers[i] = dropout_layer

    def __softmax(self, L):
        # take the exponitial for array L
        expL = np.exp(L)
        # take the sum of exponitial for array L
        sumExpL = np.sum(expL)
        probabilities = np.divide(expL, sumExpL)

        return probabilities

    def __sigmoid(self, x):
        # print("x in sigmoid function\n",x)
        return 1/(1+np.exp(-x))

    def __tanh(self, x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    def __elu(self, x):
        return np.where(x > 0, x, self.__ELU_factor * (np.exp(x) - 1))

    def __relu(self, x):
        return np.maximum(0, x)

    def __activation(self, layer_i, n_layers, x):
        """
        Calculate the activation function for input
        @param [layer_i]: layer i
        @param [n_layers]: number of layers
        @param [x]: input
        """
        if layer_i == n_layers - 1:
            # output layer
            if self.__n_classes == 1:
                # binary class
                self.__out_layers[layer_i] = self.__sigmoid(x)
            else:
                # multi classes
                self.__out_layers[layer_i] = self.__softmax(x)
        else:
            # hidden layers
            if self.__type_activation_hidden == 'sigmoid':
                self.__out_layers[layer_i] = self.__sigmoid(x)
            elif self.__type_activation_hidden == 'relu':
                self.__out_layers[layer_i] = self.__relu(x)
            elif self.__type_activation_hidden == 'tanh':
                self.__out_layers[layer_i] = self.__tanh(x)
            elif self.__type_activation_hidden == 'elu':
                self.__out_layers[layer_i] = self.__elu(x)

            else:
                print('In hidden layers')
                print(
                    f"There is no definition for activation function {self.__type_activation_hidden}")
                exit(2)

    def __feedForward(self, x, train=False):
        """
        [describe] Take input (data or signal) and return the output
        Make the whole calculations feedforward in the NN
        [param]       x: data point or all the features
        [return] output: probabilities prediction for the input data
        """
        output = None
        n_weights = len(self.__weights)
        self.__in_layers = [None for _ in range(n_weights)]
        self.__out_layers = [None for _ in range(n_weights)]
        # initialize in_signal that inputs into the the neurons
        in_signal = x
        for i in range(n_weights):

            # Add bias to in_signal if its true
            if self.__bias:
                if in_signal.ndim == 1:
                    m_rows = 1
                    in_signal = np.r_[in_signal, np.ones(m_rows)]
                else:
                    m_rows = in_signal.shape[0]
                    in_signal = np.c_[in_signal, np.ones(m_rows)]
            if self.__dropout_activate and train:
                in_signal *= self.__dropout_layers[i]
            # linear combinations
            self.__in_layers[i] = np.dot(in_signal, self.__weights[i])
            # Apply the activation function
            self.__activation(i, n_weights, self.__in_layers[i])
            # update in_signal
            in_signal = self.__out_layers[i]
        output = self.__out_layers[-1]
        return output

    def __activation_hidden_prime(self, hidden_layer_i):
        """
        [describe] The derivative the choice activation function with respect input_activation
        @param [hidden_layer_i] used to get
        [output_activation] if not None used for performance
        [input_activation] if not None used if the output_activation not exist
        """
        output_hidden_activation = self.__out_layers[hidden_layer_i]
        if self.__type_activation_hidden == 'sigmoid':
            return output_hidden_activation * (1 - output_hidden_activation)
        elif self.__type_activation_hidden == 'relu':
            return output_hidden_activation >= 0
        elif self.__type_activation_hidden == 'tanh':
            return (1 - np.power(output_hidden_activation, 2))
        elif self.__type_activation_hidden == 'elu':
            x = self.__in_layers[hidden_layer_i]
            return np.where(x > 0, 1, self.__ELU_factor * np.exp(x))
        else:
            print('In hidden layers')
            print(
                f"There is no definition for derivative activation function {self.__type_activation_hidden}")
            exit(4)
        pass

    def __calculate_out_error_term(self, y, output):
        # It depends on the type of loss function and the activation function
        # I used activation function sigmoid here assuming, I have binary class, for simplicity
        # I will update it to be more general
        y = np.array(y).reshape(output.shape)
        if self.__n_classes == 1:
            # Using sigmoid activation
            error = y - output
            if error.ndim == 1:
                error = error.reshape(1, -1)
            if self.__type_loss_function == 'CE':  # Cross Entopy
                return error
            elif self.__type_loss_function == 'SE':  # Square Error
                return error * output * (1 - output)
        else:
            # multi classes
            print("Needs softmax function")
            exit(3)

    def __calculate_error_terms(self, y, output):
        n_weights = len(self.__weights)
        self.__error_terms = [None for _ in range(n_weights)]
        # Calculate the first error term related to the output
        self.__error_terms[-1] = self.__calculate_out_error_term(y, output)
        for i in range(n_weights - 2, -1, -1):
            w = self.__weights[i + 1]
            if self.__bias:
                # Dont take last row in calculations
                w = w[:-1, :]
            error = np.matmul(self.__error_terms[i + 1], w.T)
            activation_prime = self.__activation_hidden_prime(i)
            self.__error_terms[i] = error * activation_prime

    def __updateWeights(self, x, size_data):
        n_weights = len(self.__weights)
        in_signal = None
        for i in range(n_weights-1, -1, -1):
            if i == 0:
                in_signal = x
            else:
                in_signal = self.__out_layers[i-1]

            # Add bias to in_signal if its true
            if self.__bias:
                if in_signal.ndim == 1:
                    m_rows = 1
                    in_signal = np.r_[in_signal, np.ones(m_rows)]
                else:
                    m_rows = in_signal.shape[0]
                    in_signal = np.c_[in_signal, np.ones(m_rows)]

            if self.__dropout_activate:
                in_signal *= self.__dropout_layers[i]

            error_term = self.__error_terms[i]
            gradient_descent = None
            if in_signal.ndim == 2:
                gradient_descent = np.matmul(in_signal.T, error_term)
            elif in_signal.ndim == 1:
                gradient_descent = np.matmul(in_signal[:, None], error_term)

            if self.__activate_regularization:
                if self.__regularization_type == 'L2':
                    derivative_reg_term = self.__reg_factor * self.__weights[i]
                    # update the gradient_descent because of the regularization
                elif self.__regularization_type == 'L1':
                    derivative_reg_term = np.array(
                        np.zeros(shape=self.__weights[i].shape))
                    derivative_reg_term[self.__weights[i] > 0] = 1
                    derivative_reg_term[self.__weights[i] < 0] = -1
                    derivative_reg_term *= self.__reg_factor
                    # update the gradient_descent because of the regularization
                if self.__enhance_weights:
                    # motivate the  weights to increase
                    gradient_descent += derivative_reg_term
                else:
                    # The normal regularization
                    # punch increasing weights to decrease
                    gradient_descent -= derivative_reg_term
            self.__gradMomentum[i] = self.__beta * self.__gradMomentum[i] +  gradient_descent
            del_w = self.__gradMomentum[i]
            self.__del_weights[i] += del_w
            if (self.__batch_size > 0 and size_data % (self.__batch_size) == 0) or (self.__batch_size <= 0 and size_data == self.__n_records):
                self.__weights[i] += self.__learning_rate * \
                    self.__del_weights[i] / self.__n_records
                self.__del_weights[i] *= 0

        pass

    def __backpropagation(self, x, y, output, size_data):
        # Caluclate the error_terms
        self.__calculate_error_terms(y, output)
        #########################################################################################
        # Update weights
        self.__updateWeights(x, size_data)

    def __regularization_term(self):
        if self.__activate_regularization:
            termsSum = 0
            for weight in self.__weights:
                if self.__regularization_type == 'L1':
                    term = np.abs(weight)
                elif self.__regularization_type == 'L2':
                    term = np.power(weight, 2)/2

                termsSum += np.sum(term)
            reg_term = self.__reg_factor * termsSum
            return reg_term
        else:
            return 0

    def __square_error(self, y, output):
        if self.__n_classes == 1:
            return (y-output)**2
        else:
            return

    def __cross_entropy(self, y, output):
        epsilon = 10**-10
        if self.__n_classes == 1:
            return -y * np.log(output + epsilon) - (1-y)*np.log(1-output + epsilon)
        else:
            return - np.dot(y, np.log(output + epsilon))

    def __lossError(self, loss_function, y, output):
        # under test

        return loss_function(y, output) + self.__regularization_term()

    def __loss_accuracy(self, features, targets):
        # Descide what loss function type you will use
        if self.__type_loss_function == 'CE':  # Cross Entopy
            loss_function = self.__cross_entropy
        elif self.__type_loss_function == 'SE':  # Square Error
            loss_function = self.__square_error
        #########################################################################################
        # Feed forward process to get Loss function
        outputs = self.__feedForward(features)
        targets = targets.reshape(outputs.shape)
        loss = np.mean(self.__lossError(
            loss_function, targets, outputs))

        predictions = outputs >= 0.5
        accuracy = np.mean(predictions == targets)

        return loss, accuracy

    def __Display_results(self, last_loss, loss, accuracy, category):
        print(f"{category} loss: {loss}", end='    ')
        if last_loss and loss > last_loss and last_loss >= 0:
            print(f"{category} loss increasing!")
        else:
            print()
        print(f"{category} accuracy: {accuracy :0.3f}")

    def __early_stopping(self, accuracy_validation, loss_validation, epoch_step):
        """This algorithm to stop The Neural Network from getting overfitting
            And becomes good fit
            It tunes the epochs
        """
        # if accuracy_validation == 1:
        #     print("accuracy_validation = 1")
        #     return True
        if loss_validation <= 1e-04:
            print("loss_validation reached almost zero!")
            return True
        if epoch_step % (self.__jumps) == 0:
            if self.__last_loss_validation and loss_validation >= self.__last_loss_validation:
                return True

        return False

    def train(self, features_train, targets_train,
              features_validation, targets_validation):
        #########################################################################################
        # Conver from pd.Datafram into numpy
        features_train, targets_train = np.array(
            features_train), np.array(targets_train)

        features_validation, targets_validation = np.array(
            features_validation), np.array(targets_validation)
        #########################################################################################
        # extract metadata from the data to train the NN
        self.__extract_metadata(features_train, targets_train)
        # #########################################################################################
        # Construct the neural network
        self.__construct_NN()
        #########################################################################################
        # initialize train_errors for visualization
        if self.__graph:
            train_errors = []
            validation_errors = []
        #########################################################################################
        # Iterate over the epochs
        for e in range(self.__epochs):
            #########################################################################################
            # Iterate over each data point
            for size_data, (x, y) in enumerate(zip(features_train, targets_train), start=1):
                # Add dropout layers if self.__dropout_activate
                self.__add_dropout_layers()
                # Feed Forward process
                output = self.__feedForward(x, train=True)

                if self.__display_stat_layers:
                    print("statistcs of the nerons")
                    print(f"X:\n{stats.describe(x)}")
                    for i in range(len(self.__in_layers)):
                        print(
                            f"h[{i+1}]\n{stats.describe(np.array(self.__in_layers[i]))}")
                        print(
                            f"a[{i+1}]\n{stats.describe(np.array(self.__out_layers[i]))}")
                # Backpropagation process
                self.__backpropagation(x, y, output, size_data)

            # Caluclate Loss function and accuracy over each epoch
            loss_train, accuracy_train = self.__loss_accuracy(
                features_train, targets_train)
            loss_validation, accuracy_validation = self.__loss_accuracy(
                features_validation, targets_validation)

            if self.__graph:
                train_errors.append(loss_train)
                validation_errors.append(loss_validation)

            #########################################################################################
            # Show resuluts over each tenths epochs
            if e % (self.__jumps) == 0:
                print("\n========== Epoch", e + 1, "==========")

                #########################################################################################
                # Display the results
                self. __Display_results(
                    self.__last_loss_train, loss_train, accuracy_train, "Train")
                self. __Display_results(
                    self.__last_loss_validation, loss_validation, accuracy_validation, "Validation")

                #########################################################################################
            if self.__activate_early_stopping:
                if self.__early_stopping(accuracy_validation, loss_validation, e):
                    break
            # Update the flag each jump
            if e % (self.__jumps) == 0:
                self.__last_loss_validation = loss_validation
                self.__last_loss_train = loss_train

        if self.__display_weights:
            for i, weight in enumerate(self.__weights):
                print(f"\n==========weight{i+1}==========")
                print(weight)

        # plot the errors
        if self.__graph:
            self.__plot_errors(train_errors, validation_errors)

    def test(self, features_test, targets_test):
        #########################################################################################
        # Conver from pd.Datafram into numpy
        features_test, targets_test = np.array(
            features_test), np.array(targets_test)
        #########################################################################################
        # Feed forward process to get accuracy results
        loss_test, accuracy_test = self.__loss_accuracy(
            features_test, targets_test)

        #########################################################################################
        # Display the results
        self. __Display_results(-1, loss_test, accuracy_test, "Test")
        print()

    def plot_boundary(self, features, targets, title, x1_n_values=100,
                      x2_n_values=100):

        condition = self.__n_features == 2
        if not condition:
            print(f"__n_features should be 2d to visualize")
            print(f"self.__n_features = {self.__n_features}")
            print(f"self.__bias = {self.__bias}")
            return

        fig = plt.figure(figsize=(6, 5))

        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        ax = fig.add_axes([left, bottom, width, height])

        scale_factor = 0.25
        x1_start, x1_stop = features['x1'].min(), features['x1'].max()
        x1_start += scale_factor * x1_start
        x1_stop += scale_factor * x1_stop

        x2_start, x2_stop = features['x2'].min(), features['x2'].max()
        x2_start += scale_factor * x2_start
        x2_stop += scale_factor * x2_stop

        x1_vals = np.linspace(x1_start, x1_stop, x1_n_values)
        x2_vals = np.linspace(x2_start, x2_stop, x2_n_values)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)

        m = x1_n_values
        n = x2_n_values
        Z = np.zeros((m, n))

        for i in range(m):
            for j in range(n):
                self.__feedForward(np.array([X1[i, j], X2[i, j]]))
                Z[i, j] = self.__in_layers[-1]

        contour = plt.contour(X1, X2, Z, levels=[0])
        # plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)

        ax.set_title('Boundary line')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')

        # add points
        self.__plot_points(features, targets, title + " Descition boundary")

    def __plot_points(self, features, targets, title):
        admitted = features[targets == 1]
        rejected = features[targets == 0]
        plt.scatter(admitted['x1'], admitted['x2'],
                    s=25, color='cyan', edgecolor='k')
        plt.scatter(rejected['x1'], rejected['x2'],
                    s=25, color='red', edgecolor='k')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(title)

        pass

    def __plot_errors(self, train_errors, validation_errors):
        plt.plot(train_errors, '-b', label='Train Error')
        plt.plot(validation_errors, '--r', label='Validation Error')

        plt.xlabel('Epochs')
        plt.ylabel('Error')

        plt.title('Train vs validation error')
        plt.legend()


if __name__ == "__main__":
    main()
