import numpy as np
from data_prep import features, targets


def main():
    nn = NeuralNetwork()
    nn.train(features, targets)
    nn.test(features, targets)


class NeuralNetwork:
    def __init__(self, activate_hidden_layers=True, hidden_layers=(2,),
                 epochs=1000, learning_rate=0.1,
                 type_loss_function="CE", type_activation_hidden="sigmoid", graph=True):
        """
        [describe]: initialize hyper parameters
        activate_hidden_layers: boolean, The architecture of NN has hidden layers or not
        hidden_layers: tuple of nurons in each hidden layer in order
        epochs: number of iterations for NN to learn
        learning_rate: scaling the gradient descent to improve the stability of learning
        type_loss_function: the type of loss function that used as measure of the error
        and used in updating the weights
        graph: as debug mode
        """
        np.random.seed(42)
        self.__activate_hidden_layers = activate_hidden_layers
        self.__hidden_layers = hidden_layers
        self.__epochs = epochs
        self.__learning_rate = learning_rate
        self.__type_loss_function = type_loss_function  # 'CE' / 'SE'
        self.__type_activation_hidden = type_activation_hidden  # "sigmoid"
        self.__graph = graph
        self.__n_records = None
        self.__n_features = None
        self.__n_classes = None
        self.__features = None
        self.__targets = None
        self.__weights = []
        self.__in_layers = []
        self.__out_layers = []
        self.__error_terms = []

    def __extract_metadata(self, features, targets):
        self.__features = features
        self.__targets = targets
        self.__n_records, self.__n_features = self.__features.shape
        if len(self.__targets.shape) == 1:
            self.__n_classes = 1
        else:
            _, self.__n_classes = self.__targets.shape

    def __addWeights(self):
        # intialize number of the hidden layers if there is no any hidden layer
        n_hidden_layers = 0
        # If there are hidden layers
        if(self.__activate_hidden_layers):
            n_hidden_layers = len(self.__hidden_layers)

        self.__weights = [None for _ in range(n_hidden_layers + 1)]
        n_rows = self.__n_features
        # It will iterate if there are hidden layers
        # if self.__activate_hidden_layers is true
        for i in range(len(self.__weights) - 1):
            # The weights connect the (input layer or hidden layer) with other hidden layers
            self.__weights[i] = np.random.normal(loc=0, scale=np.power(self.__n_features, -0.5),
                                                 size=(n_rows, self.__hidden_layers[i]))
            # update number of rows of the matrix weight
            n_rows = self.__hidden_layers[i]
        # The weights connect the layer(input or hidden) with output layer
        self.__weights[-1] = np.random.normal(loc=0, scale=np.power(self.__n_features, -0.5),
                                              size=(n_rows, self.__n_classes))

    def __activation(self, activation_function, x):
        return activation_function(x)

    def __sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def __sigmoid_prime(self, x):
        return self.__sigmoid(x) * (1-self.__sigmoid(x))

    def __feedForward(self, x):
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
            # linear combinations
            self.__in_layers[i] = np.dot(in_signal, self.__weights[i])
            self.__out_layers[i] = self.__activation(
                self.__sigmoid, self.__in_layers[i])
            # update in_signal
            in_signal = self.__out_layers[i]
        output = self.__out_layers[-1]
        return output

    def __lossError(self, loss_function, y, output):
        return loss_function(y, output)

    def __square_error(self, y, output):
        return (y-output)**2

    def __cross_entropy(self, y, output):
        return -y * np.log(output) - (1-y)*np.log(1-output)

    def __calculate_out_error_term(self, y, output):
        # It depends on the type of loss function and the activation function
        # I used activation function sigmoid here assuming, I have binary class, for simplicity
        # I will update it to be more general
        y = np.array(y).reshape(-1, self.__n_classes)
        error = y - output

        if self.__type_loss_function == 'CE':  # Cross Entopy
            return error
        elif self.__type_loss_function == 'SE':  # Square Error
            return error * output * (1 - output)

    def __calculate_error_terms(self, y, output):
        n_weights = len(self.__weights)
        self.__error_terms = [None for _ in range(n_weights)]
        # Calculate the first error term related to the output
        self.__error_terms[-1] = self.__calculate_out_error_term(y, output)
        for i in range(n_weights - 2, -1, -1):

            error = np.matmul(
                self.__error_terms[i + 1], self.__weights[i + 1].T)
            activation_prime = None
            if self.__type_activation_hidden == 'sigmoid':
                out_layer = self.__out_layers[i]
                activation_prime = out_layer * (1 - out_layer)
            self.__error_terms[i] = error * activation_prime

    def __updateWeights(self, x):
        n_weights = len(self.__weights)
        in_signal = None
        for i in range(n_weights-1, -1, -1):
            if i == 0:
                in_signal = x
            else:
                in_signal = self.__out_layers[i-1]
            error_term = self.__error_terms[i]

            # self.__weights[i] += self.__learning_rate * \
            #     np.matmul(in_signal.T, error_term)
            self.__weights[i] += self.__learning_rate * \
                np.matmul(in_signal[:, None], error_term)
        pass

    def train(self, features, targets):
        last_loss = None
        #########################################################################################
        # extract metadata from the data to train the NN
        self.__extract_metadata(features, targets)
        #########################################################################################
        # Add the weights for the NN
        self.__addWeights()
        for e in range(self.__epochs):
            for (x, y) in zip(features, targets):

                #########################################################################################
                # Feed forward process
                output = self.__feedForward(x)
                #########################################################################################
                # Back propagation process
                #########################################################################################
                # Caluclate the error_terms
                self.__calculate_error_terms(y, output)
                #########################################################################################
                # Update weights
                self.__updateWeights(x)

            if (e/10) % 10 == 0:
                loss_function = None
                if self.__type_loss_function == 'CE':  # Cross Entopy
                    loss_function = self.__cross_entropy
                elif self.__type_loss_function == 'SE':  # Square Error
                    loss_function = self.__square_error

                output = self.__feedForward(features)
                targets = np.array(targets)[:, None]
                loss = np.mean(self.__lossError(
                    loss_function, targets, output))
                print(f"Train loss: {loss}", end='    ')
                if last_loss and loss > last_loss:
                    print("Loss increasing!")
                else:
                    print()
                predictions = output >= 0.5

                accuracy = np.mean(predictions == targets)
                print(f"Train accuracy: {accuracy :0.3f}")
                last_loss = loss
        pass

    def test(self, features_test, targets_test):
        print()
        output = self.__feedForward(features_test)
        predictions = output >= 0.5
        accuracy = np.mean(predictions == targets_test)
        print(f"Test accuracy: {accuracy :0.3f}")

    def plot_boundary(self):
        pass


if __name__ == "__main__":
    main()
