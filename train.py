import numpy as np
from data_prep import features_train, targets_train, features_validation, targets_validation, features_test, targets_test
from sys import exit

visualize_mode = True
if visualize_mode:
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
                 epochs=1000, learning_rate=0.5, bias=True, validation=True,
                 jumps=10, type_loss_function="CE", type_activation_hidden="sigmoid",
                 debug=False, graph=True, random_seed=42):
        """
        [describe]: initialize hyper parameters
        activate_hidden_layers: boolean, The architecture of NN has hidden layers or not
        hidden_layers: tuple of nurons in each hidden layer in order
        epochs: number of iterations for NN to learn
        learning_rate: scaling the gradient descent to improve the stability of learning
        bias: Add bias (shift the boundary descition) or Not
        validation: if activated the algorithm early stopping
        will run to tune the epochs and prevent the trainning from overfitting
        type_loss_function: the type of loss function that used as measure of the error
        and used in updating the weights
        graph: as debug mode

        """
        if debug:
            np.random.seed(random_seed)
        self.__activate_hidden_layers = activate_hidden_layers
        self.__hidden_layers = hidden_layers
        self.__epochs = epochs
        self.__learning_rate = learning_rate
        self.__bias = bias  # add bias or not
        self.__validation = validation  # to tune the epochs
        self.__jumps = jumps  # jumps on epochs as sensetive tunning for epochs
        self.__type_loss_function = type_loss_function  # 'CE' / 'SE'
        self.__type_activation_hidden = type_activation_hidden  # "sigmoid"
        self.__graph = graph
        self.__debug = debug

        self.__n_records = None
        self.__n_features = None
        self.__n_classes = None
        #######################################################################################
        # Initialize flag to warn you if the error is increasing
        self.__last_loss_train = None
        self.__last_loss_validation = None
        #######################################################################################

        self.__weights = []
        self.__in_layers = []
        self.__out_layers = []
        self.__error_terms = []

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

        n_rows = self.__n_features + self.__bias
        # It will iterate if there are hidden layers
        # if self.__activate_hidden_layers is true
        for i in range(len(self.__weights) - 1):
            # The weights connect the (input layer or hidden layer) with other hidden layers
            self.__weights[i] = np.random.normal(loc=0, scale=np.power(self.__n_features, -0.5),
                                                 size=(n_rows, self.__hidden_layers[i]))
            # update number of rows of the matrix weight
            n_rows = self.__hidden_layers[i] + + self.__bias
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
            # Add bias to in_signal if its true
            if self.__bias:
                if in_signal.ndim == 1:
                    m_rows = 1
                    in_signal = np.r_[in_signal, np.ones(m_rows)]
                else:
                    m_rows = in_signal.shape[0]
                    in_signal = np.c_[in_signal, np.ones(m_rows)]

            # linear combinations
            self.__in_layers[i] = np.dot(in_signal, self.__weights[i])
            self.__out_layers[i] = self.__activation(
                self.__sigmoid, self.__in_layers[i])
            # update in_signal
            in_signal = self.__out_layers[i]
        output = self.__out_layers[-1]
        return output

    def __backpropagation(self, x, y, output):
        # Caluclate the error_terms
        self.__calculate_error_terms(y, output)
        #########################################################################################
        # Update weights
        self.__updateWeights(x)

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
        y = np.array(y).reshape(output.shape)
        error = y - output
        if error.ndim == 1:
            error = error.reshape(1, -1)
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
            w = self.__weights[i + 1]
            if self.__bias:
                # Dont take last row in calculations
                w = w[:-1, :]
            error = np.matmul(self.__error_terms[i + 1], w.T)
            activation_prime = None
            if self.__type_activation_hidden == 'sigmoid':
                out_layer = self.__out_layers[i]
                activation_prime = out_layer * (1 - out_layer)
            else:
                print("Need other activation function")
                exit(1)
            self.__error_terms[i] = error * activation_prime

    def __updateWeights(self, x):
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

            error_term = self.__error_terms[i]
            gradient = None
            if in_signal.ndim == 2:
                gradient = np.matmul(in_signal.T, error_term)
            elif in_signal.ndim == 1:
                gradient = np.matmul(in_signal[:, None], error_term)
            del_w = self.__learning_rate * gradient
            self.__weights[i] += del_w

        pass

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
        if accuracy_validation == 1:
            print("accuracy_validation = 1")
            return True

        if epoch_step % (self.__jumps) == 0:
            if self.__last_loss_validation and loss_validation > self.__last_loss_validation:
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
        # Add the weights for the NN
        self.__addWeights()
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
            for (x, y) in zip(features_train, targets_train):
                # Feed Forward process
                output = self.__feedForward(x)
                # Backpropagation process
                self.__backpropagation(x, y, output)

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
                print("\n========== Epoch", e, "==========")

                #########################################################################################
                # Display the results
                self. __Display_results(
                    self.__last_loss_train, loss_train, accuracy_train, "Train")
                self. __Display_results(
                    self.__last_loss_validation, loss_validation, accuracy_validation, "Validation")

            #########################################################################################
            if self.__early_stopping(accuracy_validation, loss_validation, e):
                break
            # Update the flag
            self.__last_loss_validation = loss_validation
            self.__last_loss_train = loss_train
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

        plt.contour(X1, X2, Z, levels=[0])
        ax.set_title('Boundary line')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')

        # add points
        self.__plot_points(features, targets, title + " Descition boundary")
        # plt.xlim(x1_start + scale_factor * x1_start,
        #          x1_stop + scale_factor * x1_stop)
        # plt.ylim(x2_start + scale_factor * x2_start,
        #          x2_stop + scale_factor * x2_stop)
        # plt.ylim(x2_start, x2_stop)
        # plt.show()

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
