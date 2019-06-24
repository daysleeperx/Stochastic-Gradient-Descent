"""Linear Regression Model."""


from util import *
import matplotlib.pyplot as plt


class LinearRegression:
    """
    Represent multivariate linear regression,
    where multiple correlated dependent variables are predicted,
    rather than a single scalar variable.
    """

    def __init__(self, x, y, learning_rate=0.01):
        """
        Class constructor.

        :param x: training set as numpy array
        :param y: labels as numpy array
        :param learning_rate: learning rate, which determines
        to what extent newly acquired information overrides old information
        """
        self._x = x
        self._y = y
        self._learning_rate = learning_rate
        self._weights = self.random_weights(x.shape[1])

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    @staticmethod
    def random_weights(size):
        return np.random.randn(size)

    @classmethod
    def from_csv(cls, file_path) -> 'LinearRegression':
        """
        Generate model from csv file.

        This method assumes, that the last column of training data array
        represents the labels.

        :param file_path: str
        :return: model as LinearRegression object
        """
        train = np.loadtxt(file_path, delimiter=",")

        x = add_one_bias(normalize(train[:, :-1]))
        y = normalize(train[:, -1])

        return cls(x, y)

    def predict(self, x):
        """
        Return prediction using current weights.

        :param x: test data as numpy array
        :return: predictions as int
        """
        return np.dot(x, self._weights)

    def loss_sse(self, x, y):
        """
        Return cost using the SSE(Sum of Squared Errors) equation.

        :param x: train data as numpy array
        :param y: labels as numpy array
        :return: cost as int
        """
        predictions = self.predict(x)
        return np.sum(np.square(predictions - y))

    def fit_sgd(self, epochs=10):
        """
        Stochastic Gradient Descent - randomly select a sample to evaluate gradient,
        make step towards minimizing the loss function.

        :param epochs: number of training epochs
        :return: updated weights
        """
        cost_history = np.zeros(epochs)
        m = len(self._x)

        for epoch in range(epochs):
            cost = 0
            for i in range(m):
                rand_idx = random_int(m)
                sample, label = self._x[rand_idx], self._y[rand_idx]

                prediction = self.predict(sample)

                self.__update_weights(label - prediction, sample)
                cost += self.loss_sse(sample, label)
            cost_history[epoch] = cost

        return self._weights, cost_history

    def __update_weights(self, error, sample):
        self._weights += sample * self._learning_rate * error

    def evaluate(self, x, y, threshold=0.5):
        """
        Return accuracy of model for binary classification, where
        accuracy is calculated as follows:

        (true positive + true negative) divided by total number of examples

        :param x: test data as numpy array
        :param y: labels as numpy array
        :param threshold:
        :return: accuracy as int
        """
        predictions = (self.predict(x) > threshold).astype(int)
        correct_predictions = (predictions == y).astype(int).sum()
        return correct_predictions / y.shape[0]

    @staticmethod
    def plot_loss(cost_history):
        x = np.arange(0, cost_history.shape[0])
        y = cost_history

        plt.xlabel('Epochs')
        plt.ylabel('Loss function')

        plt.scatter(x, y, c='b', marker='o')
        plt.show()

