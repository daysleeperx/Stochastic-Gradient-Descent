"""Linear Regression Model."""


import numpy as np


class LinearRegression:
    """
    Represent multivariate linear regression,
    where multiple correlated dependent variables are predicted,
    rather than a single scalar variable
    """

    def __init__(self, train, learning_rate=0.01):
        """
        Class constructor.

        :param train: initial training set
        :param learning_rate: learning rate, which determines
        to what extent newly acquired information overrides old information
        """
        self._train = train
        self._learning_rate = learning_rate

        self._weights = self.random_weights(train.shape[1])

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, value):
        self._train = value

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    @staticmethod
    def random_weights(length: int):
        return np.random.uniform(low=0, high=1, size=length)

    @classmethod
    def from_csv(cls, file_path):
        """Generate model from csv file."""
        train = np.loadtxt(file_path, delimiter=",")

        return cls(train)

    def predict(self, train_x):
        """
        Return prediction using current weights.

        :param train_x:
        :return:
        """
        return np.dot(train_x, self._weights)

    def loss_sse(self, train_x, expected):
        """
        Return cost using the SSE(Sum of Squared Errors) equation.

        :param train_x: sample row(s) of x values
        :param expected: actual values of y
        :return: cost as int
        """
        predictions = self.predict(train_x)
        return np.sum(np.square(predictions - expected))

    def fit_sgd(self, epochs=100, cost=0.0):
        """
        Stochastic Gradient Descent - randomly select a sample to evaluate gradient,
        make step towards minimizing the loss function.

        :param epochs: number of training epochs
        :param cost: initial cost value
        :return: updated weights
        """
        cost_history = np.zeros(epochs)
        m = self.train.shape[0]

        for epoch in range(epochs):
            for i in range(m):
                rand_ind = np.random.randint(0, m)
                # TODO:
                pass


if __name__ == '__main__':
    lr = LinearRegression.from_csv('data/test.csv')
    print(lr.train.shape)

    np.random.randn()
