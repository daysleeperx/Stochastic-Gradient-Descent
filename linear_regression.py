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

    @classmethod
    def from_csv(cls, file_path):
        """Generate model from csv file."""
        train = np.loadtxt(file_path, delimiter=",")

        return cls(train)


if __name__ == '__main__':
    lr = LinearRegression.from_csv('data/test.csv')
    print(lr.train.shape)


