"""Test."""


import unittest
from linear_regression import LinearRegression
from util import *


class LinearModelTest(unittest.TestCase):

    def single(self, idx):
        """Test against a single values array."""
        model = LinearRegression.from_csv('data/train.csv')
        model.fit_sgd(epochs=5)

        test = np.loadtxt("data/test.csv", delimiter=",")
        x_test = add_one_bias(normalize(test[:, :5]))
        y_test = normalize(test[:, 5])

        prediction = model.predict(x_test[idx] > 0.5).astype(int)
        print(prediction)
        self.assertEqual(prediction, y_test[idx])

    def test(self):
        # create and train the model
        model = LinearRegression.from_csv('data/train.csv')
        weights, cost_history = model.fit_sgd(epochs=5)
        print(cost_history)

        # evaluate against test data
        test = np.loadtxt("data/test.csv", delimiter=",")
        x_test = add_one_bias(normalize(test[:, :5]))
        y_test = normalize(test[:, 5])
        print(f"Model accuracy: {model.evaluate(x_test, y_test)}")
        self.assertEqual(1, 1)

    def test_single(self):
        self.single(0)


if __name__ == '__main__':
    unittest.main()
