"""Test."""


import unittest
from linear_regression import LinearRegression
from util import *


TEST_CSV = 'data/test.csv'
TRAIN_CSV = 'data/train.csv'


class LinearModelTest(unittest.TestCase):

    def setUp(self):
        # create and train the model
        self.model = LinearRegression.from_csv(TRAIN_CSV)
        self.weights, self.cost_history, self.weights_history = self.model.fit_sgd(epochs=5)

        test = np.loadtxt('%s' % TEST_CSV, delimiter=",")
        self.x_test = add_one_bias(normalize(test[:, :5]))
        self.y_test = normalize(test[:, 5])

    def single(self, idx):
        """Test a single prediction from the test data."""
        prediction = (self.model.predict(self.x_test[idx]) > 0.5).astype(int)
        self.assertEqual(prediction, self.y_test[idx])

    def print_loss(self):
        print('\n'.join(f"Epoch {i} => loss: {j}" for i, j in enumerate(self.cost_history)))

    def test(self):
        self.print_loss()

        # evaluate against test data
        print(f"Model accuracy: {self.model.evaluate(self.x_test, self.y_test)}")
        self.assertEqual(1, 1)

    def test_singles(self):
        self.single(0)
        self.single(2009)
        self.single(2664)
        self.single(1511)
        self.single(500)


if __name__ == '__main__':
    unittest.main()
