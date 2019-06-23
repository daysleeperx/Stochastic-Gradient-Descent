"""Utils."""


import numpy as np


def normalize(data):
    """Normalize data by dividing by the column maximum."""
    return data / np.max(np.abs(data), axis=0)


def add_one_bias(data):
    """
    Add a bias unit of 1 to every vector in data list
    in order to simplify matrix multiplication.

    :param data: original numpy array of of x
    :return: modified numpy array
    """
    return np.c_[np.ones((len(data), 1)), data]


def random_int(size):
    """
    Return a random integer in range 0 to size.

    :param size: int
    :return: int
    """
    return np.random.randint(0, size)

