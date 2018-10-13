"""Function used to compute the loss."""

import numpy as np


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    return 1 / len(y) * sum((y - tx.dot(w)) ** 2)


def compute_mae_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    return 1 / len(y) * sum(np.abs(y - tx.dot(w)))
