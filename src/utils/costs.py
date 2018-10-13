"""Function used to compute the loss."""
import numpy as np


def compute_loss(y, tx, w, loss_function='mse'):
    """Calculate the loss given a specific function.

    The possible loss functions are:
        * MSE (By default)
        * MAE
    """
    return {
        'mse': 1 / len(y) * sum((y - tx.dot(w)) ** 2),
        'mae': 1 / len(y) * sum(np.abs(y - tx.dot(w)))
    }[loss_function]


def compute_gradient(y, tx, w, loss_function='mse'):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    return {
        'mse': -1 / len(y) * tx.transpose().dot(y - tx.dot(w)),
        'mae': -1 / len(y) * tx.transpose().dot([-1 if e <= 0 else 1 for e in (y - tx.dot(w))])
    }[loss_function]
