"""Function used to compute the loss."""
import numpy as np


def compute_loss(y, tx, w, loss_function='mse', lambda_=0):
    """Calculate the loss given a specific function and an optional regularizer factor.

    :param y: labels
    :param tx: features
    :param w: weights
    :param loss_function: Loss function, possibilities specified below
    :param lambda_: Regularizer factor

    The possible loss functions are:
        * MSE (By default)
        * MAE
        * RMSE (Root MSE)
        * Logistic
    """
    return lambda_ * np.linalg.norm(w) ** 2 + {
        'mse': 1 / (2 * len(y)) * sum((y - tx.dot(w)) ** 2),
        'rmse': np.sqrt(2 * compute_loss(y, tx, w, 'mse', lambda_)),
        'mae': 1 / len(y) * sum(np.abs(y - tx.dot(w))),
        'logistic': np.sum(np.log(1. + np.exp(np.dot(tx, w)))) - np.dot(y.transpose(), np.dot(tx, w))
    }[loss_function]


def compute_gradient(y, tx, w, loss_function='mse', lambda_=0):
    """Compute a stochastic gradient from just few examples n and
    their corresponding y_n labels.

    :param y: labels
    :param tx: features
    :param w: weights
    :param loss_function: Loss function, possibilities specified below
    :param lambda_: Regularizer factor

    The possible loss functions are:
        * MSE (By default)
        * MAE
        * RMSE (Root MSE)
        * Logistic
    """
    return 2 * lambda_ * w + {
        'mse': -1 / len(y) * tx.transpose().dot(y - tx.dot(w)),
        'rmse': np.sqrt(len(y)) * compute_gradient(y, tx, w, 'mae'),
        'mae': -1 / len(y) * tx.transpose().dot([-1 if e <= 0 else 1 for e in (y - tx.dot(w))]),
        'logistic': np.dot(tx.transpose(), sigmoid(np.dot(tx, w)) - y)
    }[loss_function]


def sigmoid(x):
    """Apply sigmoid function on x"""
    return 1 / (1 + np.exp(-x))
