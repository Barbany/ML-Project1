"""Implementation of the methods seen in class and in the labs"""

import numpy as np

from utils.helpers import batch_iter
from utils.costs import compute_loss, compute_gradient


def least_squares_gd(y, tx, initial_w, max_iters, gamma, loss_function='mse'):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    loss = None
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w, loss_function)
        # Compute the gradient for mse loss
        w = w - gamma * compute_gradient(y, tx, w, loss_function)
        print("Gradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))
        n_iter = n_iter + 1
    return w, loss


def least_squares_sgd(y, tx, initial_w, batch_size, max_iters, gamma, loss_function='mse'):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    n_iter = 0
    loss = None
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=max_iters):
        loss = compute_loss(minibatch_y, minibatch_tx, w, loss_function)
        # Compute a stochastic gradient from just few examples n and their corresponding y_n labels
        w = w - gamma * compute_gradient(minibatch_y, minibatch_tx, w, loss_function)
        print("Stochastic Gradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))
        n_iter = n_iter + 1
    return w, loss


def least_squares(y, tx):
    """Least squares algorithm."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)


def ridge_regression(y, tx, lambda_):
    """Ridge regression."""
    aI = lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)
