"""Implementation of the methods seen in class and in the labs"""
import numpy as np
from utils.helpers import batch_iter


def compute_mse_loss(y, tx, w):
    """Calculate the mse loss."""
    return 1 / len(y) * sum((y - tx.dot(w)) ** 2)


def compute_mae_loss(y, tx, w):
    """Calculate the mae loss."""
    return 1 / len(y) * sum(np.abs(y - tx.dot(w)))


def least_squares_gd(y, tx, initial_w, max_iters, gamma, mse=True):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    loss = None
    for n_iter in range(max_iters):
        if mse:
            loss = compute_mse_loss(y, tx, w)
            # Compute the gradient for mse loss
            grad = -1 / len(y) * tx.transpose().dot(y - tx.dot(w))
        else:  # MAE
            loss = compute_mae_loss(y, tx, w)
            # Compute the gradient for mae loss
            grad = -1 / len(y) * tx.transpose().dot([-1 if e <= 0 else 1 for e in (y - tx.dot(w))])
        w = w - gamma * grad
        print("Gradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))
    return w, loss


def least_squares_sgd(y, tx, initial_w, batch_size, max_iters, gamma, mse=True):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    n_iter = 0
    loss = None
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=max_iters):
        if mse:
            loss = compute_mse_loss(minibatch_y, minibatch_tx, w)
        else:  # MAE
            loss = compute_mae_loss(minibatch_y, minibatch_tx, w)
        # Compute a stochastic gradient from just few examples n and their corresponding y_n labels
        stoch_grad = -1 / len(minibatch_y) * minibatch_tx.transpose().dot(minibatch_y - minibatch_tx.dot(w))
        w = w - gamma * stoch_grad
        print("Stochastic Gradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))
    return w, loss


def least_squares(y, tx):
    """Least squares algorithm."""
    a = np.linalg.inv(np.dot(np.transpose(tx), tx))
    return np.matmul(a, np.dot(np.transpose(tx), y))


def ridge_regression(y, tx, lambda_):
    """Ridge regression."""
    mat = np.dot(np.transpose(tx), tx)
    a = np.linalg.inv(mat + lambda_ * np.ones((mat.shape[0], mat.shape[1])))
    return np.matmul(a, np.dot(np.transpose(tx), y))
