"""Implementation of the methods seen in class and in the labs"""

import numpy as np

from utils.helpers import batch_iter
from utils.costs import compute_loss, compute_gradient


def least_squares_gd(y, tx, initial_w, max_iters, gamma, loss_function='mse', lambda_=0):
    """Gradient descent algorithm

    :param y: 1-D Array of labels
    :param tx: Matrix of features
    :param initial_w: Initial weights
    :param max_iters: Maximum Number of iterations
    :param gamma: Learning rate
    :param loss_function: Loss function, possibilities specified below
    :param lambda_: Regularizer factor (optional. Default = 0)
    :return: w, loss (Last weights and loss)

    The possible loss functions are:
        * MSE (By default)
        * MAE
        * RMSE (Root MSE)
        * Logistic
    """
    # Define parameters to store w and loss
    w = initial_w
    loss = None
    losses = []
    threshold = 1e-3
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w, loss_function, lambda_)
        # Compute the gradient for mse loss
        grad = compute_gradient(y, tx, w, loss_function, lambda_)
        w = w - gamma * grad
        if n_iter % 50 == 0:
             print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
        if n_iter % 2000 == 0:
            # Adaptive learning rate
            gamma = gamma*0.1
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, loss


def least_squares_sgd(y, tx, initial_w, batch_size, max_iters, gamma, loss_function='mse', lambda_=0):
    """Stochastic Gradient descent algorithm

    :param y: 1-D Array of labels
    :param tx: Matrix of features
    :param initial_w: Initial weights
    :param batch_size: Data points considered in each gradient
    :param max_iters: Maximum Number of iterations
    :param gamma: Learning rate
    :param loss_function: Loss function, possibilities specified below
    :param lambda_: Regularizer factor (optional. Default = 0)
    :return: w, loss (Last weights and loss)

    The possible loss functions are:
        * MSE (By default)
        * MAE
        * RMSE (Root MSE)
        * Logistic
    """
    w = initial_w
    n_iter = 0
    loss = None
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=max_iters):
        loss = compute_loss(minibatch_y, minibatch_tx, w, loss_function, lambda_)
        # Compute a stochastic gradient from just few examples n and their corresponding y_n labels
        w = w - gamma * compute_gradient(minibatch_y, minibatch_tx, w, loss_function, lambda_)
        print("Stochastic Gradient Descent({bi}/{ti}): loss={ls}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, ls=loss, w0=w[0], w1=w[1]))
        n_iter = n_iter + 1
    return w, loss


def least_squares(y, tx):
    """Least squares algorithm.

    :param y: labels
    :param tx: features
    :return: w, loss (Last weights and loss)
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)


def ridge_regression(y, tx, lambda_):
    """Ridge regression.

    :param y: labels
    :param tx: features
    :param lambda_: Regularizer factor
    :return: w, loss (Last weights and loss)
    """
    a = np.dot(np.transpose(tx), tx) + 2*tx.shape[0]*lambda_*np.identity(tx.shape[1])
    b = np.dot(np.transpose(tx), y)
    return np.linalg.solve(a, b)


def logistic_regression(y, tx, initial_w, max_iters, gamma, batch_size=None):
    """Logistic regression with gradient descent (if no batch argument) or SGD

    :param y: labels
    :param tx: features
    :param initial_w: Initial weights
    :param max_iters: Maximum number of iterations
    :param gamma: Learning rate
    :param batch_size: Data points considered in each gradient (Default: None, i.e. GD - not SGD)
    :return: w, loss (Last weights and loss)
    """
    if batch_size is None:
        return least_squares_gd(y, tx, initial_w, max_iters, gamma, loss_function='logistic')
    else:
        return least_squares_sgd(y, tx, initial_w, batch_size, max_iters, gamma, loss_function='logistic')


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, batch_size=None):
    """Regularized logistic regression.

    :param y: labels
    :param tx: features
    :param lambda_: Regularizer factor
    :param initial_w: Initial weights
    :param max_iters: Maximum number of iterations
    :param gamma: Learning rate
    :param batch_size: Data points considered in each gradient (Default: None, i.e. GD - not SGD)
    :return: w, loss (Last weights and loss)
    """
    if batch_size is None:
        return least_squares_gd(y, tx, initial_w, max_iters, gamma,
                                loss_function='logistic', lambda_=lambda_)
    else:
        return least_squares_sgd(y, tx, initial_w, batch_size, max_iters, gamma,
                                 loss_function='logistic', lambda_=lambda_)
