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
    return {
        'mse': 1 / (2 * len(y)) * sum((y - tx.dot(w)) ** 2) + lambda_ * np.linalg.norm(w) ** 2,
        'rmse': 1 / (len(y)) * np.abs(sum((y - tx.dot(w)))) + lambda_ * np.linalg.norm(w) ** 2,
        'mae': 1 / len(y) * sum(np.abs(y - tx.dot(w))) + lambda_ * np.linalg.norm(w) ** 2,
        'logistic': logistic(y, tx, w, lambda_)
    }[loss_function]


def logistic(y, tx, w, lambda_):
    in_pred = tx.dot(w)
    pred = sigmoid(in_pred)
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    loss_reg = np.squeeze(-loss) + lambda_ * np.squeeze(w.T.dot(w))
    return loss_reg


def logistic_2(y, tx, w, lambda_):
    in_pred = tx.dot(w)
    in_pred[in_pred >= 10] = 10
    in_pred[in_pred <= -10] = -10
    pred = sigmoid(in_pred)
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss) + lambda_ * np.squeeze(w.T.dot(w))


def accuracy(y_true, y_pred):
    # sum([1 for i in range(len(y_true)) if y_pred[i] == y_true[i]]) / len(y_true)
    return sum(y_true == y_pred) / len(y_true)


def correct_predictions(y, tx, w):
    """
        Return the percentage of wrong predictions (between 0 and 1)
    """

    y_pred = sigmoid(np.dot(tx, w))
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0)] = 1
    correct_pred = np.sum(y_pred == y)

    return float(correct_pred) / float(len(y))


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
    return {
        'mse': -1 / len(y) * tx.transpose().dot(y - tx.dot(w)) + 2 * lambda_ * w,
        'rmse': -1 / np.sqrt(len(y)) * tx.transpose().dot([-1 if e <= 0 else 1 for e in (y - tx.dot(w))])
                + 2 * lambda_ * w,
        'mae': -1 / len(y) * tx.transpose().dot([-1 if e <= 0 else 1 for e in (y - tx.dot(w))]) + 2 * lambda_ * w,
        'logistic': tx.T.dot(sigmoid(tx.dot(w)) - y) + 2 * lambda_ * w
    }[loss_function]


def sigmoid(x):
    """Apply sigmoid function on x"""
    return 1.0 / (1 + np.exp(-x))
