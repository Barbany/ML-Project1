import numpy as np

from ml_methods.implementations import ridge_regression
from utils.costs import compute_loss
from utils.helpers import build_poly, build_k_indices


def compute_loss_cv(y, x, k_indices, k, lambda_, degree):
    x_test = x[k_indices[k], :]
    x_train = np.delete(x, k_indices[k], axis=0)
    y_test = y[k_indices[k]]
    y_train = np.delete(y, k_indices[k], axis=0)

    tx_train = build_poly(x_train, degree)
    tx_test = build_poly(x_test, degree)

    w = ridge_regression(y_train, tx_train, lambda_)

    loss_tr = np.sqrt(2 * compute_loss(y_train, tx_train, w))
    loss_te = np.sqrt(2 * compute_loss(y_test, tx_test, w))
    return loss_tr, loss_te


def cross_validation(x, y, k_fold, degree, seed, lambdas):
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []

    for ind, lambda_ in enumerate(lambdas):
        loss_tr_k = 0
        loss_te_k = 0
        for k in range(k_fold):
            loss_tr, loss_te = compute_loss_cv(y, x, k_indices, k, lambda_, degree)
            loss_tr_k = loss_tr_k + loss_tr
            loss_te_k = loss_te_k + loss_te
        rmse_tr.append(loss_tr_k / k_fold)
        rmse_te.append(loss_te_k / k_fold)

    best_lambda = lambdas[np.argmin(rmse_te)]
    return best_lambda