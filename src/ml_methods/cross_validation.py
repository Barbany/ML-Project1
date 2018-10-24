import numpy as np

from ml_methods.implementations import ridge_regression
from utils.costs import compute_loss
from utils.helpers import build_poly, build_k_indices

from preproc.data_clean import load_csv_data_no_na
from utils.costs import compute_loss
from ml_methods.implementations import ridge_regression, logistic_regression, reg_logistic_regression
from utils.helpers import predict_labels
from utils.helpers import standardize, standardize_by_feat
import sys


def get_data_cv(y, x, k_indices, k, degree):
    x_test = x[k_indices[k]]
    x_train = np.delete(x, k_indices[k], axis=0)
    y_test = y[k_indices[k]]
    y_train = np.delete(y, k_indices[k], axis=0)

    tx_train = build_poly(x_train, degree)
    tx_test = build_poly(x_test, degree)

    tx_train[:, 1:], mean_x, std_x = standardize_by_feat(tx_train[:, 1:])
    tx_test[:, 1:] = (tx_test[:, 1:] - mean_x) / std_x

    return tx_train, y_train, tx_test, y_test


def cross_validation(y, x, k_fold, lambdas, degrees, max_iters, gamma, seed=123, verbose=False):
    """
    K-fold cross validation
    :param y: labels
    :param x: features
    :param k_fold: number of folds
    :param lambdas: range of the lambdas to consider
    :param degrees: range of degrees to consider
    :param max_iters: Maximum number of iterations
    :param gamma: Learning rate
    :param seed: random seed
    :param verbose: Print information
    :return: minimum loss (min_loss) and optimal value for lambda (best_lambda) and degree (best_degree)
    """

    loss_degrees = np.zeros(len(degrees))
    min_lambdas = np.zeros(len(degrees))

    k_indices = build_k_indices(y, k_fold, seed)

    min_loss = 1e10

    w_star = []

    for ind_deg, degree in enumerate(degrees):
        loss_lambdas = np.zeros(len(lambdas))
        for ind_lamb, lambda_ in enumerate(lambdas):
            loss_tr = []
            loss_te = []
            ws = []
            for k in range(k_fold):
                tx_tr, y_tr, tx_te, y_te = get_data_cv(y, x, k_indices, k, degree)
                initial_w = np.zeros(tx_tr.shape[1])
                w, loss = reg_logistic_regression(y_tr, tx_tr, lambda_, initial_w, max_iters, gamma)
                loss_tr.append(loss)
                loss2 = compute_loss(y_te, tx_te, w, loss_function='logistic', lambda_=lambda_)
                loss_te.append(loss2)
                ws.append(w)
            loss_lambdas[ind_lamb] = np.mean(loss_te)
            if loss_lambdas[ind_lamb] < min_loss:
                w_star = ws[loss_te == np.min(loss_te)]
        ind_min_loss_lamb = np.argmin(loss_lambdas)
        min_lambdas[ind_deg] = lambdas[ind_min_loss_lamb]
        loss_degrees[ind_deg] = loss_lambdas[ind_min_loss_lamb]

        if verbose:
            print("For degree {0}, best lambda: {1} with a test loss: {2}".format(degree, min_lambdas[ind_deg],
                                                                                  loss_degrees[ind_deg]))

    ind_min_loss = np.argmin(loss_degrees)
    min_loss = loss_degrees[ind_min_loss]
    best_lambda = min_lambdas[ind_min_loss]
    best_degree = degrees[ind_min_loss]

    return min_loss, best_lambda, best_degree, w_star
