import numpy as np

from ml_methods.implementations import ridge_regression
from utils.costs import compute_loss
from utils.helpers import build_poly, build_k_indices, build_poly_cross_terms

from preproc.data_clean import load_csv_data_no_na
from utils.costs import compute_loss, accuracy
from ml_methods.implementations import *
from utils.helpers import predict_labels, predict_labels_logistic
from utils.helpers import standardize, standardize_by_feat
import matplotlib.pyplot as plt
import sys


def get_data_cv(y, x, k_indices, k, degree):
    """
    Get train and test data from number of fold and randomized indexes. Returns data after applying polynomial of
    a certain degree
    :param y: Labels
    :param x: Features
    :param k_indices: Random indices obtained with build_k_indices
    :param k: Number of fold that will be used as train
    :param degree: Degree of polynomial to use
    :return: tx_train, y_train, tx_test, y_test
    """
    x_test = x[k_indices[k]]
    x_train = np.delete(x, k_indices[k], axis=0)
    y_test = y[k_indices[k]]
    y_train = np.delete(y, k_indices[k], axis=0)

    tx_train = build_poly_cross_terms(x_train, degree)
    tx_test = build_poly_cross_terms(x_test, degree)

    # Standardize test and train with train values for each feature
    # Note that first column is not included since it's the bias term
    tx_train[:, 1:], mean_x, std_x = standardize_by_feat(tx_train[:, 1:])
    tx_test[:, 1:] = (tx_test[:, 1:] - mean_x) / std_x

    return tx_train, y_train, tx_test, y_test


def cross_validation(y, x, k_fold, lambdas, degrees, max_iters, gamma, loss_function, batch_size=None,
                     seed=123, verbose=False, jet=0, mass=False):
    """
    K-fold cross validation
    :param y: labels
    :param x: features
    :param k_fold: number of folds
    :param lambdas: range of the lambdas to consider
    :param degrees: range of degrees to consider
    :param max_iters: Maximum number of iterations
    :param gamma: Learning rate
    :param loss_function: Loss function
    :param batch_size: Data points considered in each gradient (Default: None, i.e. GD - not SGD)
    :param seed: random seed
    :param verbose: Print information
    :param jet: Number of jet if we are using it
    :param mass: Boolean indicating if we are splitting with mass
    :return: minimum loss (min_loss) and optimal value for lambda (best_lambda) and degree (best_degree)
    """

    acc_degrees = np.zeros(len(degrees))
    acc_lambdas = np.zeros(len(degrees))

    k_indices = build_k_indices(y, k_fold, seed)

    max_acc = -1

    w_star = []

    for ind_deg, degree in enumerate(degrees):
        acc_lambdas_te = np.zeros(len(lambdas))
        acc_lambdas_tr = np.zeros(len(lambdas))
        for ind_lamb, lambda_ in enumerate(lambdas):
            acc_tr = []
            acc_te = []
            ws = []
            for k in range(k_fold):
                tx_tr, y_tr, tx_te, y_te = get_data_cv(y, x, k_indices, k, degree)
                initial_w = np.zeros(tx_tr.shape[1])
                if batch_size is None:
                    w, loss = least_square_gd(y_tr, tx_tr, initial_w, max_iters,
                                              gamma, loss_function=loss_function, lambda_=lambda_)
                else:
                    w, loss = least_square_sgd(y_tr, tx_tr, initial_w, batch_size, max_iters,
                                               gamma, loss_function=loss_function, lambda_=lambda_)
                y_pred_tr = predict_labels_logistic(w, tx_tr)
                acc_tr.append(accuracy(y_tr, y_pred_tr))
                y_pred_te = predict_labels_logistic(w, tx_te)
                acc_te.append(accuracy(y_te, y_pred_te))
                ws.append(w)

            # Mean across all folds
            acc_lambdas_te[ind_lamb] = np.mean(acc_te)
            acc_lambdas_tr[ind_lamb] = np.mean(acc_tr)

            print('With degree ', degree, ' and lambda ', lambda_, ' we obtain an accuracy in test of ',
                  np.mean(acc_te), '+-', np.std(acc_te))

            if acc_lambdas_te[ind_lamb] > max_acc:
                w_star = ws[int(np.argmax(acc_te))]
                max_acc = acc_lambdas_te[ind_lamb]

        ind_max_acc_lamb = np.argmax(acc_lambdas_te)
        acc_lambdas[ind_deg] = lambdas[ind_max_acc_lamb]
        acc_degrees[ind_deg] = acc_lambdas_te[ind_max_acc_lamb]
        plt.plot(lambdas, acc_lambdas_te, 'b-', label='Test')
        plt.plot(lambdas, acc_lambdas_tr, 'r-', label='Train')
        plt.legend(loc='upper left')
        plt.title('Accuracy evolution for degree ' + str(degree) + ' in jet ' + str(jet) + ' with mass'*mass)
        plt.ylabel('Accuracy')
        plt.xlabel('Lambda')
        plt.savefig('../results/plots/plot_degree_' + str(degree) + '_jet_' + str(jet) + ' w_mass'*mass + '.png')
        plt.clf()

        if verbose:
            print("For degree {0}, best lambda: {1} with a test accuracy: {2}".format(degree, acc_lambdas[ind_deg],
                  acc_degrees[ind_deg]))

    ind_max_acc = np.argmax(acc_degrees)
    max_acc = acc_degrees[ind_max_acc]
    best_lambda = acc_lambdas[ind_max_acc]
    best_degree = degrees[ind_max_acc]
    plt.plot(degrees, acc_degrees)
    plt.title('Maximum accuracies')
    plt.ylabel('Accuracy')
    plt.xlabel('Degree')
    plt.savefig('../results/plots/plot_max_lambdas_jet_' + str(jet) + ' w_mass'*mass + '.png')
    plt.clf()

    return max_acc, best_lambda, best_degree, w_star
