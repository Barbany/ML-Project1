""" Some helper functions for project 1."""
import csv
import numpy as np
from utils.implementations import ridge_regression
from utils.costs import compute_loss


def load_csv_data(data_path, sub_sample=None):
    """
    Loads data and returns y (class labels), tX (features) and ids (event ids)
    Use the sub_sample optional argument to test a reduced dataset by indicating the sampling period
    """
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=[1])
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1
    
    # sub-sample
    if sub_sample is not None:
        yb = yb[::sub_sample]
        input_data = input_data[::sub_sample]
        ids = ids[::sub_sample]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


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

