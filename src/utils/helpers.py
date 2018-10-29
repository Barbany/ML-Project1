""" Some helper functions for project 1."""
import csv
import numpy as np
from utils.costs import sigmoid


def load_csv_data(data_path, sub_sample=None):
    """
    Loads data and returns y (class labels), tX (features) and ids (event ids)
    Use the sub_sample optional argument to test a reduced dataset by indicating the sampling period
    :param data_path: Path where the train file is located
    :param sub_sample: Sub sampling factor (integer value that skips as much events as its value). Default: None
    :return yb, input_data, ids: (labels, features, IDs)
    """
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=[1])
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # Convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1
    
    # Sub-sample
    if sub_sample is not None:
        yb = yb[::sub_sample]
        input_data = input_data[::sub_sample]
        ids = ids[::sub_sample]

    return yb, input_data, ids


def predict_labels(weights, data):
    """
    Generates class predictions given weights, and a test data matrix
    :param weights
    :param data: Features
    :return y_pred: Predicted labels
    """
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def predict_labels_logistic(weights, data):
    """
    Generates class predictions given weights, and a test data matrix for logistic regression
    The difference between this function and predict_labels is that this takes a threshold in 0.5 (predictions (0, 1))
    :param weights
    :param data: Features
    :return y_pred: Predicted labels
    """
    y_pred = sigmoid(np.dot(data, weights))
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1

    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to Kaggle
    :param ids: Event IDs associated with each prediction
    :param y_pred: Predicted class labels
    :param name: Path of csv output file to be created
    """
    with open(name, 'w', newline='') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


def standardize(x):
    """
    Standardize the original data set. Use this function only for training dataset and perform same computation
    in-site with the test using the return parameters of mean and standard deviation
    :param x: Features
    :return x, mean_x, std_x: Z_score normalized features and statistic parameters
    """
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def standardize_by_feat(x):
    """
    Standardize the original data set by features. Use this function only for training dataset and perform same
    computation in-site with the test using the return parameters of mean and standard deviation
    :param x: Features
    :return x, mean_x, std_x: Z_score normalized features by column and array of associated statistic parameters
    """
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use (For one batch):
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    :param y: Labels
    :param tx: Features
    :param batch_size: Data points considered in each gradient
    :param num_batches: Number of batches. Default=1
    :param shuffle: Random indexing of features. Default=True
    :return Iterator (See usage above)
    """
    data_size = len(y)

    # Do random permutations of indexes
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx

    # Create iterators for each batch
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree.
    :param x: Features
    :param degree: Degree of the polynomial
    :return: poly: Expanded features
    """
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def build_poly_cross_terms(x, degree):
    """
    Build a polynomial of a certain degree with crossed terms (applying sum, product and square of product)
    :param x: Features
    :param degree: Degree of the polynomial (for each individual feature)
    :return: poly: Expanded features
    """
    features = x.shape[1]
    # Create powers for each of the features
    poly = np.ones((len(x), 1))
    for feat in range(features):
        for deg in range(1, degree + 1):
            poly = np.c_[poly, np.power(x[:, feat], deg)]

    # Sum, multiply and features between them
    for this_feat in range(features):
        for that_feat in range(this_feat + 1, features):
            poly = np.c_[poly, x[:, this_feat] + x[:, that_feat],
                         x[:, this_feat] * x[:, that_feat],
                         np.power(x[:, this_feat] * x[:, that_feat], 2)]
    return poly


def build_k_indices(y, k_fold, seed):
    """
    Build k indices for k-fold. Use the return parameter as follows:
    x_test = x[k_indices[k]]
    x_train = np.delete(x, k_indices[k], axis=0)
    :param y: Labels
    :param k_fold: Number of folds
    :param seed: Seed of the random number generator
    :return: k_indices: Array of indexes. Size k_fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)
