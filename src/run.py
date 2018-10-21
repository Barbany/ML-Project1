"""Main function of the project. Execute it with the appropriate arguments to perform training of a linear model
with given constraints."""

# Auxiliary libraries to open files and parse arguments
import os
import sys
import argparse

import numpy as np

from utils.helpers import predict_labels, create_csv_submission, standardize
from preproc.data_clean import pca, load_csv_data_no_na, correlation_coefficient, load_csv_split_jet
from ml_methods.implementations import least_squares_sgd

default_params = {
    'verbose': False,
    'pca': True,
    'correlation': False,
    'results_path': '../results',
    'raw_data': '../data',
    'seed': 123,
    'max_iters': 50,
    'gamma': 1e-5,
    'batch_size': 8,
    'bias': True,
    'split_jet': False,
    'loss_function': 'logistic'
}
tag_params = [
    'pca', 'bias', 'loss_function', 'split_jet'
]


def make_tag(params):
    def to_string(value):
        if isinstance(value, bool):
            return 'T' if value else 'F'
        elif isinstance(value, list):
            return ','.join(map(to_string, value))
        else:
            return str(value)

    return '-'.join(
        key + '_' + to_string(params[key])
        for key in tag_params
    )


def setup_results_dir(params):
    def ensure_dir_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)

    tag = make_tag(params)
    results_path = os.path.abspath(params['results_path'])
    ensure_dir_exists(results_path)
    results_path = os.path.join(results_path, tag)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    return results_path


def tee_stdout(log_path):
    log_file = open(log_path, 'w', 1)
    stdout = sys.stdout

    class Tee:
        @staticmethod
        def write(string):
            log_file.write(string)
            stdout.write(string)

        @staticmethod
        def flush():
            log_file.flush()
            stdout.flush()

    sys.stdout = Tee()


def main(**params):
    params = dict(
        default_params,
        **params
    )
    np.random.seed(params['seed'])

    # Put all outputs on the log file stored in the result directory
    results_path = setup_results_dir(params)
    tee_stdout(os.path.join(results_path, 'log'))

    npy_file = os.path.join(results_path, 'processed_data.npz')
    if os.path.isfile(npy_file):
        data = np.load(npy_file)
        yb = data['yb']
        input_data = data['input_data']
        test_data = data['test_data']
        test_ids = data['test_ids']
    else:
        if params['split_jet']:
            yb, input_data, _, _, test_data, test_ids = load_csv_split_jet(
                os.path.join(params['raw_data'], 'train.csv'),
                os.path.join(params['raw_data'], 'test.csv'))
        else:
            yb, input_data, _, _, test_data, test_ids = load_csv_data_no_na(
                os.path.join(params['raw_data'], 'train.csv'),
                os.path.join(params['raw_data'], 'test.csv'))
        if params['correlation']:
            input_data = correlation_coefficient(input_data)

        if params['pca']:
            input_data, test_data = pca(input_data, test_data)
        else:
            input_data, _, _ = standardize(input_data)
            test_data, _, _ = standardize(test_data)

        if params['bias']:
            input_data = np.append(np.ones((input_data.shape[0], 1)), input_data, axis=1)
            test_data = np.append(np.ones((test_data.shape[0], 1)), test_data, axis=1)

        np.savez(npy_file, yb=yb, input_data=input_data, test_data=test_data, test_ids=test_ids)

    # Could we improve by initializing with other configurations? E.g. random
    initial_w = np.zeros(input_data.shape[1])

    # Train the model. Note that SGD and GD save every weights and loss whilst optimizers in implementations don't
    w, loss = least_squares_sgd(yb, input_data, initial_w, batch_size=params['batch_size'],
                                max_iters=params['max_iters'], gamma=params['gamma'],
                                loss_function=params['loss_function'])

    y_pred = predict_labels(w, test_data)
    create_csv_submission(test_ids, y_pred, results_path + '/results.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS
    )


    def parse_bool(arg):
        arg = arg.lower()
        if 'true'.startswith(arg):
            return True
        elif 'false'.startswith(arg):
            return False
        else:
            raise ValueError()


    parser.add_argument(
        '--pca', type=parse_bool, help='Perform experiment with Principal Component Analysis'
    )
    parser.add_argument(
        '--verbose', type=parse_bool,
        help='Provide additional details about the program. This level of detail'
             ' can be very helpful for troubleshooting problems'
    )
    parser.add_argument(
        '--bias', type=parse_bool,
        help='Include a bias term in the linear regression model'
    )
    parser.add_argument(
        '--max_iterations', type=int,
        help='Maximum number of iterations of the Gradient Descent algorithm'
    )
    parser.add_argument(
        '--batch_size', type=int,
        help='Number of data points used to update the weight vector in each iteration'
    )
    parser.add_argument(
        '--gamma', type=float,
        help='Learning rate: Determines how fast the weight converges to an optimum'
    )
    parser.set_defaults(**default_params)

    main(**vars(parser.parse_args()))
