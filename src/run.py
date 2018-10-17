"""Main function of the project. Execute it with the appropriate arguments to perform training of a linear model
with given constraints."""

import argparse
# Auxiliary libraries to open files and parse arguments
import os
import sys

import numpy as np
from utils.helpers import load_csv_data, predict_labels, create_csv_submission
from utils.implementations import least_squares_sgd

default_params = {
    'verbose': False,
    'pca': False,
    'mda': False,
    'results_path': '../results',
    'raw_data': '../data',
    'seed': 123,
    'max_iters': 50,
    'gamma': 1e-5,
    'batch_size': 8,
    'bias': True,
    'loss_function': 'mse',
    'checkpoint': 'last',
    'save_checkpoint': True,
    'predict_checkpoint': 'best'
}
tag_params = [
    'pca', 'mda', 'bias', 'loss_function'
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

    yb, input_data, ids = load_csv_data(os.path.join(params['raw_data'], 'train.csv'))

    if params['bias']:
        input_data = np.append(np.ones((input_data.shape[0], 1)), input_data, axis=1)

    checkpoint = [file for file in os.listdir(results_path) if file.startswith(params['checkpoint'])]
    if not any(checkpoint):
        initial_w = np.random.rand(input_data.shape[1])
        print('No checkpoints')
    else:
        initial_w = np.load(os.path.join(results_path, checkpoint[0]))
        print('checkpoint loaded')

    # Train the model. Note that SGD and GD save every weights and loss whilst optimizers in implementations don't
    losses, ws = least_squares_sgd(yb, input_data, initial_w, batch_size=params['batch_size'],
                                   max_iters=params['max_iters'], gamma=params['gamma'],
                                   results_path=params['results_path'],
                                   loss_function=params['loss_function'])
    """""
    Save checkpoints. This is only valid for SGD and GD
    np.save(os.path.join(results_path, 'last-it' + str(params['max_iters']) + '-loss' +
                         '{:.2f}'.format(losses[-1]) + '.npy'), ws[-1])
    best_iteration = np.argmin(losses)
    np.save(os.path.join(results_path, 'best-it' + str(best_iteration) + '-loss' +
                         '{:.2f}'.format(losses[best_iteration]) + '.npy'), ws[best_iteration])"""""

    _, test_data, test_ids = load_csv_data(os.path.join(params['raw_data'], 'test.csv'))
    if params['bias']:
        test_data = np.append(np.ones((test_data.shape[0], 1)), test_data, axis=1)

    # predicting_weights = [file for file in os.listdir(results_path) if file.startswith(params['predict_checkpoint'])]
    # y_pred = predict_labels(np.load(os.path.join(results_path, predicting_weights)), test_data)

    y_pred = predict_labels(ws[-1], test_data)
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
        '--mda', type=parse_bool, help='Perform experiment with Multiple Discriminant Analysis'
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
        help='Maximum number of iterations of the SGD algorithm'
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
