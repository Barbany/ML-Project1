"""Main function of the project. Execute it with the appropriate arguments to perform training of a linear model
with given constraints."""

# Libraries to deal with files
import os
import sys

import argparse

import numpy as np

from utils.helpers import load_csv_data
from utils.stochastic_gradient_descent import stochastic_gradient_descent

default_params = {
    'verbose': False,
    'pca': False,
    'mda': False,
    'results_path': '../results',
    'raw_data': '../data',
    'seed': 123,
    'max_iters': 50,
    'gamma': 0.01,
    'batch_size': 128,
    'bias': True,
    'loss_function': 'mse'
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

    results_path = params['results_path']

    # Put all outputs on the log file stored in the result directory
    tee_stdout(os.path.join(results_path, make_tag(params)))

    yb, input_data, ids = load_csv_data(os.path.join(params['raw_data'], 'train.csv'))

    if params['bias']:
        input_data = np.append(np.ones((input_data.shape[0], 1)), input_data, axis=1)

    initial_w = np.random.rand(input_data.shape[0])
    stochastic_gradient_descent(yb, input_data, initial_w, batch_size=params['batch_size'],
                                max_iters=params['max_iters'], gamma=params['gamma'],
                                loss_function=params['loss_function'])


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
