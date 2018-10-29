"""Argument parser and default parameters."""

import argparse

default_params = {
        'results_path': '../results',
        'raw_data': '../data',
        'seed': 123,
        'max_iters': 6000,
        'gamma': 1e-5,
        'batch_size': 8,
        'loss_function': 'logistic',
        'k-fold': 10
    }


def parse_arguments():
    """
    Set arguments from the command line when running 'run.py'. Run with option '-h' or '--help' for
    information about parameters and its usage.
    :return: parser.parse_args()
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS
    )

    parser.add_argument(
        '-pca', action='store_true', help='Perform experiment with Principal Component Analysis', default=False
    )
    parser.add_argument(
        '-cv', action='store_true', help='Perform cross validation', default=False
    )
    parser.add_argument(
        '-outliers', action='store_true', help='Delete events with one or some outliers', default=False
    )
    parser.add_argument(
        '-split_jet', action='store_false', help='Split data depending on the number of jets', default=True
    )
    parser.add_argument(
        '-split_mass', action='store_false', help='Treat differently data with meaning-full '
                                                  'estimated mass of Higgs boson', default=True
    )
    parser.add_argument(
        '-correlation', action='store_true', help='Delete pairs of features that are very correlated', default=False
    )
    parser.add_argument(
        '-visualize', action='store_true', help='Plot histogram of features without outliers', default=False
    )
    parser.add_argument(
        '-verbose', action='store_true',
        help='Provide additional details about the program. This level of detail'
             ' can be very helpful for troubleshooting problems', default=False
    )
    parser.add_argument(
        '-max_iterations', type=int,
        help='Maximum number of iterations of the Gradient Descent algorithm'
    )
    parser.add_argument(
        '-batch_size', type=int,
        help='Number of data points used to update the weight vector in each iteration'
    )
    parser.add_argument(
        '-gamma', type=float,
        help='Learning rate: Determines how fast the weight converges to an optimum'
    )
    parser.set_defaults(**default_params)

    return parser.parse_args()
