import argparse

def parse_bool(arg):
    arg = arg.lower()
    if 'true'.startswith(arg):
        return True
    elif 'false'.startswith(arg):
        return False
    else:
        raise ValueError()

def parse_arguments(default_params):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS
    )

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

    return parser.parse_args()