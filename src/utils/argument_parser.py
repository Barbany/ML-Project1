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
        '-pca', action='store_true', help='Perform experiment with Principal Component Analysis', default=False
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Provide additional details about the program. This level of detail'
             ' can be very helpful for troubleshooting problems', default=False
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
