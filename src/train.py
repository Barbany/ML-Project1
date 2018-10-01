"""Main function of the project. Execute it with the appropriate arguments to perform training of a linear model
with given constraints."""

import os
import sys
import argparse


default_params = {
    'verbose': False,
    'pca': False,
    'mda': False,
    'results_path': '../results',
    'raw_data': '../../data',
    'seed': 123
}
tag_params = [
    'pca', 'mda'
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
        if key not in default_params or params[key] != default_params[key]
    )


def tee_stdout(log_path):
    log_file = open(log_path, 'a', 1)
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
    seed = params.get('seed')

    results_path = params['results_path']

    # Put all outputs on the log file stored in the result directory
    tee_stdout(os.path.join(results_path, make_tag(params)))


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
    parser.set_defaults(**default_params)

    main(**vars(parser.parse_args()))