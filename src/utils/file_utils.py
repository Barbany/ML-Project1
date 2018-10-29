"""Function used to create results folder and save a log of all prints."""

import os
import sys


def make_tag(params):
    """Return string which will be used as folder name with all tag_params and their value"""
    tag_params = [
        'pca', 'loss_function', 'split_jet', 'split_mass', 'outliers', 'correlation'
    ]

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
    """Create results folder if it doesn't exist and get its path"""
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
    """Functions to write the log file situated in the results folder"""
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
