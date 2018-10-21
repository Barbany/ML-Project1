import os
import sys

from parameters import tag_params


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