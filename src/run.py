"""Main function of the project. Execute it with the appropriate arguments to perform training of a linear model
with given constraints."""

# Auxiliary libraries to open files and parse arguments
import os

import numpy as np

from utils.helpers import predict_labels, create_csv_submission, standardize
from preproc.data_clean import pca, load_csv_data_no_na, correlation_coefficient, load_csv_split_jet
from ml_methods.implementations import least_squares_sgd

from parameters import default_params
from utils.argument_parser import parse_arguments
from utils.file_utils import setup_results_dir, tee_stdout


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

    if params['split_jet']:
        predictions = []
        ids_prediction = []
        for jet in range(input_data.shape[0]):
            # Use features and labels associated to a given jet
            input_data_jet = input_data[jet]
            yb_jet = input_data[jet]

            # Could we improve by initializing with other configurations? E.g. random
            initial_w = np.zeros(input_data_jet.shape[1])

            # Train the model
            w, loss = least_squares_sgd(yb_jet, input_data_jet, initial_w, batch_size=params['batch_size'],
                                        max_iters=params['max_iters'], gamma=params['gamma'],
                                        loss_function=params['loss_function'])
            predictions.append(predict_labels(w, test_data[jet]))
            ids_prediction.append(test_ids[jet])

        # Sort predictions according to IDs
        test_ids, y_pred = zip(*sorted(zip(ids_prediction, predictions)))
    else:
        # Could we improve by initializing with other configurations? E.g. random
        initial_w = np.zeros(input_data.shape[1])

        # Train the model
        w, loss = least_squares_sgd(yb, input_data, initial_w, batch_size=params['batch_size'],
                                    max_iters=params['max_iters'], gamma=params['gamma'],
                                    loss_function=params['loss_function'])

        y_pred = predict_labels(w, test_data)

    create_csv_submission(test_ids, y_pred, results_path + '/results.csv')


if __name__ == '__main__':
    main(**vars(parse_arguments(default_params)))
