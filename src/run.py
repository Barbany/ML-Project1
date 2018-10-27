"""Main function of the project. Execute it with the appropriate arguments to perform training of a linear model
with given constraints."""

# Auxiliary libraries to open files and parse arguments
import os

import numpy as np

from utils.helpers import *
from preproc.data_clean import *
from ml_methods.implementations import *

from parameters import default_params
from utils.argument_parser import parse_arguments
from utils.file_utils import setup_results_dir, tee_stdout

from ml_methods.cross_validation import cross_validation


def main(**params):
    params = dict(
        default_params,
        **params
    )
    np.random.seed(params['seed'])

    # Put all outputs on the log file stored in the result directory
    results_path = setup_results_dir(params)
    tee_stdout(os.path.join(results_path, 'log'))

    # Look if data have already been processed
    npy_files = [file for file in os.listdir(results_path) if file.endswith(".npz")]

    if len(npy_files) == 0:
        if params['split_jet']:
            load_csv_split_jet(
                os.path.join(params['raw_data'], 'train.csv'),
                os.path.join(params['raw_data'], 'test.csv'), results_path, mass=params['split_mass'],
                outliers=params['outliers'], verbose=params['verbose'])

            npy_files = [file for file in os.listdir(results_path) if file.endswith(".npz")]
        else:
            yb, input_data, _, _, test_data, test_ids = load_csv_data_no_na(
                os.path.join(params['raw_data'], 'train.csv'),
                os.path.join(params['raw_data'], 'test.csv'))
            if params['correlation']:
                input_data = correlation_coefficient(input_data)

            if params['pca']:
                input_data, test_data = pca(input_data, test_data)
            else:
                input_data, mean_x, std_x = standardize_by_feat(input_data)
                test_data = (test_data - mean_x) / std_x

            # Save all values in a unique file. If we did this with the split_jet we could get a Memory Error exception
            np.savez(os.path.join(results_path, 'processed_data.npz'), yb=yb, input_data=input_data, test_data=test_data, test_ids=test_ids)

    # Load data from npz files of separated jets (and mass if parameter set to True)
    if params['split_jet']:
        predictions = []
        ids_prediction = []
        for file_jet in npy_files:
            if params['verbose']:
                print('Processing file ', file_jet)

            # Extract information from file name
            jet = file_jet.split('_')[2][-1]
            mass = file_jet.split('_')[3] != 'no'

            # Load data
            data = np.load(os.path.join(results_path, file_jet))
            yb = data['yb']
            input_data = data['input_data']
            test_data = data['test_data']
            test_ids = data['test_ids']

            if params['loss_function'] == 'logistic':
                # Convert labels to {0, 1} instead of {-1, 1}
                yb[yb == -1] = 0

            # Train the model
            if params['verbose']:
                print('-' * 120)
                print('Start training with jet ', jet, ' with mass'*mass, 'without mass'*(not mass))
                print('-' * 120)

            if params['cross_validation']:
                _, best_lambda, best_degree, w_star = cross_validation(yb, input_data, params['k-fold'],
                                                                       lambdas=np.logspace(-10, 0, 2),
                                                                       degrees=range(8, 12),
                                                                       max_iters=params['max_iters'],
                                                                       gamma=params['gamma'],
                                                                       verbose=params['verbose'], jet=jet,
                                                                       mass=mass,
                                                                       loss_function=params['loss_function'])
                tx_train = build_poly(train_data, best_degree)
                tx_test = build_poly(test_data, best_degree)

                # Standardize test and train with train values for each feature
                # Note that first column is not included since it's the bias term
                _, mean_x, std_x = standardize_by_feat(tx_train[:, 1:])
                tx_test[:, 1:] = (tx_test[:, 1:] - mean_x) / std_x
            else:
                best_degree = 9
                best_lambda = 1e2
                tx_train = build_poly(train_data, best_degree)
                tx_test = build_poly(test_data, best_degree)

                # Standardize test and train with train values for each feature
                # Note that first column is not included since it's the bias term
                _, mean_x, std_x = standardize_by_feat(tx_train[:, 1:])
                tx_test[:, 1:] = (tx_test[:, 1:] - mean_x) / std_x

                w_star = reg_logistic_regression(yb, tx_train, lambda_=best_lambda, gamma=params['gamma'],
                                                 max_iters=params['max_iters'], initial_w=np.zeros(tx_tr.shape[1]))

            predictions = predictions + list(predict_labels_logistic(w_star, tx_test, jet=file_jet,
                                                                     mass=params['split_mass']))
            ids_prediction = ids_prediction + list(test_ids)

        # Sort predictions according to IDs
        test_ids, y_pred = zip(*sorted(zip(ids_prediction, predictions)))

    else:
        data = np.load(npy_files)
        yb = data['yb']
        input_data = data['input_data']
        test_data = data['test_data']
        test_ids = data['test_ids']

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
