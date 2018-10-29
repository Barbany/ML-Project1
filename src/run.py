"""Main function of the project. Execute it with the appropriate arguments to perform training of a linear model
with given constraints."""

import os

import numpy as np

from preproc.data_clean import *
from preproc.visualizations import plot_distribution

from utils.helpers import *
from utils.argument_parser import parse_arguments
from utils.file_utils import setup_results_dir, tee_stdout

from ml_methods.cross_validation import cross_validation
from ml_methods.implementations import *


def main(**params):
    params = dict(
        default_params,
        **params
    )
    # Set random seed
    np.random.seed(params['seed'])

    # Put all outputs on the log file stored in the result directory
    results_path = setup_results_dir(params)
    tee_stdout(os.path.join(results_path, 'log'))

    # Look if data have already been processed
    npy_files = [file for file in os.listdir(results_path) if file.endswith(".npz")]

    # Data is not saved in the results path (with the given parameters)
    if len(npy_files) == 0:
        # Save data by splitting by jets. This procedure do not return data but creates several files
        # This is done because an array with matrices of different sizes prompted Memory Errors in 8 GB devices
        if params['split_jet']:
            load_csv_split_jet(
                os.path.join(params['raw_data'], 'train.csv'),
                os.path.join(params['raw_data'], 'test.csv'), results_path, mass=params['split_mass'],
                outliers=params['outliers'], verbose=params['verbose'])

            # Refresh the processed data files. Now we have those created with previous call
            npy_files = [file for file in os.listdir(results_path) if file.endswith(".npz")]
        else:
            # Load data by eliminating those features with NANs
            yb, input_data, _, _, test_data, test_ids = load_csv_data_no_nan(
                os.path.join(params['raw_data'], 'train.csv'),
                os.path.join(params['raw_data'], 'test.csv'))
            if params['correlation']:
                # Delete a feature if it's very correlated with another
                input_data = correlation_coefficient(input_data)
            if params['pca']:
                # Apply PCA dimensionality reduction
                input_data, test_data = pca(input_data, test_data)
            else:
                # Standardize data by features with statistical values extracted from training data
                input_data, mean_x, std_x = standardize_by_feat(input_data)
                test_data = (test_data - mean_x) / std_x

            # Save all values in a unique file. If we did this with the split_jet we could get a Memory Error exception
            # The keywords let us load data as a dictionary
            np.savez(os.path.join(results_path, 'processed_data.npz'), yb=yb, input_data=input_data,
                     test_data=test_data, test_ids=test_ids)

    # Load data from npz files of separated jets (and mass if parameter set to True)
    if params['split_jet']:
        predictions = []
        ids_prediction = []
        for file_jet in npy_files:
            if params['verbose']:
                print('Processing file ', file_jet)

            # Extract information from file name
            jet = int(file_jet.split('_')[2][-1])
            mass = file_jet.split('_')[3] != 'no'

            # Load data
            data = np.load(os.path.join(results_path, file_jet))
            yb = np.asarray(data['yb'])
            input_data = np.asarray(data['input_data'])
            test_data = np.asarray(data['test_data'])
            test_ids = np.asarray(data['test_ids'])

            # Visualize the histogram of data without outliers
            if params['visualize']:
                plot_distribution(input_data[jet].T, 'jet_filtering_' + str(jet) + '_mass'*mass, verbose=True)

            # Logistic regression uses sigmoid, which goes from 0 to 1
            if params['loss_function'] == 'logistic':
                # Convert labels to {0, 1} instead of {-1, 1}
                yb[yb == -1] = 0

            # Train the model
            if params['verbose']:
                print('-' * 120)
                print('Start training with jet ', jet, ' with mass'*mass, 'without mass'*(not mass))
                print('-' * 120)

            # Perform cross validation to find optimum parameters
            if params['cv']:
                if params['verbose']:
                    print('Start cross validation')
                _, best_lambda, best_degree, w_star = cross_validation(yb, input_data, params['k-fold'],
                                                                       lambdas=np.logspace(-10, -2, 9),
                                                                       degrees=range(8, 12),
                                                                       max_iters=params['max_iters'],
                                                                       gamma=params['gamma'],
                                                                       verbose=params['verbose'], jet=jet,
                                                                       mass=mass,
                                                                       loss_function=params['loss_function'])

                # Build polynomial with optimal degree from perspective of highest accuracy
                tx_train = build_poly_cross_terms(input_data, best_degree)
                tx_test = build_poly_cross_terms(test_data, best_degree)

                # Standardize test and train with train values for each feature
                # Note that first column is not included since it's the bias term
                _, mean_x, std_x = standardize_by_feat(tx_train[:, 1:])
                tx_test[:, 1:] = (tx_test[:, 1:] - mean_x) / std_x
            else:
                # Values found in cross validation. We provide them because it take a lot of time to run
                best_degree = [12, 9, 7, 9, 10, 9, 8, 9]
                best_lambda = [1e-5, 1e-2, 1e-5, 1e-4, 1e-6, 1e-4, 1e-5, 1e-10]
                if params['verbose']:
                    print('Predict samples for file with jet ', jet, ' and mass'*mass)
                    print('Build polynomial of cross terms with degree ', best_degree[2*jet + mass])

                # Build polynomial with optimal degree from perspective of highest accuracy
                tx_train = build_poly_cross_terms(input_data, best_degree[2*jet + mass])
                tx_test = build_poly_cross_terms(test_data, best_degree[2*jet + mass])

                # Standardize test and train with train values for each feature
                # Note that first column is not included since it's the bias term
                tx_train[:, 1:], mean_x, std_x = standardize_by_feat(tx_train[:, 1:])
                tx_test[:, 1:] = (tx_test[:, 1:] - mean_x) / std_x

                # Initialize weight vector coefficients as zeros
                initial_w = np.zeros(tx_train.shape[1])

                # Perform regularized logistic regression with optimal values
                w_star, _ = reg_logistic_regression(yb, tx_train, lambda_=best_lambda[2*jet + mass],
                                                    gamma=params['gamma'], max_iters=params['max_iters'],
                                                    initial_w=initial_w)

            if params['verbose']:
                print('Predicting samples for jet ', jet, ' with mass'*mass)

            # Predict labels of the given data chunk and append it to the other ones. Same with IDs
            predictions = predictions + list(predict_labels_logistic(w_star, np.asarray(tx_test)))
            ids_prediction = ids_prediction + list(test_ids)

        # Sort predictions according to IDs
        test_ids, y_pred = zip(*sorted(zip(ids_prediction, predictions)))

    else:
        # Load data
        data = np.load(npy_files)
        yb = data['yb']
        input_data = data['input_data']
        test_data = data['test_data']
        test_ids = data['test_ids']

        # Visualize the histogram of data without outliers
        if params['visualize']:
            plot_distribution(input_data, 'input_data', verbose=True)

        # Initialize weight vector coefficients as zeros
        initial_w = np.zeros(input_data.shape[1])

        # Train the model
        w, loss = least_squares_sgd(yb, input_data, initial_w, batch_size=params['batch_size'],
                                    max_iters=params['max_iters'], gamma=params['gamma'],
                                    loss_function=params['loss_function'])

        y_pred = predict_labels(w, test_data)

    # Create a CSV with the predictions either if it was by splitting jets or not
    create_csv_submission(test_ids, y_pred, results_path + '/results.csv')


if __name__ == '__main__':
    main(**vars(parse_arguments()))
