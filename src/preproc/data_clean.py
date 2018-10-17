import numpy as np
from utils.helpers import load_csv_data, standardize


def load_csv_data_no_na(db_path, na_indicator=-999):
    """
    Load raw data and eliminate NAs indicated with a certain numeric value
    Returns data without the contaminated features
    :param db_path: Path of the raw data
    :param na_indicator: Numeric NA Indicator (optional). Default value = -999
    :return: labels, features, ids
    """
    # Load RAW data
    yb, input_data, ids = load_csv_data(db_path)

    # Label NANs with numpy built-in indicator
    input_data[input_data == na_indicator] = np.nan

    for feature in range(input_data.shape[1]):
        print('Feature #', feature, ' has \t',
              '{0:.2f}'.format(np.count_nonzero(np.isnan(input_data[:, feature]))/len(input_data[:, feature])*100),
              '% NANs')

    # Eliminate all features with 1 or more NANs
    # We could also try to only eliminate those over a given threshold (e.g. 20%)
    return yb, input_data[:, ~np.any(np.isnan(input_data), axis=0)], ids


def pca(features, threshold=1e-4, verbose=False):
    """
    Perform a PCA transformation to only keep the most relevant features given a certain threshold
    :param features: Matrix of features (input_data output argument of load_csv_data_no_na). Shape = (x, num_features)
    :param threshold: Minimum ratio between a given eigenvalue and the sum of all eigenvalues to keep the associated
    eigenvector in the transformation matrix (optional). Default value = 1e-4 (0.01%)
    :param verbose: Print information of each eigenvalue and its percentage of contribution
    :return: transformed_features. Shape = (x, num_new_features) Note that num_new_features <= num_features
    """
    # z-score normalisation
    features, _, _ = standardize(features)

    # Compute eigenvalues and eigenvectors with covariance matrix
    eig_val_cov, eig_vec_cov = np.linalg.eig(np.cov(features.T))
    sum_eig_cov = np.sum(eig_val_cov)

    # Pair eigenvalues with the associated eigenvectors and sort them in decreasing order of eigenvalues
    eig_pairs_cov = [(np.abs(eig_val_cov[i]), eig_vec_cov[:, i]) for i in range(len(eig_val_cov))]
    eig_pairs_cov.sort(key=lambda x: x[0], reverse=True)

    if verbose:
        # Print eigenvalues and its percentage of contribution
        for i in eig_pairs_cov:
            print('%.2f' % (i[0] * 100 / sum_eig_cov), '%')
        print('-'*30)

    # Take those eigenvectors with associated eigenvalues with more than 'threshold' of contribution compared
    # to the sum of all eigenvalues for the transformation matrix
    w_mat = np.asarray([i[1] for i in eig_pairs_cov if i[0] > sum_eig_cov*threshold])

    # Return transformed data
    return features.dot(w_mat.T)
