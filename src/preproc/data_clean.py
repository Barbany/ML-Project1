import numpy as np
import matplotlib.pyplot as plt
import seaborn
import time

from utils.helpers import load_csv_data, standardize


def load_csv_data_no_na(train_path, test_path, na_indicator=-999, verbose=False):
    """
    Load raw data and eliminate NAs indicated with a certain numeric value
    Returns data without the contaminated features
    :param train_path: Path of the training raw data
    :param test_path: Path of the training raw data
    :param na_indicator: Numeric NA Indicator (optional). Default value = -999
    :param verbose: Print number of NANs for each feature (optional). Default value = False
    :return: labels_tr, features_tr, ids_tr, labels_te, features_te, ids_te
    """
    # Load RAW data
    yb_tr, input_data_tr, ids_tr = load_csv_data(train_path)
    yb_te, input_data_te, ids_te = load_csv_data(test_path)

    # Label NANs with numpy built-in indicator
    input_data_tr[input_data_tr == na_indicator] = np.nan

    if verbose:
        for feature in range(input_data_tr.shape[1]):
            print('Feature #', feature, ' has \t',
                  '{0:.2f}'.format(np.count_nonzero(np.isnan(input_data_tr[:, feature]))
                                   / len(input_data_tr[:, feature]) * 100), '% NANs')

    # Eliminate all features with 1 or more NANs
    # We could also try to only eliminate those over a given threshold (e.g. 20%)
    remaining_cols = ~np.any(np.isnan(input_data_tr), axis=0)

    input_data_tr = input_data_tr[:, remaining_cols]
    input_data_te = input_data_te[:, remaining_cols]
    return yb_tr, input_data_tr, ids_tr, yb_te, input_data_te, ids_te


def load_csv_split_jet(train_path, test_path, na_indicator=-999, verbose=False, categorical_col=22):
    """"
    Load raw data by splitting depending on the 'PRI_jet_num', which determines the existence of some
    other features according to the explanation of the dataset included in
    'doc/[Background of dataset] - The Higgs Boson.pdf'. See highlighted lines p. 15 (Appendix B)

    :param train_path: Path of the training raw data
    :param test_path: Path of the training raw data
    :param na_indicator: Numeric NA Indicator (optional). Default value = -999
    :param verbose: Print number of NANs for each feature (optional). Default value = False
    :param categorical_col: Number of column where the categorical feature, in this case Jet, conditions the others
    :return: labels_tr, features_tr, ids_tr, labels_te, features_te, ids_te
    (All output variables are lists where each element is a the labels, features and IDs of a given jet)
    """

    # Load RAW data
    yb_tr, input_data_tr, ids_tr = load_csv_data(train_path)
    yb_te, input_data_te, ids_te = load_csv_data(test_path)
    if verbose:
        print('Train and test data successfully loaded')

    # Filter data for every different jet and append it to a list
    num_jets = np.unique(input_data_tr[:, categorical_col])

    yb_jets_tr = tx_jets_tr = ids_jets_tr = []
    yb_jets_te = tx_jets_te = ids_jets_te = []

    for jet in num_jets:
        if verbose:
            print("Creating data for jet ", jet)

        # Get indexes of data points with the same jet
        idx_tr = input_data_tr[:, categorical_col] == jet
        idx_te = input_data_te[:, categorical_col] == jet

        # Filter labels, features and IDs by jet
        y_tr = yb_tr[idx_tr]
        tx_tr = input_data_tr[idx_tr]
        id_tr = ids_tr[idx_tr]

        y_te = yb_te[idx_te]
        tx_te = input_data_te[idx_te]
        id_te = ids_te[idx_te]

        # Label NANs with numpy built-in indicator
        tx_tr[tx_tr == na_indicator] = np.nan

        # Only keep columns without NANs
        remaining_cols = ~np.any(np.isnan(tx_tr), axis=0)
        # Delete PRI_jet_num feature
        remaining_cols[categorical_col] = False

        # Delete features with NANs
        tx_tr = tx_tr[:, remaining_cols]
        tx_te = tx_te[:, remaining_cols]

        # For Jet 0, there is a really big outlier in the column 3. So, we will remove it
        if jet == 0:
            to_remove = (tx_tr[:, 3] < 200)
            tx_tr = tx_tr[to_remove, :]
            y_tr = y_tr[to_remove]
            ids_tr = ids_tr[to_remove]

            # Remove last column (only 0s)
            tx_tr = tx_tr[:, :tx_tr.shape[1] - 1]

        # Append every filtered matrix into the list
        yb_jets_tr.append(y_tr)
        tx_jets_tr.append(tx_tr)
        ids_jets_tr.append(id_tr)

        yb_jets_te.append(y_te)
        tx_jets_te.append(tx_te)
        ids_jets_te.append(id_te)

    return yb_jets_tr, tx_jets_tr, ids_jets_tr, yb_jets_te, tx_jets_te, ids_jets_te


def pca(features_tr, features_te, threshold=1e-4, verbose=False):
    """
    Perform a PCA transformation to only keep the most relevant features given a certain threshold
    :param features_tr: Matrix of train features. Shape = (-1, num_features)
    :param features_te: Matrix of test features. Shape = (-1, num_features)
    :param threshold: Minimum ratio between a given eigenvalue and the sum of all eigenvalues to keep the associated
    eigenvector in the transformation matrix (optional). Default value = 1e-4 (0.01%)
    :param verbose: Print information of each eigenvalue and its percentage of contribution
    :return: transformed_features (Shape = (x, num_new_features) Note that num_new_features <= num_features)
    """
    # z-score normalisation
    features_tr, _, _ = standardize(features_tr)

    # Compute eigenvalues and eigenvectors with covariance matrix
    eig_val_cov, eig_vec_cov = np.linalg.eig(np.cov(features_tr.T))
    sum_eig_cov = np.sum(eig_val_cov)

    # Pair eigenvalues with the associated eigenvectors and sort them in decreasing order of eigenvalues
    eig_pairs_cov = [(np.abs(eig_val_cov[i]), eig_vec_cov[:, i]) for i in range(len(eig_val_cov))]
    eig_pairs_cov.sort(key=lambda x: x[0], reverse=True)

    if verbose:
        # Print eigenvalues and its percentage of contribution
        for i in eig_pairs_cov:
            print('%.2f' % (i[0] * 100 / sum_eig_cov), '%')
        print('-' * 30)

    # Take those eigenvectors with associated eigenvalues with more than 'threshold' of contribution compared
    # to the sum of all eigenvalues for the transformation matrix
    w_mat = np.asarray([i[1] for i in eig_pairs_cov if i[0] > sum_eig_cov * threshold])

    # Return transformed data
    return features_tr.dot(w_mat.T), features_te.dot(w_mat.T)


def correlation_coefficient(features, threshold=0.9, visualize=False):
    """
        Finds the Pearson correlation coefficients between the features.
        High correlated features imply that the features are highly linked, so one of them is removed.
        :param features: Matrix of features (input_data output argument of load_csv_data_no_na).
               Shape = (x, num_features)
        :param threshold: Minimum correlation. Features with correlation above the threshold will be removed.
                          Default value = 0.9
        :param visualize: Print information of each eigenvalue and its percentage of contribution
        :return: uncorrelated_features (Shape = (x, num_new_features) Note that num_new_features <= num_features),
        """
    features, _, _ = standardize(features)

    # calculate the transpose matrix to find the correlated array of shape (#features, #features)
    correlation_array = np.corrcoef(features.T)

    # take the upper triangle of the correlation array without the diagonal
    upper_correlation_array = np.triu(correlation_array, 1)

    # Find index of feature columns with correlation greater than threshold
    remaining_cols = ~np.any(upper_correlation_array > threshold, axis=0)

    uncorrelated_features = features[:, remaining_cols]

    # visualization of the correlation array plot
    if visualize:
        mask = np.zeros_like(correlation_array)
        mask[np.triu_indices_from(mask)] = True
        # Create the heatmap using seaborn library.
        seaborn.heatmap(correlation_array, cmap='RdYlGn_r', vmax=1.0, vmin=-1.0, mask=mask, linewidths=2.5)

        # Show the plot we reorient the labels for each column and row to make them easier to read.
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)

        timestr = time.strftime("%d.%m.%Y-%H:%M:%S")
        plt.savefig('correlation_coefficient_' + timestr + '.png')
        plt.show()

    return uncorrelated_features
