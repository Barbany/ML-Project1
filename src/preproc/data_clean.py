import numpy as np
from utils.helpers import load_csv_data


def load_csv_data_no_na(db_path='../../data/train.csv'):
    # Load RAW data
    yb, input_data, ids = load_csv_data(db_path)

    # Label NANs with numpy built-in indicator
    input_data[input_data == -999] = np.nan

    for feature in range(input_data.shape[1]):
        print('Feature #', feature, ' has \t',
              '{0:.2f}'.format(np.count_nonzero(np.isnan(input_data[:, feature]))/len(input_data[:, feature])*100),
              '% NANs')

    # Eliminate all features with 1 or more NANs
    # We could also try to only eliminate those over a given threshold (e.g. 20%)
    return yb, input_data[:, ~np.any(np.isnan(input_data), axis=0)], ids


def pca(features):
    n, m = features.shape       # n is the number of data points and m the number of features
    means = np.mean(features, axis=0)
    scatter_mat = np.zeros((m, m))
    for i in range(n):
        scatter_mat += (features[i] - means).dot((features[i] - means).T)
    print(scatter_mat)


yb, input_data, ids = load_csv_data_no_na()
pca(input_data)
