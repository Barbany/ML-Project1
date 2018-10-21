import numpy as np
import matplotlib.pyplot as plt
import seaborn
import time

from src.preproc.data_clean import load_csv_data_no_na
from src.utils.helpers import standardize


def correlation_coefficient(features, threshold = 0.9, visualize = False):
    """
        Finds the Pearson correlation coefficients between the features.
        High correlated features imply that the features are highly linked, so one of them is removed.
        :param features: Matrix of features (input_data output argument of load_csv_data_no_na). Shape = (x, num_features)
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
        plt.savefig('correlation_coefficient_' + timestr +'.png')
        plt.show()

    return uncorrelated_features



_, input_data, _, _ = load_csv_data_no_na('../../data/train.csv')
print(input_data.shape)
correlation_coefficient(input_data)