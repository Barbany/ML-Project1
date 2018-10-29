import matplotlib.pyplot as plt
import pathlib
import numpy as np


def plot_distribution(features, filter_type, verbose=False, iqr_ratio=3):
    """
    Plots the distribution for each of the features in the given.
    The plots are saved in the path directory

    :param features: the array of the features. It is the inverse of the input data. Shape(# of features, # of samples)
    :param filter_type: the type of filtering that was applied to the data
    (ex none, without NaNs). Used for naming the files
    :param verbose: Display information
    :param iqr_ratio: Inter-quartile ratio. This defines the expected minimum and maximum by respectively subtracting
    or adding the inter-quartile range times this ratio to the median. Default value is 3 (Tukey 1977)
    :return: void
    """
    print(features.shape)

    for feat_no, feature in enumerate(features):
        # Compute quartiles
        lower_quartile = np.quantile(feature, 0.25)
        median = np.quantile(feature, 0.5)
        upper_quartile = np.quantile(feature, 0.75)

        # Compute expected minimum and maximum values
        diff_no_outlier = (upper_quartile-lower_quartile)*iqr_ratio
        min_no_outlier = median-diff_no_outlier
        max_no_outlier = median+diff_no_outlier

        # Plot histogram after cutting the expected bounds
        plt.hist(feature, bins=250, facecolor='green', range=(min_no_outlier, max_no_outlier))
        plt.title("Distribution of Feature [" + str(feat_no) + "] - " + filter_type)
        plt.show()
        if verbose:
            print('Min is ', min_no_outlier, ' and max ', max_no_outlier)
            print('Obviating ', (sum(feature < min_no_outlier) + sum(feature > max_no_outlier)) * 100 / len(feature),
                  '% of features for feature', feat_no, ' ', filter_type)
