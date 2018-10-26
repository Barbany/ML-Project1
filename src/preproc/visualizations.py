import seaborn as sns
import matplotlib.pyplot as plt
import pathlib

from src.preproc.data_clean import load_csv_data_no_na, load_csv_split_jet


def plot_distribution(features, filter_type, path):
    '''
    Plots the distribution for each of the features in the given.
    The plots are saved in the path directory

    :param features: the array of the features. It is the inverse of the input data.
                    Shape(# of features, # of samples)
    :param filter_type: the type of filtering that was applied to the data
                    (ex none, without NaNs). Used for naming the files
    :param path: the path to save the plots
    :return: void
    '''
    sns.set(color_codes=True)
    print(features.shape)

    feat_no = 0
    for feature in features:
        # draw a histogram and fit a kernel density estimate (KDE).
        sns.distplot(feature)
        plt.title("Distribution of Feature [" + str(feat_no) + "] - " + filter_type)
        plt.savefig(path+"distribution_"+str(feat_no)+"_("+filter_type+").png")
        # plt.show()
        feat_no += 1

directory = '../../plots/'
pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

_, input_data, _, _, _, _ = load_csv_data_no_na('../../data/train.csv', '../../data/test.csv')
plot_distribution(input_data.T, 'no_na', directory)

_, input_data, _, _, _, _ = load_csv_split_jet('../../data/train.csv', '../../data/test.csv', verbose=True)
for jet in range(len(input_data)):
    plot_distribution(input_data[jet].T, 'jet_filtering_'+str(jet), directory)
