import seaborn as sns
import matplotlib.pyplot as plt
import pathlib

from preproc.visualizations import plot_distribution
from preproc.data_clean import *

directory = '../plots/'
pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

_, input_data, _, _, _, _ = load_csv_data_no_na('../data/train.csv', '../data/test.csv')
plot_distribution(input_data.T, 'no_na', directory)

_, input_data, _, _, _, _ = load_csv_split_jet('../data/train.csv', '../data/test.csv', verbose=True)
for jet in range(len(input_data)):
    plot_distribution(input_data[jet].T, 'jet_filtering_'+str(jet), directory)
