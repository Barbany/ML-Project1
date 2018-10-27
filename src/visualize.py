import seaborn as sns
import matplotlib.pyplot as plt
import pathlib

from preproc.visualizations import plot_distribution
from preproc.data_clean import *

directory = '../plots/no_outliers/'
pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

_, input_data, _, _, _, _ = load_csv_split_jet('../data/train.csv', '../data/test.csv', verbose=True)
for jet in range(len(input_data)):
    plot_distribution(input_data[jet].T, 'jet_filtering_'+str(jet), directory, verbose=True)