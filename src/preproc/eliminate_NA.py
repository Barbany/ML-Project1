import numpy as np
from utils.helpers import load_csv_data

# Load RAW data
yb, input_data, ids = load_csv_data('../../data/train.csv')

# Label NANs with numpy built-in indicator
input_data[input_data == -999] = np.nan


for feature in range(input_data.shape[1]):
    print('Feature #', feature, ' has \t',
          '{0:.2f}'.format(np.count_nonzero(np.isnan(input_data[:, feature]))/len(input_data[:, feature])*100),
          '% NANs')

# Eliminate all features with 1 or more NANs
# We could also try to only eliminate those over a given threshold (e.g. 20%)
input_data = input_data[:, ~np.any(np.isnan(input_data), axis=0)]

