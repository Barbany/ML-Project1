"""Guess labels with random values to get a reference of the accuracy and compare to our models"""

import numpy as np
import os

from utils.helpers import load_csv_data
from utils.costs import accuracy
from utils.argument_parser import default_params

# Load labels from train data. Note that features are not relevant with random guess
labels, _, _ = load_csv_data(os.path.join(params['raw_data'], 'train.csv'))

acc = []

# Try 50 different seeds to obtain a certain variety of accuracies
for seed in range(123, 1123, 50):
    # Set the seed for the random number generator
    np.random.seed(seed=seed)

    # Obtain random ints {0, 1} and transform them into {-1, 1}
    predictions = 2*np.random.randint(2, size=len(labels))-1

    # Compute accuracy of random predictions
    acc.append(accuracy(labels, predictions))

print('Accuracy for random guess is ', np.mean(acc), '+-', np.std(acc))
