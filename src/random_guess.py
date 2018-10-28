import numpy as np
from utils.helpers import load_csv_data
from utils.costs import accuracy

labels, _, _ = load_csv_data('../data/train.csv')

acc = []

for seed in range(123,1123,50):
    np.random.seed(seed=seed)
    predictions = 2*np.random.randint(2, size=len(labels))-1
    acc.append(accuracy(labels, predictions))
print('Accuracy for random guess is ', np.mean(acc), '+-', np.std(acc))
