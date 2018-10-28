default_params = {
    'correlation': False,
    'results_path': '../results',
    'raw_data': '../data',
    'seed': 123,
    'max_iters': 6000,
    'gamma': 1e-5,
    'batch_size': 8,
    'split_jet': True,
    'split_mass': True,
    'loss_function': 'logistic',
    'k-fold': 10,
    'outliers': False,
    'pca': False
}

tag_params = [
    'pca', 'loss_function', 'split_jet', 'split_mass', 'outliers'
]
