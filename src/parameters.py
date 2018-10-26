default_params = {
    'verbose': True,
    'pca': False,
    'correlation': False,
    'results_path': '../results',
    'raw_data': '../data',
    'seed': 123,
    'max_iters': 5000,
    'gamma': 1e-7,
    'batch_size': 8,
    'bias': False,
    'split_jet': True,
    'loss_function': 'logistic',
    'k-fold': 10
}

tag_params = [
    'pca', 'bias', 'loss_function', 'split_jet'
]