default_params = {
    'verbose': True,
    'pca': False,
    'correlation': False,
    'results_path': '../results',
    'raw_data': '../data',
    'seed': 123,
    'max_iters': 10000,
    'gamma': 0.05,
    'batch_size': 8,
    'bias': False,
    'split_jet': True,
    'loss_function': 'logistic',
    'k-fold': 10
}

tag_params = [
    'pca', 'bias', 'loss_function', 'split_jet'
]