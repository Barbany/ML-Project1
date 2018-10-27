default_params = {
    'correlation': False,
    'results_path': '../results',
    'raw_data': '../data',
    'seed': 123,
    'max_iters': 1000,
    'gamma': 1e-7,
    'batch_size': 8,
    'split_jet': True,
    'split_mass': True,
    'loss_function': 'logistic',
    'k-fold': 10
}

tag_params = [
    'pca', 'loss_function', 'split_jet', 'split_mass'
]
