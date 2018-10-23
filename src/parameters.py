default_params = {
    'verbose': False,
    'pca': True,
    'correlation': False,
    'results_path': '../results',
    'raw_data': '../data',
    'seed': 123,
    'max_iters': 50,
    'gamma': 1e-5,
    'batch_size': 8,
    'bias': True,
    'split_jet': True,
    'loss_function': 'logistic'
}

tag_params = [
    'pca', 'bias', 'loss_function', 'split_jet'
]