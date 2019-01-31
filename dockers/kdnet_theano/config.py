config = {
    # General
    'batchsize': 227,
    'shuffle': True,
    # Augmentations
    'flip': False,
    'ascale': True, 'as_min': 0.6667, 'as_max': 1.5,
    'rotate': False, 'r_positions': 12, 'test_pos': None,
    'translate': True, 't_rate': 0.1,
    # Point clouds and kd-trees generation
    'steps': 10, # also control the depth of the network
    'dim': 3,
    'lim': 1,
    'det': False,
    'gamma': 10.,
    # NN options
    'input_features': 'all', # 'all' for point coordinates, 'no' for feeding 1's as point features
    'n_f': [16, 
            32,  32,  
            64,  64,  
            128, 128, 
            256, 256, 
            512, 128], # representation sizes
    'n_output': 40,
    'l2': 1e-3,
    'lr': 1e-3,
    'n_ens': 4 
}