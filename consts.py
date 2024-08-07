# Caminhos de arquivos
CSV_FILE = './IoTProcessed_Data.csv'

#Configurações de modelos
MPL_CONFIGURER = [
        {
            'hidden_layer_sizes': 7,
            'activation': 'logistic',
            'max_iter': 500,
            'random_state': 1,
            'alpha': 0.01,
            'momentum': 0.9
        },
        {
            'hidden_layer_sizes': 7,
            'activation': 'relu',
            'max_iter': 500,
            'random_state': 1,
            'alpha': 0.01,
            'momentum': 0.9
        },
        {
            'hidden_layer_sizes': 7,
            'activation': 'tanh',
            'max_iter': 500,
            'random_state': 1,
            'alpha': 0.01,
            'momentum': 0.9
        },
        {
            'hidden_layer_sizes': 10,
            'activation': 'logistic',
            'max_iter': 700,
            'random_state': 1,
            'alpha': 0.02,
            'momentum': 0.5
        },
        {
            'hidden_layer_sizes': 7,
            'activation': 'relu',
            'max_iter': 1000,
            'random_state': 1,
            'alpha': 0.01,
            'momentum': 0.9
        }
]

DESCION_TREE_CONFIGURER = [
        {
            'criterion' :'squared_error',
            'max_depth' : 5,
            'min_samples_split' : 10,
            'min_samples_leaf' : 5,
            'max_features' : 'sqrt',
            'random_state' : 42
        },
        {
            'criterion' :'friedman_mse',
            'max_depth' : 5,
            'min_samples_split' : 10,
            'min_samples_leaf' : 5,
            'max_features' : 'sqrt',
            'random_state' : 42
        },
        {
            'criterion' :'absolute_error',
            'max_depth' : 5,
            'min_samples_split' : 10,
            'min_samples_leaf' : 5,
            'max_features' : 'sqrt',
            'random_state' : 42
        },
        {
            'criterion' :'squared_error',
            'max_depth' : 10,
            'min_samples_split' : 15,
            'min_samples_leaf' : 7,
            'max_features' : 'log2',
            'random_state' : 31
        },
        {
            'criterion' :'poisson',
            'max_depth' : 13,
            'min_samples_split' : 9,
            'min_samples_leaf' : 10,
            'max_features' : 'log2',
            'random_state' : 57
        }
]

KNEIGHBORS_CONFIGURER = [
        {
            'n_neighbors' : 5,
            'weights' : 'distance',
            'algorithm' : 'auto',
            'leaf_size' : 30,
            'metric' : 'euclidean',
            'p' : 2,
            'n_jobs' : -1
        },
        {
            'n_neighbors' : 10,
            'weights' : 'uniform',
            'algorithm' : 'kd_tree',
            'leaf_size' : 30,
            'metric' : 'minkowski',
            'p' : 1,
            'n_jobs' : -1
        },
        {
            'n_neighbors' : 20,
            'weights' : 'distance',
            'algorithm' : 'auto',
            'leaf_size' : 30,
            'metric' : 'euclidean',
            'p' : 2,
            'n_jobs' : -1
        },
        {
            'n_neighbors' : 20,
            'weights' : 'distance',
            'algorithm' : 'auto',
            'leaf_size' : 30,
            'metric' : 'manhattan',
            'p' : 1,
            'n_jobs' : -1
        },
        {
            'n_neighbors' : 20,
            'weights' : 'uniform',
            'algorithm' : 'ball_tree',
            'leaf_size' : 35,
            'metric' : 'manhattan',
            'p' : 2,
            'n_jobs' : -1
        }
]

LINEAR_REGRESSION_CONFIGURER = [
        {
            'fit_intercept': True, 
            'copy_X' : True, 
            'n_jobs' : -1
        },
        {
            'fit_intercept': False, 
            'copy_X' : True, 
            'n_jobs' : -1
        },
        {
            'fit_intercept': True, 
            'copy_X' : False, 
            'n_jobs' : -1
        },
        {
            'fit_intercept': False,
            'copy_X' : False, 
            'n_jobs' : -1
        },  
        {
            'fit_intercept': True, 
            'copy_X' : True, 
            'n_jobs' : 1
        }
]