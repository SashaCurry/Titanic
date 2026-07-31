from omegaconf import OmegaConf

config = {
    'general': {
        'experiment_name': 'Titanic_v0.1',
        'seed': 0xFACED,
        'num_classes': 2
    },
    'paths': {
        'path_to_train': './content/train.csv',
        'path_to_test': './content/test.csv'
    },
    'training': {
        'device': 'cuda',
        'n_splits': 5
    },
    'logreg': {
        'max_iter': 1000,
    },
    'logreg_l1': {
        'max_iter': 1000,
        'penalty': 'l1',
        'solver': 'liblinear',
        'C': 1
    },
    'logreg_l2': {
        'max_iter': 1000,
        'penalty': 'l2',
        'solver': 'lbfgs',
        'C': 0.01
    },
    'logreg_elnet': {
        'max_iter': 1000,
        'penalty': 'elasticnet',
        'l1_ratio': 0.5,
        'solver': 'saga',
        'C': 1
    },
    'knn': {
        'n_neighbors': 6,
        'weights': 'uniform',
        'metric': 'chebyshev'
    },
    'decision_tree': {
        'max_depth': 4,
        'criterion': 'gini',
        'splitter': 'best'
    },
    'random_forest': {
        'n_estimators': 50,
        'min_samples_leaf': 2
    },
    'catboost': {
        'iterations': 100,
        'learning_rate': 0.075,
        'depth': 5,
        'loss_function': 'Logloss'
    },
    'lightgbm': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'num_leaves': 31
    },
    'xgboost': {
        'num_boost_round': 1000,
        'learning_rate': 0.1,
        'max_depth': 5,
        'subsample': 0.75
    },
    'neural_network': {
        'num_epochs': 20,
        'loss_fn': {
            'name': 'BCELoss',
            'params': {
            }
        },
        'optimizer': {
            'name': 'Adam',
            'params': {
                'lr': 0.001
            }
        },
        'scheduler': {
            'name': 'CosineAnnealingLR',
            'params': {
                'T_max': 100,
                'eta_min': 1e-6
            }
        }
    },
    'bagging': {
        'base_model': {
            'module': 'tree',
            'name': 'DecisionTreeClassifier',
            'params': {
                'max_depth': 6
            }
        },
        'params': {
            'n_estimators': 100
        }
    },
    'stacking': {
        'base_models': [
            {
                'module': 'neighbors',
                'name': 'KNeighborsClassifier',
                'params': {
                    'n_neighbors': 6
                }
            },
            {
                'module': 'tree',
                'name': 'DecisionTreeClassifier',
                'params': {
                    'max_depth': 3
                }
            },
            {
                'module': 'ensemble',
                'name': 'RandomForestClassifier',
                'params': {
                    'n_estimators': 100
                }
            }
        ],
        'meta_model': {
            'module': 'linear_model',
            'name': 'LogisticRegression',
            'params': {
                'max_iter': 1000
            }
        }
    },
    'stacking_l2': {
        'base_models': [
            {
                'module': 'neighbors',
                'name': 'KNeighborsClassifier',
                'params': {
                    'n_neighbors': 6
                }
            },
            {
                'module': 'tree',
                'name': 'DecisionTreeClassifier',
                'params': {
                    'max_depth': 3
                }
            },
            {
                'module': 'ensemble',
                'name': 'RandomForestClassifier',
                'params': {
                    'n_estimators': 100
                }
            }
        ],
        'meta_model': {
            'module': 'linear_model',
            'name': 'LogisticRegression',
            'params': {
                'max_iter': 1000,
                'penalty': 'l2'
            }
        }
    }
}

config = OmegaConf.create(config)