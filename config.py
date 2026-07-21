from omegaconf import OmegaConf

config = {
    'general': {
        'experiment_name': 'None',
        'seed': 0xFACED,
        'num_classes': 250
    },
    'paths': {
        'path_to_train': './content/train.csv',
        'path_to_test': './content/test.csv'
    },
    'training': {
        'is_train': True,
    },
    'dataloader_params': {
    },
    'split': {
        'n_splits': 5
    },
    'augmentation': {
    },
    'optimizer': {
    },
    'scheduler': {
    },
    'loss': {
    },
    'metric': {
    },
    'logging': {
    }
}

config = OmegaConf.create(config)