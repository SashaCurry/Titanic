from catboost import CatBoostClassifier, cv, Pool
import lightgbm as lgb
from lightgbm import LGBMClassifier

from config import config
from data_handle import *


def train_catboost(X, y):
    X = preprocessing(X, handle_categorical='None')

    data_pool = Pool(data=X, label=y,
                     cat_features=['Age_Group', 'Fare_Range', 'Alone', 'Sex', 'Embarked', 'Honorifics'])

    params={
        'iterations': 100,
        'learning_rate': 0.075,
        'depth': 5,
        'loss_function': 'Logloss',
        'eval_metric': 'Accuracy'
    }

    cv_data = cv(pool=data_pool,
                 params=params,
                 fold_count=5,
                 shuffle=True,
                 partition_random_seed=config.general.seed,
                 stratified=True,
                 early_stopping_rounds=50,
                 logging_level='Silent')

    model_acc = round(cv_data.tail(1)['test-Accuracy-mean'].item(), 2)
    model_std = round(cv_data.tail(1)['test-Accuracy-std'].item(), 2)

    model = CatBoostClassifier(iterations=100,
                               learning_rate=0.075,
                               depth=5,
                               loss_function='Logloss',
                               cat_features=['Age_Group', 'Fare_Range', 'Alone', 'Sex', 'Embarked', 'Honorifics'])
    model.fit(X, y, verbose=False)

    return model, model_acc, model_std


def train_lightgbm(X, y):
    dataset = lgb.Dataset(data=X,
                          label=y,
                          feature_name=X.columns.tolist(),
                          categorical_feature=['Age_Group', 'Fare_Range', 'Alone', 'Sex', 'Embarked', 'Honorifics'])

    params = {'objective': 'binary',
              'leraning_rage': 0.1,
              'num_leaves': 31,
              'metric': 'accuracy',
              'num_boost_round': 100,
              'verbosity': -1
    }

    cv_output = lgb.cv(params=params,
                       train_set=dataset,
                       nfold=5,
                       stratified=True,
                       shuffle=True,
                       return_cvbooster=True)

    model_acc = round(cv_output['accuracy-mean'].items(), 2)
    model_std = round(cv_output['accuracy-std'].items(), 2)

    model = LGBMClassifier(objective='binary', n_estimators=10)
    model.fit(X, y)

    return model, model_acc, model_std