import os
import random

import numpy as np
import pandas as pd
import torch

from data_handle import *
from models_sklearn import *
from models_boost import *

def train(config):
    train_data = pd.read_csv(config.paths.path_to_train)

    # ↓↓↓ Предобработка данных ↓↓↓

    train_data_handled = preprocessing(train_data, handle_categorical=True)
    X = train_data_handled.drop(columns=['Survived'])
    y = train_data_handled['Survived']

    # ↓↓↓ Логистическая регрессия ↓↓↓

    logreg_model, logreg_acc, logreg_std = train_model_sklearn(X, y, model_name='logistic_regression')
    print(f'\nLogisticRegression \nMean Score: {logreg_acc}, Std Score: {logreg_std}')

    # ↓↓↓ Логистическая регрессия с L1-регуляризацией ↓↓↓

    logreg_l1_model, logreg_l1_acc, logreg_l1_std = train_model_sklearn(X, y, model_name='logistic_regression_l1')
    print(f'\nLogisticRegression with L1-reg \nMean Score: {logreg_l1_acc}, Std Score: {logreg_l1_std}')

    # ↓↓↓ Логистическая регрессия с L2-регуляризацией ↓↓↓

    logreg_l2_model, logreg_l2_acc, logreg_l2_std = train_model_sklearn(X, y, model_name='logistic_regression_l2')
    print(f'\nLogisticRegressin with L2-reg \nMean Score: {logreg_l2_acc}, Std Score: {logreg_l2_std}')

    # ↓↓↓ Логистическая регрессия с ElasticNet-регуляризацией ↓↓↓

    logreg_en_model, logreg_en_acc, logreg_en_std = train_model_sklearn(X, y, model_name='logistic_regression_elasticnet')
    print(f'\nLogisticRegression with ElNet-reg \nMean Score: {logreg_en_acc}, Std Score: {logreg_en_std}')

    # ↓↓↓ Метод ближайших соседей KNN ↓↓↓

    knn_model, knn_acc, knn_std = train_model_sklearn(X, y, model_name='knn')
    print(f'\nKNN \nMean Score: {knn_acc}, Std Score: {knn_std}')

    # ↓↓↓ Решающее дерево DecisionTree ↓↓↓

    dt_model, dt_acc, dt_std = train_model_sklearn(X, y, model_name='decision_tree')
    print(f'\nDecisionTree \nMean Score: {dt_acc}, Std Score: {dt_std}')

    # ↓↓↓ Случайный лес RandomForest ↓↓↓

    rf_model, rf_acc, rf_std = train_model_sklearn(X, y, model_name='random_forest')
    print(f'\nRandomForest \nMean Score: {rf_acc}, Std Score: {rf_std}')

    # ↓↓↓ Бустинг CatBoost ↓↓↓

    # т.к. CatBoost может самостоятельно обрабатывать категориальные фичи, стоит использовать эту возможность
    train_data_non_handled = preprocessing(train_data, handle_categorical=False)
    X_non_handled = train_data_non_handled.drop(columns=['Survived'])
    y_non_handled = train_data_non_handled['Survived']

    catboost_model, catboost_acc, catboost_std = train_catboost(X_non_handled, y_non_handled)
    print(f'\nCatBoost \nMean Score: {catboost_acc}, Std Score: {catboost_std}')

    # ↓↓↓ Бустинг LightGBM ↓↓↓


def test(config):
    pass


def main(config):
    # Устанавливаем детерменированность
    random.seed(config.general.seed)
    np.random.seed(config.general.seed)
    torch.manual_seed(config.general.seed)
    torch.cuda.manual_seed(config.general.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(config.general.seed)

    if config.training.is_train:
        train(config)
    else:
        test(config)


if __name__ == "__main__":
    main(config)