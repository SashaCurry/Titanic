import os
import random

import numpy as np
import pandas as pd
import torch

from models_sklearn import *
from models_boost import *

def train(config):
    train_data = pd.read_csv(config.paths.path_to_train)

    # Здесь раньше была предобработка данных, но т.к. каждная модель требует индивидуального подхода
    # к категориальным фичам, данные полномочия были делигированны моделям
    # TODO: Может стоит вернуть обратно, если каждая модель будет использовать one-hot-encoding (?)
    X = train_data.drop(columns=['Survived'])
    y = train_data['Survived']

    # ↓↓↓ Логистическая регрессия ↓↓↓

    logreg_model, logreg_acc, logreg_std = train_model_sklearn(X, y, model_name='logistic_regression')
    print(f'\nLogisticRegression \nMean Accuracy: {logreg_acc}, Std Accuracy: {logreg_std}')

    # ↓↓↓ Логистическая регрессия с L1-регуляризацией ↓↓↓

    logreg_l1_model, logreg_l1_acc, logreg_l1_std = train_model_sklearn(X, y, model_name='logistic_regression_l1')
    print(f'\nLogisticRegression with L1-reg \nMean Accuracy: {logreg_l1_acc}, Std Accuracy: {logreg_l1_std}')

    # ↓↓↓ Логистическая регрессия с L2-регуляризацией ↓↓↓

    logreg_l2_model, logreg_l2_acc, logreg_l2_std = train_model_sklearn(X, y, model_name='logistic_regression_l2')
    print(f'\nLogisticRegressin with L2-reg \nMean Accuracy: {logreg_l2_acc}, Std Accuracy: {logreg_l2_std}')

    # ↓↓↓ Логистическая регрессия с ElasticNet-регуляризацией ↓↓↓

    logreg_en_model, logreg_en_acc, logreg_en_std = train_model_sklearn(X, y, model_name='logistic_regression_elasticnet')
    print(f'\nLogisticRegression with ElNet-reg \nMean Accuracy: {logreg_en_acc}, Std Accuracy: {logreg_en_std}')

    # ↓↓↓ Метод ближайших соседей KNN ↓↓↓

    knn_model, knn_acc, knn_std = train_model_sklearn(X, y, model_name='knn')
    print(f'\nKNN \nMean Accuracy: {knn_acc}, Std Accuracy: {knn_std}')

    # ↓↓↓ Решающее дерево DecisionTree ↓↓↓

    dt_model, dt_acc, dt_std = train_model_sklearn(X, y, model_name='decision_tree')
    print(f'\nDecisionTree \nMean Accuracy: {dt_acc}, Std Accuracy: {dt_std}')

    # ↓↓↓ Случайный лес RandomForest ↓↓↓

    rf_model, rf_acc, rf_std = train_model_sklearn(X, y, model_name='random_forest')
    print(f'\nRandomForest \nMean Accuracy: {rf_acc}, Std Accuracy: {rf_std}')

    # ↓↓↓ Бустинг CatBoost ↓↓↓

    catboost_model, catboost_acc, catboost_std = train_catboost(X, y)
    print(f'\nCatBoost \nMean Accuracy: {catboost_acc}, Std Accuracy: {catboost_std}')

    # ↓↓↓ Бустинг LightGBM ↓↓↓

    lightgbm_model, lightgbm_acc, lightgbm_std = train_lightgbm(X, y)
    print(f'\nLightGBM \nMean Accuracy: {lightgbm_acc}, Std Accuracy: {lightgbm_std}')


    # ↓↓↓ Бустинг XGBoost ↓↓↓

    xgboost_model, xgboost_acc, xgboost_std = train_xgboost(X, y)
    print(f'\nXGBoost \nMean Accuracy: {xgboost_acc}, Std Accuracy: {xgboost_std}')



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