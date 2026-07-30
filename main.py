import os
import random

import numpy as np
import pandas as pd
import torch

from models_sklearn import *
from models_boost import *
from model_nn import *
from models_ensembles import *

def train(config):
    train_data = pd.read_csv(config.paths.path_to_train)

    # Здесь раньше была предобработка данных, но т.к. каждая модель требует индивидуального подхода
    # к категориальным фичам, данные полномочия были делигированны моделям
    X = train_data.drop(columns=['Survived'])
    y = train_data['Survived']

    model_data = []

    # ↓↓↓ Логистическая регрессия ↓↓↓

    logreg_model, logreg_acc, logreg_std = train_model_sklearn(X, y, model_name='logistic_regression')
    model_data.append(['LogReg', logreg_acc, logreg_std])

    # ↓↓↓ Логистическая регрессия с L1-регуляризацией ↓↓↓

    logreg_l1_model, logreg_l1_acc, logreg_l1_std = train_model_sklearn(X, y, model_name='logistic_regression_l1')
    model_data.append(['LogReg-L1', logreg_l1_acc, logreg_l1_std])

    # ↓↓↓ Логистическая регрессия с L2-регуляризацией ↓↓↓

    logreg_l2_model, logreg_l2_acc, logreg_l2_std = train_model_sklearn(X, y, model_name='logistic_regression_l2')
    model_data.append(['LogReg-L2', logreg_l2_acc, logreg_l2_std])

    # ↓↓↓ Логистическая регрессия с ElasticNet-регуляризацией ↓↓↓

    logreg_en_model, logreg_en_acc, logreg_en_std = train_model_sklearn(X, y, model_name='logistic_regression_elasticnet')
    model_data.append(['LogReg-ElNet', logreg_en_acc, logreg_en_std])

    # ↓↓↓ Метод ближайших соседей KNN ↓↓↓

    knn_model, knn_acc, knn_std = train_model_sklearn(X, y, model_name='knn')
    model_data.append(['KNN', knn_acc, knn_std])

    # ↓↓↓ Решающее дерево DecisionTree ↓↓↓

    dt_model, dt_acc, dt_std = train_model_sklearn(X, y, model_name='decision_tree')
    model_data.append(['DecisionTree', dt_acc, dt_std])

    # ↓↓↓ Случайный лес RandomForest ↓↓↓

    rf_model, rf_acc, rf_std = train_model_sklearn(X, y, model_name='random_forest')
    model_data.append(['RandomForest', rf_acc, rf_std])

    # ↓↓↓ Бустинг CatBoost ↓↓↓

    catboost_model, catboost_acc, catboost_std = train_catboost(X, y)
    model_data.append(['CatBoost', catboost_acc, catboost_std])

    # ↓↓↓ Бустинг LightGBM ↓↓↓

    lightgbm_model, lightgbm_acc, lightgbm_std = train_lightgbm(X, y)
    model_data.append(['LightGBM', lightgbm_acc, lightgbm_std])

    # ↓↓↓ Бустинг XGBoost ↓↓↓

    xgboost_model, xgboost_acc, xgboost_std = train_xgboost(X, y)
    model_data.append(['XGBoost', xgboost_acc, xgboost_std])

    # ↓↓↓ Нейронная сеть ↓↓↓

    nn_model, nn_acc = train_nn(X, y)
    model_data.append(['NeuralNetwork', nn_acc, '—'])
    
    # ↓↓↓ Ансамбль Bagging ↓↓↓

    bagging_model, bagging_acc, bagging_std = train_bagging(X, y)
    model_data.append(['Bagging', bagging_acc, bagging_std])

    # ↓↓↓ Ансамбль Stacking via LogReg ↓↓↓

    stacking_models, stacking_acc, stacking_std = train_stacking(X, y)
    model_data.append(['Stacking via LogReg', stacking_acc, stacking_std])

    # ↓↓↓ Ансамбль Stacking via LogReg-L2 ↓↓↓

    stacking_l2_models, stacking_l2_acc, stacking_l2_std = train_stacking_l2(X, y)
    model_data.append(['Stacking via LogReg-L2', stacking_l2_acc, stacking_l2_std])

    # Total output
    header = f'{"Approach":<22} | {"CV":>10} | {"CV STD":>10}'
    print(header)
    print('-' * len(header))

    for model_name, model_acc, model_std in model_data:
        model_std = f'{model_std:.2f}' if isinstance(model_std, (int, float)) else str(model_std)
        model_acc = f'{model_acc:.2f}'
        print(f'{model_name:<22} | {model_acc:>10} | {model_std:>10}')


def main(config):
    # Устанавливаем детерменированность
    random.seed(config.general.seed)
    np.random.seed(config.general.seed)
    torch.manual_seed(config.general.seed)
    torch.cuda.manual_seed(config.general.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(config.general.seed)

    train(config)


if __name__ == "__main__":
    main(config)