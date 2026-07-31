import numpy as np
import sklearn
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict

from config import config
from data_handle import *


def train_bagging(X, y):
    X = preprocessing(X, handle_categorical='One-hot-encoding')

    submodule = getattr(sklearn, config.bagging.base_model.module)
    base_model = getattr(submodule, config.bagging.base_model.name)(**config.bagging.base_model.params)

    bagging_model = BaggingClassifier(estimator=base_model,
                                      random_state=config.general.seed,
                                      **config.bagging.params)

    scores = cross_val_score(bagging_model, X, y, cv=config.training.n_splits)
    model_acc = round(scores.mean(), 2)
    model_std = round(scores.std(), 2)

    bagging_model.fit(X, y)

    return bagging_model, model_acc, model_std


def train_stacking(X, y):
    X = preprocessing(X, handle_categorical='One-hot-encoding')

    # Создаём базовые модели
    base_models = []
    for model in config.stacking.base_models:
        submodule = getattr(sklearn, model.module)
        base_model = getattr(submodule, model.name)(**model.params)
        base_models.append(base_model)

    # Получаем мета-признаки для итоговой модели
    X_meta = np.zeros((X.shape[0], len(base_models)))
    for idx, model in enumerate(base_models):
        X_meta[:, idx] = cross_val_predict(model, X, y, cv=config.training.n_splits)

    # Создаём мета-модель
    submodule = getattr(sklearn, config.stacking.meta_model.module)
    meta_model = getattr(submodule, config.stacking.meta_model.name)(**config.stacking.meta_model.params)

    # Метрики мета-модели
    scores = cross_val_score(meta_model, X, y, cv=config.training.n_splits)
    meta_model_acc = round(scores.mean(), 2)
    meta_model_std = round(scores.std(), 2)

    # Дообучаем для прода
    models = []
    for model in base_models:
        model.fit(X ,y)
        models.append(model)
    models.append(meta_model.fit(X, y))

    return models, meta_model_acc, meta_model_std

def train_stacking_l2(X, y):
    X = preprocessing(X, handle_categorical='One-hot-encoding')

    # Создаём базовые модели
    base_models = []
    for model in config.stacking_l2.base_models:
        submodule = getattr(sklearn, model.module)
        base_model = getattr(submodule, model.name)(**model.params)
        base_models.append(base_model)

    # Получаем мета-признаки для итоговой модели
    X_meta = np.zeros((X.shape[0], len(base_models)))
    for idx, model in enumerate(base_models):
        X_meta[:, idx] = cross_val_predict(model, X, y, cv=config.training.n_splits)

    # Создаём мета-модель
    submodule = getattr(sklearn, config.stacking_l2.meta_model.module)
    meta_model = getattr(submodule, config.stacking_l2.meta_model.name)(**config.stacking.meta_model.params)

    # Метрики мета-модели
    scores = cross_val_score(meta_model, X, y, cv=config.training.n_splits)
    meta_model_acc = round(scores.mean(), 2)
    meta_model_std = round(scores.std(), 2)

    # Дообучаем для прода
    models = []
    for model in base_models:
        model.fit(X ,y)
        models.append(model)
    models.append(meta_model.fit(X, y))

    return models, meta_model_acc, meta_model_std