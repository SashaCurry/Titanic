import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from config import config
from data_handle import *


def train_bagging(X, y):
    X = preprocessing(X, handle_categorical='One-hot-encoding')

    base_model = DecisionTreeClassifier(max_depth=6)
    bagging_model = BaggingClassifier(estimator=base_model,
                                      n_estimators=100,
                                      random_state=config.general.seed)

    scores = cross_val_score(bagging_model, X, y, cv=5)
    model_acc = round(scores.mean(), 2)
    model_std = round(scores.std(), 2)

    bagging_model.fit(X, y)

    return bagging_model, model_acc, model_std


def train_stacking(X, y):
    X = preprocessing(X, handle_categorical='One-hot-encoding')

    # Создаём базовые модели
    model_1 = KNeighborsClassifier(n_neighbors=6)
    model_2 = DecisionTreeClassifier(max_depth=3, random_state=config.general.seed)
    model_3 = RandomForestClassifier(n_estimators=100, random_state=config.general.seed)
    base_models = [model_1, model_2, model_3]

    # Получаем мета-признаки для итоговой модели
    X_meta = np.zeros((X.shape[0], len(base_models)))
    for idx, model in enumerate(base_models):
        X_meta[:, idx] = cross_val_predict(model, X, y, cv=5)

    # Создаём мета-модель
    meta_model = LogisticRegression(max_iter=1000)

    # Метрики мета-модели
    scores = cross_val_score(meta_model, X, y, cv=5)
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
    model_1 = KNeighborsClassifier(n_neighbors=6)
    model_2 = DecisionTreeClassifier(max_depth=3, random_state=config.general.seed)
    model_3 = RandomForestClassifier(n_estimators=100, random_state=config.general.seed)
    base_models = [model_1, model_2, model_3]

    # Получаем мета-признаки для итоговой модели
    X_meta = np.zeros((X.shape[0], len(base_models)))
    for idx, model in enumerate(base_models):
        X_meta[:, idx] = cross_val_predict(model, X, y, cv=5)

    # Создаём мета-модель
    meta_model = LogisticRegression(max_iter=1000, penalty='l2')

    # Метрики мета-модели
    scores = cross_val_score(meta_model, X, y, cv=5)
    meta_model_acc = round(scores.mean(), 2)
    meta_model_std = round(scores.std(), 2)

    # Дообучаем для прода
    models = []
    for model in base_models:
        model.fit(X ,y)
        models.append(model)
    models.append(meta_model.fit(X, y))

    return models, meta_model_acc, meta_model_std