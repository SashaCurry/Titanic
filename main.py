import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import config
from data_handle import preprocessing

def train(config):
    train_data = pd.read_csv(config.paths.path_to_train)

    # Предобработка данных

    train_data = preprocessing(train_data)
    X = train_data.drop(columns=['Survived'])
    y = train_data['Survived']

    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size=0.2,
                                                      random_state=config.general.seed,
                                                      shuffle=True
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.general.seed)

    # Логистическая регрессия

    logreg_model = Pipeline([
        ('scale', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000))
    ])
    logreg_scores = []

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        logreg_model.fit(X_train, y_train)

        accuracy = logreg_model.score(X_val, y_val)
        logreg_scores.append(accuracy)

    logreg_acc = round(sum(logreg_scores) / len(logreg_scores), 2)
    logreg_std = round(np.array(logreg_scores).std(), 2)
    print(f'LogReg \nMean Score: {logreg_acc}, Std Score: {logreg_std}')

    logreg_model.fit(X, y)

    # Логистическая регрессия с L1-регуляризацией

    logreg_l1_model = Pipeline([
        ('scale', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000,
                                     penalty='l1',
                                     solver='liblinear',
                                     C=1))
    ])
    logreg_l1_scores = []

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        logreg_l1_model.fit(X_train, y_train)

        accuracy = logreg_l1_model.score(X_val, y_val)
        logreg_l1_scores.append(accuracy)

    logreg_l1_acc = round(sum(logreg_l1_scores) / len(logreg_l1_scores), 2)
    logreg_l1_std = round(np.array(logreg_l1_scores).std(), 2)
    print(f'\nLogReg with L1-reg \nMean Score: {logreg_l1_acc}, Std Score: {logreg_l1_std}')

    logreg_l1_model.fit(X, y)

    # Логистическая регрессия с L2-регуляризацией

    logreg_l2_model = Pipeline([
        ('scale', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000,
                                     penalty='l2',
                                     solver='lbfgs',
                                     C=0.01))
    ])
    logreg_l2_scores = []

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        logreg_l2_model.fit(X_train, y_train)

        accuracy = logreg_l2_model.score(X_val, y_val)
        logreg_l2_scores.append(accuracy)

    logreg_l2_acc = round(sum(logreg_l2_scores) / len(logreg_l2_scores), 2)
    logreg_l2_std = round(np.array(logreg_l2_scores).std(), 2)
    print(f'\nLogReg with L2-reg \nMean Score: {logreg_l2_acc}, Std Score: {logreg_l2_std}')

    logreg_l2_model.fit(X, y)

    # Логистическая регрессия с ElasticNet-регуляризацией

    logreg_en_model = Pipeline([
        ('scale', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000,
                                     penalty='elasticnet',
                                     l1_ratio=0.5,
                                     solver='saga',
                                     C=1))
    ])
    logreg_en_scores = []

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        logreg_en_model.fit(X_train, y_train)

        accuracy = logreg_en_model.score(X_val, y_val)
        logreg_en_scores.append(accuracy)

    logreg_en_acc = round(sum(logreg_en_scores) / len(logreg_en_scores), 2)
    logreg_en_std = round(np.array(logreg_en_scores).std(), 2)
    print(f'\nLogReg with ElNet-reg \nMean Score: {logreg_en_acc}, Std Score: {logreg_en_std}')

    logreg_l2_model.fit(X, y)

    # KNN



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