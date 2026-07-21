import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from config import config

def train_model_sklearn(X, y, model_name='logistic_regression'):
    model = None
    if model_name == 'logistic_regression':
        model = Pipeline([
            ('scale', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000))
        ])
    elif model_name == 'logistic_regression_l1':
        model = Pipeline([
            ('scale', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000,
                                         penalty='l1',
                                         solver='liblinear',
                                         C=1))
        ])
    elif model_name == 'logistic_regression_l2':
        model = Pipeline([
            ('scale', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000,
                                         penalty='l2',
                                         solver='lbfgs',
                                         C=0.01))
        ])
    elif model_name == 'logistic_regression_elasticnet':
        model = Pipeline([
            ('scale', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000,
                                         penalty='elasticnet',
                                         l1_ratio=0.5,
                                         solver='saga',
                                         C=1))
        ])
    elif model_name == 'knn':
        model = Pipeline([
            ('scale', StandardScaler()),
            ('model', KNeighborsClassifier(n_neighbors=6,
                                           weights='uniform',
                                           metric='chebyshev'))
        ])
    elif model_name == 'decision_tree':
        model = DecisionTreeClassifier(max_depth=4,
                                       random_state=config.general.seed,
                                       criterion='gini',
                                       splitter='best')
    elif model_name == 'random_forest':
        model = RandomForestClassifier(n_estimators=50,
                                       random_state=config.general.seed,
                                       min_samples_leaf=2)

    skf = StratifiedKFold(n_splits=config.split.n_splits, shuffle=True, random_state=config.general.seed)
    scores = []

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model.fit(X_train, y_train)

        cur_accuracy = model.score(X_val, y_val)
        scores.append(cur_accuracy)

    model_acc = round(sum(scores) / len(scores), 2)
    model_std = round(np.array(scores).std(), 2)

    model.fit(X, y)
    return model, model_acc, model_std