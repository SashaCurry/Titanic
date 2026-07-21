from catboost import CatBoostClassifier, cv, Pool

from config import config


def train_catboost(X, y):
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
    model.fit(X, y, early_stopping_rounds=50, verbose=False)

    return model, model_acc, model_std