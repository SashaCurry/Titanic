import catboost as cb
import lightgbm as lgb
import xgboost as xgb

from config import config
from data_handle import *


def train_catboost(X, y):
    X = preprocessing(X, handle_categorical='None')

    data_pool = cb.Pool(data=X, label=y,
                        cat_features=['Age_Group', 'Fare_Range', 'Alone', 'Sex', 'Embarked', 'Honorifics'])

    params={**config.catboost.params,
            'eval_metric': 'Accuracy'}

    cv_data = cb.cv(
        pool=data_pool,
        params=params,
        fold_count=config.training.n_splits,
        shuffle=True,
        partition_random_seed=config.general.seed,
        stratified=True,
        early_stopping_rounds=50,
        logging_level='Silent'
    )

    model_acc = round(cv_data.tail(1)['test-Accuracy-mean'].item(), 2)
    model_std = round(cv_data.tail(1)['test-Accuracy-std'].item(), 2)

    model = cb.CatBoostClassifier(
        **config.catboost.params,
        cat_features=['Age_Group', 'Fare_Range', 'Alone', 'Sex', 'Embarked', 'Honorifics']
    )
    model.fit(X, y, verbose=False)

    return model, model_acc, model_std


def train_lightgbm(X, y):
    X = preprocessing(X, handle_categorical='None')

    dataset = lgb.Dataset(
        data=X,
        label=y,
        feature_name=X.columns.tolist(),
        categorical_feature=['Age_Group', 'Fare_Range', 'Alone', 'Sex', 'Embarked', 'Honorifics']
    )

    params = {
        **config.lightgbm.params,
        'objective': 'binary',
        'metric': 'binary_error',
        'verbosity': -1,
    }

    cv_output = lgb.cv(
        params=params,
        train_set=dataset,
        nfold=config.training.n_splits,
        stratified=True,
        shuffle=True
    )

    model_acc = round(1 - cv_output['valid binary_error-mean'][-1], 2)
    model_std = round(cv_output['valid binary_error-stdv'][-1], 2)

    model = lgb.LGBMClassifier(
        objective='binary',
        **config.lightgbm.params
    )
    model.fit(X, y)

    return model, model_acc, model_std


def train_xgboost(X ,y):
    X = preprocessing(X, handle_categorical='None')

    dataset = xgb.DMatrix(data=X, label=y, enable_categorical=True)

    # В XGBoost при кросс-валидации параметр num_boost_round
    # указывается отдельно, поэтому приходится его явно отделить
    params_from_config = dict(**config.xgboost.params)
    num_boost_round = params_from_config.pop('num_boost_round')

    params = {
        **params_from_config,
        'objective': 'binary:logistic',
        'tree_method': 'hist',
    }

    cv_result = xgb.cv(
        params=params,
        dtrain=dataset,
        num_boost_round=num_boost_round,
        nfold=config.training.n_splits,
        metrics='error',
        early_stopping_rounds=10,
    )

    model_acc = round(1 - cv_result.tail(1)['test-error-mean'].item(), 2)
    model_std = round(cv_result.tail(1)['test-error-std'].item(), 2)

    model = xgb.XGBClassifier(
        **params_from_config,
        objective='binary:logistic',
        n_estimators=num_boost_round,
        enable_categorical=True,
        tree_method='hist'
    )
    model.fit(X, y)

    return model, model_acc, model_std


def test_boost(X, model, model_name):
    X_test = preprocessing(X, handle_categorical='None')

    preds = model.predict(X_test)

    df = pd.DataFrame({'PassengerId': X['PassengerId'],
                       'Survived': preds})
    df.to_csv(path_or_buf=f'{config.paths.path_save_csv}{model_name}_preds.csv',
              index=False)