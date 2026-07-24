import pandas as pd


def preprocessing(train_data, handle_categorical='None'):

    # ↓↓↓ Заполняем null-значения ↓↓↓

    train_data.loc[train_data['Embarked'].isnull(), 'Embarked'] = 'S'

    train_data['Honorifics'] = train_data['Name'].str.extract('([A-Z][a-z]+)\.', expand=True)
    train_data['Honorifics'] = train_data['Honorifics'].replace(['Mlle', 'Ms'], 'Miss')
    train_data['Honorifics'] = train_data['Honorifics'].replace('Mme', 'Mrs')
    train_data['Honorifics'] = train_data.loc[~train_data['Honorifics'].isin(['Mr', 'Mrs', 'Miss', 'Master']), 'Honorifics'] = 'Rare'

    mean_ages = train_data.groupby('Honorifics')['Age'].mean()
    for honorific in mean_ages.index:
        train_data.loc[(train_data['Age'].isnull()) & (train_data['Honorifics'] == honorific), 'Age'] = mean_ages[honorific]

    # ↓↓↓ Преобразуем непрерывные величины в дискретные ↓↓↓

    train_data['Age_Group'] = pd.cut(x=train_data['Age'],
                                     bins=[0, 10, 20, 30, 40, 50, 60, 70, 80],
                                     labels=[0, 1, 2, 3, 4, 5, 6, 7],
                                     include_lowest=True)

    train_data['Fare_Range'] = pd.qcut(x=train_data['Fare'],
                                       q=5,
                                       labels=[0, 1, 2, 3, 4])

    # ↓↓↓ Создадим новые фичи ↓↓↓

    train_data['Family_Size'] = train_data['SibSp'] + train_data['Parch']

    train_data['Alone'] = False
    train_data.loc[train_data['Family_Size'] == 0, 'Alone'] = True

    # ↓↓↓ Удаляем ненужные фичи ↓↓↓

    train_data = train_data.drop(columns=['PassengerId', 'Name', 'Age', 'Ticket', 'Fare', 'Cabin'])

    # ↓↓↓ Обработка категориальных признаков в соответствие с запросом ↓↓↓

    if handle_categorical.lower() == 'none':
        train_data['Alone'] = train_data['Alone'].astype('category')
        train_data['Sex'] = train_data['Sex'].astype('category')
        train_data['Embarked'] = train_data['Embarked'].astype('category')
        train_data['Honorifics'] = train_data['Honorifics'].astype('category')
    elif handle_categorical.lower() == 'one-hot-encoding':
        train_data = pd.get_dummies(data=train_data,
                                    columns=['Sex', 'Embarked', 'Honorifics', 'Age_Group', 'Fare_Range', 'Alone'],
                                    drop_first=True,
                                    dtype=int)
    elif handle_categorical.lower() == 'ordinal-encoding':
        train_data['Sex'] = pd.factorize(train_data['Sex'])[0]
        train_data['Embarked'] = pd.factorize(train_data['Embarked'])[0]
        train_data['Honorifics'] = pd.factorize(train_data['Honorifics'])[0]
        train_data['Age_Group'] = train_data['Age_Group'].astype(int)
        train_data['Fare_Range'] = train_data['Fare_Range'].astype(int)
        train_data['Alone'] = train_data['Alone'].astype(int)
    else:
        raise NameError('"Handle_categorical" must be one of: none, one-hot-encoding, ordinal-encoding')

    return train_data