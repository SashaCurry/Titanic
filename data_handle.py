import pandas as pd

def preprocessing(train_data):

    # Заполняем null-значения

    train_data.loc[train_data['Embarked'].isnull(), 'Embarked'] = 'S'

    train_data['Honorifics'] = train_data['Name'].str.extract('([A-Z][a-z]+)\.', expand=True)
    train_data['Honorifics'] = train_data['Honorifics'].replace(['Mlle', 'Ms'], 'Miss')
    train_data['Honorifics'] = train_data['Honorifics'].replace('Mme', 'Mrs')
    train_data['Honorifics'] = train_data.loc[~train_data['Honorifics'].isin(['Mr', 'Mrs', 'Miss', 'Master']), 'Honorifics'] = 'Rare'

    mean_ages = train_data.groupby('Honorifics')['Age'].mean()
    for honorific in mean_ages.index:
        train_data.loc[(train_data['Age'].isnull()) & (train_data['Honorifics'] == honorific), 'Age'] = mean_ages[honorific]

    # Преобразуем непрерывные величины в дискретные

    train_data['Age_Group'] = pd.cut(x=train_data['Age'],
                                     bins=[0, 10, 20, 30, 40, 50, 60, 70, 80],
                                     labels=[0, 1, 2, 3, 4, 5, 6, 7],
                                     include_lowest=True)

    train_data['Fare_Range'] = pd.qcut(x=train_data['Fare'],
                                       q=5,
                                       labels=[0, 1, 2, 3, 4])

    # Создадим новые фичи

    train_data['Family_Size'] = train_data['SibSp'] + train_data['Parch']

    train_data['Alone'] = 0
    train_data.loc[train_data['Family_Size'] == 0, 'Alone'] = 1

    # Преобразуем стороковые величины в числовые

    train_data['Sex'] = train_data['Sex'].replace(['male', 'female'], [0, 1])
    train_data['Embarked'] = train_data['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2])
    train_data['Honorifics'] = train_data['Honorifics'].replace(['Mr', 'Mrs', 'Miss', 'Master', 'Rare'], [0, 1, 2, 3, 4])

    # Удаляем ненужные фичи

    train_data = train_data.drop(columns=['PassengerId', 'Name', 'Age', 'Ticket', 'Fare', 'Cabin'])

    return train_data