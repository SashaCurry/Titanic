import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from config import config
from data_handle import *


class TitanicNN(nn.Module):
    def __init__(self, num_features_count, cat_dims):
        """
        num_features_count: число обычных числовых фичей (например, 4)
        cat_dims: список кортежей [(число_классов, размер_эмбеддинга), ...] для каждой из категориальных фичей
        Пример cat_dims: [(2, 2), (4, 2), (10, 4), (3, 2), (8, 4), (5, 3)]
        """
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=num_classes, embedding_dim=emb_dim) for num_classes, emb_dim in cat_dims
        ])

        # Вычисляем вход для первого слоя
        input = num_features_count + sum(emb_dim for _, emb_dim in cat_dims)

        self.layer_1 = nn.Linear(input, 100, bias=False)
        self.batchnorm_1 = nn.BatchNorm1d(100)
        self.layer_2 = nn.Linear(100, 200)
        self.layer_3 = nn.Linear(200, 300)
        self.layer_4 = nn.Linear(300, 1)

        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
        })

        self.dropout = nn.Dropout(0.5)

    def forward(self, x_num, x_cat):
        # x_num содержит числовые фичи
        # x_cat содержит категориальные фичи
        embedding_outputs = []

        # Пробегаемся по всем категориальным столбцам и преобразуем в векторы
        for i, emb_layer in enumerate(self.embeddings):
            col_data = x_cat[:, i]
            emb_vec = emb_layer(col_data)
            embedding_outputs.append(emb_vec)

        # Склеиваем все столбцы
        x = torch.cat([x_num] + embedding_outputs, dim=1)

        # Основная работы нейросети
        x = self.layer_1(x)
        x = self.activations['relu'](x)
        x = self.batchnorm_1(x)
        x = self.layer_2(x)
        x = self.dropout(x)
        x = self.activations['relu'](x)
        x = self.layer_3(x)
        x = self.activations['relu'](x)
        x = self.layer_4(x)
        out = self.activations['sigmoid'](x)
        return out


def train_nn(X, y):
    X = preprocessing(X, handle_categorical='Ordinal-encoding')

    num_cols = ['Family_Size', 'Parch', 'Pclass', 'SibSp']
    cat_cols = ['Age_Group', 'Alone', 'Embarked', 'Fare_Range', 'Honorifics', 'Sex']

    X_num_tensor = torch.tensor(data=X[num_cols].values,
                                dtype=torch.float32,
                                device=config.training.device)
    X_cat_tensor = torch.tensor(data=X[cat_cols].values,
                                dtype=torch.long,
                                device=config.training.device)
    y_tensor = torch.tensor(data=y.values,
                            dtype=torch.float32,
                            device=config.training.device)

    dataset = TensorDataset(X_num_tensor, X_cat_tensor, y_tensor)
    train_data, val_data = random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=32, shuffle=True)

    # Кортеж для Embedding-слоёв
    cat_dims = [(X[col].nunique(), min(50, X[col].nunique() // 2 + 1)) for col in cat_cols]

    model = TitanicNN(num_features_count=len(num_cols), cat_dims=cat_dims).to(config.training.device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    num_epochs = 20

    # ЦИКЛ ОБУЧЕНИЯ
    for epoch in range(num_epochs):
        # ТРЕНИРОВКА
        running_train_loss = []
        true_answer = 0

        model.train()
        for X_num_batch, X_cat_batch, y_batch in train_loader:
            # Прямой проход + расчёт ошибки модели
            pred = model(X_num_batch, X_cat_batch)
            loss = loss_fn(pred, y_batch.view(-1, 1))

            # Обратный проход
            optimizer.zero_grad()
            loss.backward()

            # Шаг оптимизатора
            optimizer.step()

            running_train_loss.append(loss.item())
            true_answer += ((pred > 0.5).float() == y_batch.view(-1, 1)).sum().item()

        # Accuracy и Loss на тренировочных данных
        running_train_acc = true_answer / len(train_data)
        mean_train_loss = sum(running_train_loss) / len(running_train_loss)

        # ВАЛИДАЦИЯ
        running_val_loss = []
        true_answer = 0

        model.eval()
        with torch.no_grad():
            for X_num_batch, X_cat_batch, y_batch in val_loader:
                # Прямой проход + расчёт ошибки модели
                pred = model(X_num_batch, X_cat_batch)
                loss = loss_fn(pred, y_batch.view(-1, 1))

                running_val_loss.append(loss.item())
                true_answer += ((pred > 0.5).float() == y_batch.view(-1, 1)).sum().item()

        # Accuracy и Loss на валидационных данных
        running_val_acc = true_answer / len(val_data)
        mean_val_loss = sum(running_val_loss) / len(running_val_loss)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]: '
                  f'train_loss: {mean_train_loss:.4f}, train_accuracy: {running_train_acc:.4f}, '
                  f'val_loss: {mean_val_loss:.4f}, val_accuracy: {running_val_acc:.4f}')
