import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from typing import List, Tuple
from ml.preprocess import build_interaction_matrix
from tqdm import tqdm
import pandas as pd


def split_global_private(df: pd.DataFrame, private_frac: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Делит DataFrame по пользователям: у каждого пользователя оставляем private_frac записей в приватный датасет,
    остальное в глобальный.
    """
    private = df.groupby('msno').sample(frac=private_frac, random_state=42)
    global_df = df.drop(index=private.index).sort_values(by='msno')
    return private, global_df


def build_matrices(
    private_df: pd.DataFrame,
    global_df: pd.DataFrame,
) -> Tuple[csr_matrix, csr_matrix, np.ndarray]:
    # Сначала общее кодирование
    combined = pd.concat([private_df, global_df])
    interactions, le_user, le_song, _ = build_interaction_matrix(combined)

    # Получаем координаты ненулей
    data = interactions.data
    rows, cols = interactions.nonzero()

    # Узнаём индексы пользователей, относящихся к private_df
    private_users = np.unique(private_df['user_idx'])

    # Булева маска для data
    mask = np.isin(rows, private_users)

    # Разбиваем на приватные и глобальные части
    priv_data, priv_rows, priv_cols = data[mask], rows[mask], cols[mask]
    glob_data, glob_rows, glob_cols = data[~mask], rows[~mask], cols[~mask]

    priv_mat = csr_matrix((priv_data, (priv_rows, priv_cols)), shape=interactions.shape)
    glob_mat = csr_matrix((glob_data, (glob_rows, glob_cols)), shape=interactions.shape)

    # Список уникальных пользователей-клиентов
    user_ids = private_users

    return priv_mat, glob_mat, user_ids


def federated_training(
    df: pd.DataFrame,
    num_clients: int = 5,
    local_epochs: int = 3,
    agg_rounds: int = 10,
    factors: int = 50,
    regularization: float = 0.01,
) -> AlternatingLeastSquares:
    """
    Выполняет federated learning по оригинальной логике:
    1) Делит df на private_df и global_df
    2) Строит CSR-матрицы
    3) Инициализирует глобальную модель на global_df
    4) Для каждого клиента (первых num_clients): дообучает с partial_fit_users
    5) Агрегирует усреднением
    """
    # 1. Сплит данных
    private_df, global_df = split_global_private(df, private_frac=0.3)

    # 2. Построение матриц и выбор клиентов
    priv_mat, glob_mat, all_users = build_matrices(private_df, global_df)
    user_ids = all_users[:num_clients]

    # 3. Инициализация глобальной модели
    global_model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        use_gpu=False,
    )
    global_model.fit(glob_mat)

    # 4. Федеративные раунды
    for rnd in range(agg_rounds):
        collected_u, collected_i = [], []
        for u in tqdm(user_ids, desc=f"Round {rnd+1}/{agg_rounds}"):
            client_model = AlternatingLeastSquares(
                factors=factors,
                regularization=regularization,
                use_gpu=False,
                iterations=local_epochs,
            )
            # Копируем глобальные факторы
            client_model.user_factors = global_model.user_factors.copy()
            client_model.item_factors = global_model.item_factors.copy()

            # Локальное обновление
            user_data = priv_mat.getrow(u)
            client_model.partial_fit_users([u], user_data)

            collected_u.append(client_model.user_factors)
            collected_i.append(client_model.item_factors)

        # 5. Агрегация усреднением
        global_model.user_factors = np.mean(collected_u, axis=0)
        global_model.item_factors = np.mean(collected_i, axis=0)
    return global_model