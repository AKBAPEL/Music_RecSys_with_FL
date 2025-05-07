# import logging
# from typing import List, Tuple

# import numpy as np
# from implicit.als import AlternatingLeastSquares
# from ml.preprocess import build_interaction_matrix
# from scipy.sparse import csr_matrix
# from tqdm import tqdm

# logger = logging.getLogger(__name__)


# def split_data(df, private_frac: float = 0.3) -> Tuple[csr_matrix, csr_matrix]:
#     """
#     Разделяет матрицу взаимодействий на приватную и глобальную части.
#     """
#     interactions, le_user, le_song, _ = build_interaction_matrix(df)
#     assert interactions.shape[0] == len(le_user.classes_)
#     data = interactions.data
#     rows, cols = interactions.nonzero()
#     mask = np.random.rand(len(data)) < private_frac
#     priv_idx, glob_idx = np.where(mask)[0], np.where(~mask)[0]
#     private_matrix = csr_matrix(
#         (data[priv_idx], (rows[priv_idx], cols[priv_idx])), shape=interactions.shape
#     )
#     global_matrix = csr_matrix(
#         (data[glob_idx], (rows[glob_idx], cols[glob_idx])), shape=interactions.shape
#     )
#     return private_matrix, global_matrix


# # def federated_training(
# #     df,
# #     num_clients: int = 5,
# #     local_epochs: int = 3,
# #     agg_rounds: int = 10,
# #     factors: int = 50,
# #     regularization: float = 0.01,
# # ) -> AlternatingLeastSquares:
# #     """
# #     Выполняет federated learning: инициализирует глобальную модель,
# #     запускает локальные обновления на клиентах и агрегирует их.
# #     """
# #     priv_mat, glob_mat = split_data(df)
# #     global_model = AlternatingLeastSquares(
# #         factors=factors,
# #         regularization=regularization,
# #         use_gpu=False
# #     )
# #     global_model.fit(glob_mat)

# #     all_users = np.unique(priv_mat.nonzero()[0])
# #     valid_users = all_users[all_users < priv_mat.shape[0]]
# #     user_ids = valid_users[:num_clients]

# #     for rnd in range(agg_rounds):
# #         collected_u, collected_i = [], []
# #         for u in tqdm(user_ids, desc=f"Round {rnd+1}/{agg_rounds}"):
# #             client_model = AlternatingLeastSquares(
# #                 factors=factors,
# #                 regularization=regularization,
# #                 use_gpu=False,
# #                 iterations=local_epochs,
# #             )
# #             client_model.user_factors = global_model.user_factors.copy()
# #             client_model.item_factors = global_model.item_factors.copy()

# #             user_data = csr_matrix(priv_mat[u, :])
# #             client_model.partial_fit_users([u], user_data)

# #             collected_u.append(client_model.user_factors)
# #             collected_i.append(client_model.item_factors)

# #         global_model.user_factors = np.mean(collected_u, axis=0)
# #         global_model.item_factors = np.mean(collected_i, axis=0)

# #     return global_model


# def federated_training(
#     df,
#     num_clients: int = 5,
#     local_epochs: int = 3,
#     agg_rounds: int = 10,
#     factors: int = 50,
#     regularization: float = 0.01,
# ) -> AlternatingLeastSquares:
#     priv_mat, glob_mat = split_data(df)
#     global_model = AlternatingLeastSquares(
#         factors=factors, regularization=regularization, use_gpu=False
#     )
#     global_model.fit(glob_mat)

#     logger.info("Interactions shape: %s", glob_mat.shape)
#     logger.info("Private nonzero rows: %s", np.unique(priv_mat.nonzero()[0])[-10:])
#     all_users = np.unique(priv_mat.nonzero()[0])
#     valid_users = all_users[all_users < glob_mat.shape[0]]
#     user_ids = valid_users[:num_clients]
#     for rnd in range(agg_rounds):
#         collected_u, collected_i = [], []
#         for u in tqdm(user_ids, desc=f"Round {rnd+1}/{agg_rounds}"):
#             client_model = AlternatingLeastSquares(
#                 factors=factors,
#                 regularization=regularization,
#                 use_gpu=False,
#                 iterations=local_epochs,
#             )
#             client_model.user_factors = global_model.user_factors.copy()
#             client_model.item_factors = global_model.item_factors.copy()

#             user_data = csr_matrix(priv_mat.getrow(u))  # теперь безопасно
#             client_model.partial_fit_users([u], user_data)

#             collected_u.append(client_model.user_factors)
#             collected_i.append(client_model.item_factors)

#         global_model.user_factors = np.mean(collected_u, axis=0)
#         global_model.item_factors = np.mean(collected_i, axis=0)

#     return global_model
