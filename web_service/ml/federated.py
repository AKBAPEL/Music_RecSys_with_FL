import logging
from typing import Tuple

import numpy as np
from implicit.als import AlternatingLeastSquares
from ml.preprocess import build_interaction_matrix
from scipy.sparse import csr_matrix
from tqdm import tqdm

logger = logging.getLogger(__name__)


def split_data(df, private_frac: float = 0.3) -> Tuple[csr_matrix, csr_matrix]:
    interactions, le_user, le_song, _ = build_interaction_matrix(df)

    coo = interactions.tocoo()
    data = coo.data
    rows = coo.row
    cols = coo.col
    n_elements = len(data)

    mask = np.random.rand(n_elements) < private_frac

    private_matrix = csr_matrix(
        (data[mask], (rows[mask], cols[mask])), shape=interactions.shape
    )

    global_matrix = csr_matrix(
        (data[~mask], (rows[~mask], cols[~mask])), shape=interactions.shape
    )

    return private_matrix, global_matrix


def federated_training(
    df,
    num_clients: int = 5,
    local_epochs: int = 3,
    agg_rounds: int = 10,
    factors: int = 50,
    regularization: float = 0.01,
) -> AlternatingLeastSquares:
    priv_mat, glob_mat = split_data(df)
    global_model = AlternatingLeastSquares(
        factors=factors, regularization=regularization, use_gpu=False
    )
    global_model.fit(glob_mat)

    user_factors_shape = global_model.user_factors.shape[0]

    all_users = np.unique(priv_mat.nonzero()[0])
    valid_users = all_users[all_users < user_factors_shape]
    user_ids = valid_users[:num_clients]

    logger.info("Global model users: %d", user_factors_shape)
    logger.info("Valid private users: %d", len(valid_users))

    for rnd in range(agg_rounds):
        collected_u, collected_i = [], []
        for u in tqdm(user_ids, desc=f"Round {rnd+1}/{agg_rounds}"):
            if u >= user_factors_shape:
                logger.warning("Skipping user %d (out of bounds)", u)
                continue

            client_model = AlternatingLeastSquares(
                factors=factors,
                regularization=regularization,
                use_gpu=False,
                iterations=local_epochs,
            )
            client_model.user_factors = global_model.user_factors.copy()
            client_model.item_factors = global_model.item_factors.copy()

            user_data = priv_mat.getrow(u)
            client_model.partial_fit_users([u], user_data)

            collected_u.append(client_model.user_factors)
            collected_i.append(client_model.item_factors)

        if collected_u:
            global_model.user_factors = np.mean(collected_u, axis=0)
            global_model.item_factors = np.mean(collected_i, axis=0)

    return global_model
