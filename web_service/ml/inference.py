import logging
from typing import List

import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def get_recommendations(
    model: AlternatingLeastSquares,
    user_idx: int,
    interactions: csr_matrix,
    le_song: LabelEncoder,
    n: int = 10,
) -> List[str]:
    """
    Взять top-N рекомендаций (song_ids, раскодированные)
    """
    recs, _ = model.recommend(user_idx, interactions[user_idx], N=n)
    song_ids = le_song.inverse_transform(np.array(recs))
    logger.info("Recommendations: %s", song_ids.tolist())
    return song_ids.tolist()
