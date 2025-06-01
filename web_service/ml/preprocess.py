from typing import Tuple

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["msno"] = df["msno"].astype(str)
    df["song_id"] = df["song_id"].astype(str)
    df["source_type"] = df["source_type"].fillna("isnan")
    df["source_system_tab"] = df["source_system_tab"].fillna("isnan")
    return df


def build_interaction_matrix(
    df: pd.DataFrame,
) -> Tuple[csr_matrix, LabelEncoder, LabelEncoder, pd.DataFrame]:
    le_user = LabelEncoder()
    le_song = LabelEncoder()

    df["user_idx"] = le_user.fit_transform(df["msno"])
    df["song_idx"] = le_song.fit_transform(df["song_id"])

    interactions = csr_matrix(
        (df["target"], (df["user_idx"], df["song_idx"])),
        shape=(len(le_user.classes_), len(le_song.classes_)),
    )
    return interactions, le_user, le_song, df
