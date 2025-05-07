import pickle
import pandas as pd
from fastapi import FastAPI
from contextlib import asynccontextmanager
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import numpy as np

data_path = "."
TRAIN = f"{data_path}/train.csv"
train_df = pd.read_csv(TRAIN)

le_msno = LabelEncoder()
le_song_id = LabelEncoder()

le_msno.fit(train_df["msno"])
le_song_id.fit(train_df["song_id"])

train = pd.DataFrame({"msno": le_msno.transform(train_df["msno"].values), "song_id": le_song_id.transform(train_df["song_id"].values), "target": train_df["target"]})

user_ids = train["msno"].values
song_ids = train["song_id"].values
target = train["target"].values

matrix = csr_matrix((target, (user_ids, song_ids)), shape=(len(train['msno'].unique()), len(train['song_id'].unique())))

def music_recommendation(user_id, n=10):
    with open("model.pkl", 'rb') as model_file: # experiment-als модель
        loaded_model = pickle.load(model_file)
    recommendations = loaded_model.recommend(user_id, matrix[user_id], N=n)
    res = recommendations[0]
    return res.tolist()

ml_models = {}

@asynccontextmanager
async def ml_lifespan_manager(app: FastAPI):
    ml_models["music_recommendation"] = music_recommendation
    yield
    ml_models.clear()

app = FastAPI(lifespan=ml_lifespan_manager)

@app.post("/predict")
async def predict(user_id: int, n: int = 10):
    global matrix
    
    if user_id >= matrix.shape[0]: # новый пользователь
        user_id = matrix.shape[0] - 1 # временное решение по определению нового пользователя
        
        top_10_songs = train["song_id"].value_counts().head(10).index
        new_user_data = np.array(np.ones(len(top_10_songs)))
        new_user_indices = np.array(top_10_songs)
        new_data = np.concatenate([matrix.data, new_user_data])
        new_indices = np.concatenate([matrix.indices, new_user_indices])
        new_indptr = np.concatenate([matrix.indptr, [matrix.indptr[-1] + len(new_user_data)]])
        
        matrix = csr_matrix((new_data, new_indices, new_indptr), shape=(matrix.shape[0] + 1, matrix.shape[1]))
    
    return ml_models["music_recommendation"](user_id, n)
        