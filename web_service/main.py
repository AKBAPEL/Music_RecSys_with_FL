from fastapi import FastAPI, HTTPException, status
import pickle
import pandas as pd
from contextlib import asynccontextmanager
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import numpy as np
import implicit
from pydantic import BaseModel
import multiprocessing
from datetime import datetime
import time
from pathlib import Path
import uuid
from typing import List, Dict, Optional
from patterns import FitRequest, ModelInfo, FitResponse, SetResponse, ErrorResponse, PredictResponse

TRAIN = f"../train.csv"
train_df = pd.read_csv(TRAIN)


ml_models = {}
class ModelStore:
    def __init__(self):
        self.models: Dict[str, object] = {}
        self.models_info: Dict[str, ModelInfo] = {}
        self.lock = multiprocessing.Lock()
        self.active_model_id: Optional[str] = None
        # self.label_encoders: Dict[str, LabelEncoder] = {}

model_store = ModelStore()

def get_als_data(train_df):
    le_msno = LabelEncoder()
    le_song_id = LabelEncoder()

    le_msno.fit(train_df["msno"])
    le_song_id.fit(train_df["song_id"])

    train = pd.DataFrame(
        {"msno": le_msno.transform(train_df["msno"].values), "song_id": le_song_id.transform(train_df["song_id"].values),
        "target": train_df["target"]})

    user_ids = train["msno"].values
    song_ids = train["song_id"].values
    target = train["target"].values

    matrix = csr_matrix((target, (user_ids, song_ids)), shape=(len(train['msno'].unique()), len(train['song_id'].unique())))

    ml_models['le_song_id'] = le_song_id
    return matrix, le_msno, le_song_id, train

matrix, le_msno, le_song_id, train = get_als_data(train_df=train_df)


def music_recommendation(user_id, n=10):
    with open("prepared_models/model_als.pkl", 'rb') as model_file:  # experiment-als модель
        loaded_model = pickle.load(model_file)

    recommendations = loaded_model.recommend(user_id, matrix[user_id], N=n)
    res = le_song_id.inverse_transform(recommendations[0].tolist())

    model_store.models["default_als"] = loaded_model
    model_store.models_info["default_als"] = ModelInfo(
                    model_id="default",
                    params={"type": "ALS"},
                    model_path="prepared_models/model_als.pkl"
                )
    
    return list(res)
    
@asynccontextmanager
async def ml_lifespan_manager(app: FastAPI):
    ml_models["music_recommendation"] = music_recommendation
    yield
    ml_models.clear()

app = FastAPI(lifespan=ml_lifespan_manager)

@app.get("/")
async def root():
    return {"message" : "Hi, User!"}


@app.post("/predict", response_model=PredictResponse)
async def predict(user_id: int, n: int = 10):
    global matrix
    user_type = 'existing'

    if user_id >= matrix.shape[0]:  # новый пользователь
        user_type = "new"
        user_id = matrix.shape[0] - 1  # временное решение по определению нового пользователя

        top_10_songs = train["song_id"].value_counts().head(10).index
        new_user_data = np.array(np.ones(len(top_10_songs)))
        new_user_indices = np.array(top_10_songs)
        new_data = np.concatenate([matrix.data, new_user_data])
        new_indices = np.concatenate([matrix.indices, new_user_indices])
        new_indptr = np.concatenate([matrix.indptr, [matrix.indptr[-1] + len(new_user_data)]])

        matrix = csr_matrix((new_data, new_indices, new_indptr), shape=(matrix.shape[0] + 1, matrix.shape[1]))

    return PredictResponse(recommendations=ml_models["music_recommendation"](user_id, n), user_type=user_type)


def _train_model(params: FitRequest, model_id: str):
    try:
        # start_time = time.time()

        model = implicit.als.AlternatingLeastSquares(
            factors=params.factors,
            iterations=params.iterations,
            regularization=params.regularization,
            alpha=params.alpha,
            random_state=42
        )
        model.fit(matrix)
        with open(f"users_models/{params.model_name}_{params.model_type}_{model_id}.pkl", 'wb') as f:
            pickle.dump(model, f)

        # logger.info(f"Model {model_id} trained successfully")
        return True
    except Exception as e:
        # logger.error(f"Training failed: {e}")
        return False

@app.post("/fit", response_model=FitResponse, responses={500: {"model": ErrorResponse}})
async def fit_model(request: FitRequest):
    # model_id = f"model_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    model_id = str(uuid.uuid4())

    p = multiprocessing.Process(
        target=_train_model,
        args=(request, model_id)
    )
    p.start()

    start_time = time.time()
    timeout = 30
    while time.time() - start_time < timeout:
        if not p.is_alive():
            break
        time.sleep(0.1)
 
    if p.is_alive():
        p.terminate()
        # logger.warning(f"Training timeout for model {model_id}")
        return {
            "status": "error",
            "message": "Training timeout"
        }


    if not Path(f"users_models/{request.model_name}_{request.model_type}_{model_id}.pkl").exists():
        # return {
        #     "status": "error",
        #     "message": "Training failed"
        # }
        return ErrorResponse(
            detail="error, Training failed"
        )
    
    with model_store.lock:
        model_store.models_info[model_id] = ModelInfo(
            model_id=model_id,
            params=request.dict(),
            model_path=f"users_models/{request.model_name}_{request.model_type}_{model_id}.pkl"
        )

    return FitResponse(
        status="success",
        model_id=model_id,
        message="Model trained and saved",
        details={
            "factors": request.factors,
            "iterations": request.iterations,
            "regularization": request.regularization
        }
    )



@app.get("/models", response_model=List[ModelInfo])
async def get_models():
    """Список текущих моделей"""
    return [
            info
            for model_id, info in model_store.models_info.items()
        ]

@app.post("/set", response_model=SetResponse)
async def set_model(model_id: str):
    with model_store.lock:
        if model_id not in model_store.models_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )

        if model_id not in model_store.models:
            try:
                with open(model_store.models_info[model_id].model_path, 'rb') as f:
                    model_store.models[model_id] = pickle.load(f)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Model loading failed"
                )

        model_store.active_model_id = model_id

    return SetResponse(
        status="success",
        active_model=model_id
    )