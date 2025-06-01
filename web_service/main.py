import multiprocessing
import pickle
import uuid
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException, status
from ml.inference import get_recommendations
from ml.preprocess import build_interaction_matrix, load_raw_data
from ml.train import train_als_model
from patterns import (
    ErrorResponse,
    FitRequest,
    FitResponse,
    ModelInfo,
    PredictResponse,
    SetResponse,
)
from store.model_store import ModelStore

app = FastAPI()


model_store = ModelStore()


RAW_PATH = "data/train_truncated.csv"
raw_df = load_raw_data(RAW_PATH)
interactions, le_user, le_song, processed_df = build_interaction_matrix(raw_df)


@app.on_event("startup")
def load_default_model() -> None:
    default_path = Path("ml/prepared_models/model_als.pkl")
    if default_path.exists():
        # model = pickle.load(open(default_path, "rb"))
        info = ModelInfo(
            model_id="default_als", params={"type": "ALS"}, model_path=str(default_path)
        )
        model_store.add_model("default_als", info, str(default_path))
        model_store.set_active("default_als")


@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Hi, User!"}


@app.post("/fit", response_model=FitResponse, responses={500: {"model": ErrorResponse}})
async def fit_model(request: FitRequest) -> FitResponse:
    model_id = str(uuid.uuid4())
    # Запуск обучения в отдельном процессе
    p = multiprocessing.Process(target=_train_and_save, args=(request, model_id))
    p.start()
    p.join(timeout=30)
    if p.is_alive():
        p.terminate()
        return ErrorResponse(detail="Training timeout")
    model_path = (
        f"users_models/{request.model_name}_{request.model_type}_{model_id}.pkl"
    )
    if not Path(model_path).exists():
        return ErrorResponse(detail="Training failed")
    info = ModelInfo(model_id=model_id, params=request.dict(), model_path=model_path)
    model_store.add_model(model_id, info, model_path)
    return FitResponse(
        status="success",
        model_id=model_id,
        message="Model trained and saved",
        details={
            "factors": request.factors,
            "iterations": request.iterations,
            "regularization": request.regularization,
        },
    )


def _train_and_save(request: FitRequest, model_id: str) -> None:
    model = train_als_model(
        interactions=interactions,
        factors=request.factors,
        iterations=request.iterations,
        regularization=request.regularization,
        alpha=request.alpha,
    )
    model_path = (
        f"users_models/{request.model_name}_{request.model_type}_{model_id}.pkl"
    )
    pickle.dump(model, open(model_path, "wb"))


@app.get("/models", response_model=List[ModelInfo])
async def get_models() -> List[ModelInfo]:
    return model_store.list_models()


@app.post("/set", response_model=SetResponse)
async def set_model(model_id: str) -> SetResponse:
    try:
        model_store.set_active(model_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Model not found"
        )
    return SetResponse(status="success", active_model=model_id)


@app.post("/predict", response_model=PredictResponse)
async def predict(user_id: int, n: int = 10) -> PredictResponse:
    active_model = model_store.get_active_model()
    user_type = "existing"
    if user_id >= interactions.shape[0]:
        user_type = "new"
        user_idx = interactions.shape[0] - 1  # fallback
    else:
        user_idx = user_id
    recs = get_recommendations(
        model=active_model,
        user_idx=user_idx,
        interactions=interactions,
        le_song=le_song,
        n=n,
    )
    return PredictResponse(
        recommendations=recs,
        user_type=user_type,
        active_model_id=model_store.active_model_id,
    )
