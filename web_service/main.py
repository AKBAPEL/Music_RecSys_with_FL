import logging
import multiprocessing
import pickle
import uuid
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException, status
from ml.federated import federated_training
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

logger = logging.getLogger("backend")
logger.setLevel(logging.INFO)
file_handler = logging.handlers.RotatingFileHandler(
    filename="/app/logs/backend.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

app = FastAPI()


model_store = ModelStore()


RAW_PATH = "data/train_truncated.csv"
raw_df = load_raw_data(RAW_PATH)
interactions, le_user, le_song, processed_df = build_interaction_matrix(raw_df)


@app.on_event("startup")
def load_default_model() -> None:
    default_path = Path("ml/prepared_models/model_als.pkl")
    if default_path.exists():
        try:
            # model = pickle.load(open(default_path, "rb"))
            info = ModelInfo(
                model_id="default_als",
                params={"type": "ALS"},
                model_path=str(default_path),
            )
            model_store.add_model("default_als", info, str(default_path))
            model_store.set_active("default_als")
            logger.info("Default model loaded: default_als")
        except Exception as e:
            logger.error("Failed to load default model: %s", e)


@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Hi, User!"}


@app.post("/fit", response_model=FitResponse, responses={500: {"model": ErrorResponse}})
async def fit_model(request: FitRequest) -> FitResponse:
    model_id = str(uuid.uuid4())
    # Запуск обучения в отдельном процессе
    logger.info("Received fit request: %s", request.model_dump())
    p = multiprocessing.Process(target=_train_and_save, args=(request, model_id))
    p.start()
    p.join(timeout=30)
    if p.is_alive():
        p.terminate()
        logger.warning("Training timeout for model %s", model_id)
        return ErrorResponse(detail="Training timeout")
    model_path = (
        f"users_models/{request.model_name}_{request.model_type}_{model_id}.pkl"
    )
    if not Path(model_path).exists():
        logger.error("Training failed: model file not found %s", model_path)
        return ErrorResponse(detail="Training failed")
    info = ModelInfo(model_id=model_id, params=request.dict(), model_path=model_path)
    model_store.add_model(model_id, info, model_path)
    logger.info("Model trained and added: %s", model_id)
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
    try:
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
        logger.info("_train_and_save: model saved %s", model_path)
    except Exception as e:
        logger.error("_train_and_save: training error: %s", e)


@app.get("/models", response_model=List[ModelInfo])
async def get_models() -> List[ModelInfo]:
    logger.info("Models list requested")
    return model_store.list_models()


@app.post("/set", response_model=SetResponse)
async def set_model(model_id: str) -> SetResponse:
    try:
        model_store.set_active(model_id)
    except KeyError:
        logger.warning("Set model failed, not found: %s", model_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Model not found"
        )
    return SetResponse(status="success", active_model=model_id)


@app.post("/predict", response_model=PredictResponse)
async def predict(user_id: int, n: int = 10) -> PredictResponse:
    logger.info("Predict requested: user_id=%d, n=%d", user_id, n)
    active_model = model_store.get_active_model()
    user_type = "existing"
    if user_id >= interactions.shape[0]:
        user_type = "new"
        user_idx = interactions.shape[0] - 1
        logger.info(
            "Handling new user: original user_id=%d, fallback user_idx=%d",
            user_id,
            user_idx,
        )
    else:
        user_idx = user_id
    recs = get_recommendations(
        model=active_model,
        user_idx=user_idx,
        interactions=interactions,
        le_song=le_song,
        n=n,
    )
    logger.info("Recommendations for user %d: %s", user_id, recs)
    return PredictResponse(
        recommendations=recs,
        user_type=user_type,
        active_model_id=model_store.active_model_id,
    )


@app.post(
    "/federate",
    response_model=ModelInfo,
    responses={500: {"model": ErrorResponse}},
)
async def federate() -> ModelInfo:
    """
    Ручка для запуска федеративного обучения на текущем датасете.
    """
    logger.info("Federation process started")
    try:
        # Запускаем federated learning на полном наборе processed_df
        new_model = federated_training(
            df=processed_df,
            num_clients=5,
            local_epochs=3,
            agg_rounds=10,
            factors=50,
            regularization=0.01,
        )

        model_id = f"fed_{uuid.uuid4()}"
        model_path = f"users_models/{model_id}.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(new_model, f)

        info = ModelInfo(
            model_id=model_id,
            params={"type": "federated_als"},
            model_path=model_path,
        )
        model_store.add_model(model_id, info, model_path)
        model_store.set_active(model_id)

        logger.info("Federated model trained and set active: %s", model_id)
        return info

    except Exception as e:
        logger.error("Federation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
