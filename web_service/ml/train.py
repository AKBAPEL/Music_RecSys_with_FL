import pickle
import uuid

from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix


def train_als_model(
    interactions: csr_matrix,
    factors: int,
    iterations: int,
    regularization: float,
    alpha: float,
) -> AlternatingLeastSquares:
    model = AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        alpha=alpha,
        random_state=42,
    )
    model.fit(interactions)
    return model


def save_model(model: AlternatingLeastSquares, directory: str, model_name: str) -> str:
    model_id = f"{model_name}_{uuid.uuid4()}"
    path = f"{directory}/{model_id}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path
