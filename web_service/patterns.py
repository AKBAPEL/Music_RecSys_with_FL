from pydantic import BaseModel

class FitRequest(BaseModel):
    model_name: str
    model_type: str
    factors: int = 100
    iterations: int = 20
    regularization: float = 0.01
    alpha: float = 1.0

class ModelInfo(BaseModel):
    model_id: str
    params: dict
    model_path: str
    # is_active: bool = False

