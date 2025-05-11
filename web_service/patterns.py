from pydantic import BaseModel
from typing import List, Optional, Dict
class FitRequest(BaseModel):
    model_name: str
    model_type: str
    factors: int = 100
    iterations: int = 1 # 20
    regularization: float = 0.01
    alpha: float = 1.0

class ModelInfo(BaseModel):
    model_id: str
    params: dict
    model_path: str
    # is_active: bool = False

class FitResponse(BaseModel):
    status: str
    model_id: str
    message: str
    details: dict

class SetResponse(BaseModel):
    status: str
    active_model: str

class ErrorResponse(BaseModel):
    detail: str

class PredictResponse(BaseModel):
    recommendations: List[str]
    user_type: str
    active_model_id: str

class PredictRequest(BaseModel):
    user_id: int
    n: int = 10