import logging
import multiprocessing
import pickle
from typing import Any, Dict, List, Optional

from patterns import ModelInfo

logger = logging.getLogger(__name__)


class ModelStore:
    def __init__(self) -> None:
        self._lock = multiprocessing.Lock()
        self._models: Dict[str, Any] = {}
        self._info: Dict[str, ModelInfo] = {}
        self.active_model_id: Optional[str] = None

    def list_models(self) -> List[ModelInfo]:
        return list(self._info.values())

    def add_model(self, model_id: str, info: ModelInfo, model_path: str) -> None:
        with self._lock:
            try:
                self._info[model_id] = info
                self._models[model_id] = pickle.load(open(model_path, "rb"))
                logger.info("model %s added to store", model_id)
            except Exception as e:
                logger.error("failed to load model %s: %s", model_id, e)

    def set_active(self, model_id: str) -> None:
        if model_id not in self._info:
            logger.warning("non-existent model %s", model_id)
            raise KeyError(f"Model {model_id} not found")
        self.active_model_id = model_id

    def get_active_model(self) -> Any:
        if self.active_model_id is None:
            logger.error("No active model set when attempting to get it")
            raise ValueError("No active model set")
        return self._models[self.active_model_id]
