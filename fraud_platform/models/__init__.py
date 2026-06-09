"""Model registry of types — map a string name to an AnomalyModel class.

Lets training/serving/registry refer to models by name ("xgboost", etc.) without importing
each class directly. Adding a new model = implement AnomalyModel and add it here.
"""
from fraud_platform.models.autoencoder import AutoencoderModel
from fraud_platform.models.base import AnomalyModel
from fraud_platform.models.iforest import IForestModel
from fraud_platform.models.xgb import XGBModel

MODEL_TYPES: dict[str, type[AnomalyModel]] = {
    IForestModel.name: IForestModel,
    XGBModel.name: XGBModel,
    AutoencoderModel.name: AutoencoderModel,
}

# file extension used per model type when saved to the registry
ARTIFACT_EXT: dict[str, str] = {
    IForestModel.name: "pkl",
    XGBModel.name: "pkl",
    AutoencoderModel.name: "pt",
}


def make_model(model_type: str, **kwargs) -> AnomalyModel:
    if model_type not in MODEL_TYPES:
        raise KeyError(f"unknown model type: {model_type}. known: {list(MODEL_TYPES)}")
    return MODEL_TYPES[model_type](**kwargs)


def load_model(model_type: str, path: str) -> AnomalyModel:
    return MODEL_TYPES[model_type].load(path)


__all__ = ["AnomalyModel", "MODEL_TYPES", "ARTIFACT_EXT", "make_model", "load_model"]
