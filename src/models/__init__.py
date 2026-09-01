"""Detector registry.

Every model is built from a plain configuration dictionary, so a run is fully
described by its YAML file:

    from src.models import get

    model = get("DualTrAD")(**config["model"])
"""

from .dtaad.model import DTAAD
from .dualtrad.model import DualTrAD
from .predtrad.model import PredTrADv1
from .shared.base import BaseDetector, DetectorOutput
from .tranad.model import TranAD

MODELS = {
    "DualTrAD": DualTrAD,
    "PredTrAD_v1": PredTrADv1,
    "TranAD": TranAD,
    "DTAAD": DTAAD,
}


def get(model_name: str):
    """Return the class registered under ``model_name``."""
    if model_name not in MODELS:
        raise ValueError(f"unknown model {model_name!r}; available: {sorted(MODELS)}")
    return MODELS[model_name]


__all__ = ["get", "MODELS", "BaseDetector", "DetectorOutput",
           "DualTrAD", "PredTrADv1", "TranAD", "DTAAD"]
