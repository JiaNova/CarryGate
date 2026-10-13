"""Prediction heads for Phase 1 and Phase 3."""

from .forecast_head import ForecastHead, FixedSinusoidalTimeEmbedding
from .conditioned_residual_head import ConditionedResidualHead
from .persistence_residual_head import PersistenceResidualHead

__all__ = [
    "ForecastHead",
    "FixedSinusoidalTimeEmbedding",
    "PersistenceResidualHead",
    "ConditionedResidualHead",
]
