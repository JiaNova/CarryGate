"""Recency gates used by the Phase 3 residual experiments."""

from .recency_gates import (
    GRUDGate,
    KANGate,
    MLPGate,
    ScalarGate,
    build_recency_gate,
)
from .capacity_sweep import CAPACITY_SPECS, CapacityGate, CapacitySpec
from .frozen_gate import FrozenGate
from .gate_bases import BSplineEdge, FourierEdge, RBFEdge, RationalEdge, build_gate_basis
from .monotone_gate import MonotoneDecreasingGate
from .statistic_conditioned_gate import StatisticConditionedGate

__all__ = [
    "GRUDGate",
    "KANGate",
    "MLPGate",
    "ScalarGate",
    "build_recency_gate",
    "CAPACITY_SPECS",
    "CapacityGate",
    "CapacitySpec",
    "FrozenGate",
    "RBFEdge",
    "BSplineEdge",
    "FourierEdge",
    "RationalEdge",
    "build_gate_basis",
    "MonotoneDecreasingGate",
    "StatisticConditionedGate",
]
