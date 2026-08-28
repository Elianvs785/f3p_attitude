"""Attitude + thrust from (gamma, chi, V, mu) using physical_model flat plate."""

from f3p_attitude.regime import MuMode, classify_mu_mode
from f3p_attitude.solver import (
    SolveSample,
    TrajectoryResult,
    solve_one,
    solve_static_hover,
    solve_trajectory,
)

__all__ = [
    "MuMode",
    "classify_mu_mode",
    "SolveSample",
    "TrajectoryResult",
    "solve_one",
    "solve_static_hover",
    "solve_trajectory",
]
