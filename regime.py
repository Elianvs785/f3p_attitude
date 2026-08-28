"""How to interpret mu along a trajectory (bank vs yaw vs hover)."""

from __future__ import annotations

from enum import Enum

import numpy as np

from f3p_attitude.constants import GAMMA_VERTICAL_THRESH, V_HOVER_THRESH


class MuMode(str, Enum):
    """Kinematic chart for wind-frame assembly (mu is always mu: roll about velocity)."""

    CRUISE = "cruise"  # oblique path; mu = bank about v_hat (knife)
    BANK = "cruise"  # alias of CRUISE
    VERTICAL = "vertical"  # |gamma| ~ 90: v_hat vertical; mu still about v_hat
    HOVER = "hover"  # |V| very small; softened kinematics only


def classify_mu_mode(gamma: float, speed: float) -> MuMode:
    """
    Pick kinematic chart. mu always means rotation about the instantaneous velocity
    vector (knife / bank-about-velocity), never a renamed yaw angle.
    """
    if speed < V_HOVER_THRESH:
        return MuMode.HOVER
    if abs(gamma) >= GAMMA_VERTICAL_THRESH:
        return MuMode.VERTICAL
    return MuMode.CRUISE
