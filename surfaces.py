"""Seven-surface layout from physical_model.md (passive plates + tails at delta=0)."""

from dataclasses import dataclass

import numpy as np

from f3p_attitude.constants import AIL_ARM, FUSELAGE_ARM, TAIL_ARM


@dataclass(frozen=True)
class Surface:
    name: str
    r_body: np.ndarray
    area: float
    normal: np.ndarray
    hinge_axis: np.ndarray
    in_propwash: bool



def f3p_surfaces() -> list[Surface]:
    """Passive aero surfaces at delta=0 (ailerons omitted).

    Main wing and top strake share the same normal; for force-only trim they
    are merged into one 0.20 m^2 plate at the CG (physical_model: 0.10 each).
    Side strake stays separate for knife-edge Y-force.
    """
    n_down = np.array([0.0, 0.0, -1.0])
    n_side = np.array([0.0, -1.0, 0.0])
    hinge_y = np.array([0.0, 1.0, 0.0])
    hinge_z = np.array([0.0, 0.0, 1.0])
    r_strake = np.array([-FUSELAGE_ARM, 0.0, 0.0])
    r_tail = np.array([-TAIL_ARM, 0.0, 0.0])

    return [
        Surface("main_wing", np.zeros(3), 0.100, n_down, hinge_y, False),
        Surface("top_strake", r_strake, 0.100, n_down, hinge_y, False),
        Surface("side_strake", r_strake, 0.100, n_side, hinge_z, False),
        Surface("elevator", r_tail, 0.040, n_down, hinge_y, True),
        Surface("rudder", r_tail, 0.040, n_side, hinge_z, True),
    ]


# Reference only (not used when delta_a=0)
AILERON_AREA = 0.015
AILERON_R_POS = np.array([0.0, AIL_ARM, 0.0])
