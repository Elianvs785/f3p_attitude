"""Convert NED (z down) to display coordinates (z up = altitude)."""

import numpy as np

from f3p_attitude.constants import GRAVITY_WORLD


def pos_ned_to_display(p_ned: np.ndarray) -> np.ndarray:
    """(north, east, down) -> (north, east, up)."""
    p = np.asarray(p_ned, dtype=float)
    if p.ndim == 1:
        return np.array([p[0], p[1], -p[2]])
    out = p.copy()
    out[..., 2] = -out[..., 2]
    return out


def vec_ned_to_display(v_ned: np.ndarray) -> np.ndarray:
    """Flip z component for arrows shown in up-positive plots."""
    return pos_ned_to_display(v_ned)


def gravity_display() -> np.ndarray:
    """Gravity arrow in display (points down)."""
    return vec_ned_to_display(GRAVITY_WORLD)
