"""World-frame velocity and acceleration from (gamma, chi, V) and rates."""

import numpy as np

from f3p_attitude.frames import (
    dv_hat_dchi,
    dv_hat_dgamma,
    velocity_unit_ned,
)


def velocity_world(
    gamma: float, chi: float, speed: float
) -> tuple[np.ndarray, np.ndarray]:
    v_hat = velocity_unit_ned(gamma, chi)
    return speed * v_hat, v_hat


def acceleration_world(
    gamma: float,
    chi: float,
    speed: float,
    gamma_dot: float,
    chi_dot: float,
    speed_dot: float,
) -> np.ndarray:
    v_hat = velocity_unit_ned(gamma, chi)
    v_hat_dot = gamma_dot * dv_hat_dgamma(gamma, chi) + chi_dot * dv_hat_dchi(
        gamma, chi
    )
    return speed_dot * v_hat + speed * v_hat_dot
