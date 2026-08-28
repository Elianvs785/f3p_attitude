"""F3P parameters from physical_model.md."""

import numpy as np

# Mass and inertia
MASS = 0.100  # kg
WINGSPAN = 0.65  # m
CHORD = 0.20  # m
L_TOTAL = 0.50  # m
IXX = (1.0 / 12.0) * MASS * WINGSPAN**2
IYY = (1.0 / 12.0) * MASS * L_TOTAL**2
IZZ = IXX + IYY
J_INERTIA = np.diag([IXX, IYY, IZZ])

# Propulsion and environment
THRUST_MAX = 0.200 * 9.81  # N (~1.96 N)
THRUST_MIN = 0.0
RHO = 1.225  # kg/m^3
G = 9.81  # m/s^2, NED z-down
GRAVITY_WORLD = np.array([0.0, 0.0, G])

# Geometry
L_PROP = 0.10  # m
L_TAIL = 0.40  # m
TAIL_ARM = L_TAIL - L_PROP  # 0.30 m behind CG
AIL_ARM = 0.13  # m
FUSELAGE_ARM = 0.05  # m, strake x-offset behind CG

PROP_DIAM = 0.15  # m
PROP_DISK_AREA = np.pi * (PROP_DIAM / 2.0) ** 2

DELTA_MAX = np.deg2rad(30.0)
# Solver attitude bounds (must allow |alpha|>75 deg for vertical static hover at gamma=0)
ALPHA_BETA_MAX = np.deg2rad(89.9)

# Solver tolerances
RESIDUAL_TOL = 1e-4
FSOLVE_XTOL = 1e-9

# mu interpretation (see regime.py)
GAMMA_VERTICAL_THRESH = np.deg2rad(85.0)  # |gamma| >= this -> mu is yaw
V_HOVER_THRESH = 0.5  # m/s; mu_mode label only (no solver branch)
