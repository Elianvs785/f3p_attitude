# f3p_attitude

Attitude and thrust reference from `(gamma, chi, V, mu)` (Interface A).

## Layout

| Part | Files | Run |
|------|--------|-----|
| **Solver** | `solver.py` + `frames`, `kinematics`, `aero`, `surfaces`, `constants`, `regime`, `ned_display` | `from f3p_attitude import solve_one` |
| **Tests** | `verify.py` | `python -m f3p_attitude.verify` |
| **Plots** | `visualize.py` | `python -m f3p_attitude.visualize` |
| **Example paths** | `scenarios.py` (used by verify + visualize only) | — |

There are only **three** entry points (`verify`, `visualize`, and the library API). Everything else is solver internals or shared demo trajectories.

## mu is always mu

**mu** = roll about the **instantaneous velocity** (knife / bank-about-velocity).

## Usage

```bash
python -m f3p_attitude.verify
python -m f3p_attitude.visualize
```

```python
from f3p_attitude import solve_one
import numpy as np

s = solve_one(0.0, 0.0, 5.0, 0.0)  # level, V=5 m/s, mu=0
# s.quat_wb, s.thrust, s.alpha
```

Plots use **Up = -z_NED**. Geometric **alpha** = angle between body +x and **velocity** (cyan vs red on side views).

## Force balance

**m a = F_aero + T x_b + m g** with **a = v_dot** from path kinematics. At **V=0**, aero forces go to zero in `flat_plate_force`; same `solve_one` (no separate branch). Optional: `solve_static_hover` for explicit zero-wind hover tests.

## Not in scope (yet)

- Step 4 moment / delta inversion
- Full trajectory optimizer (min-snap); `scenarios.py` only has a few example paths
