# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
"""Stage lists of 2nd-order (Strang) steps for each product formula.

Each stage is one S2(dt): every fragment with angle dt/2 in order,
then in reverse. Order 4 is Suzuki's composition
S4(dt) = S2(p dt)^2 S2((1-4p) dt) S2(p dt)^2.
"""

import numpy as np

P4 = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))


def stages(t, steps, order):
    """Stage step sizes for one repetition; repeated `steps` times."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    dt = t / steps
    if order == 2:
        return np.array([dt])
    if order == 4:
        return np.array([P4 * dt, P4 * dt, (1 - 4 * P4) * dt,
                         P4 * dt, P4 * dt])
    raise ValueError("order must be 2 or 4")
