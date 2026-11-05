# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
import numpy as np
import pytest

from trotterkit.schedule import P4, stages


def test_order2_is_one_stage():
    np.testing.assert_array_equal(stages(2.0, 4, 2), [0.5])


def test_order4_stages_sum_to_dt():
    s = stages(1.0, 10, 4)
    assert len(s) == 5
    assert s.sum() == pytest.approx(0.1, abs=1e-15)
    assert s[0] == s[1] == s[3] == s[4] == pytest.approx(P4 * 0.1)
    assert s[2] == pytest.approx((1 - 4 * P4) * 0.1)


@pytest.mark.parametrize("bad", [(1.0, 0, 2), (1.0, 3, 3)])
def test_rejects_bad_arguments(bad):
    with pytest.raises(ValueError):
        stages(*bad)
