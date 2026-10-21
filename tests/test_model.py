import numpy as np
import pandas as pd
import pytest

from pv_shade.model import day_table, design, expected, huber_fit, prepare
from pv_shade.site import SystemConfig


def test_huber_ignores_outliers():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 500)
    X = np.column_stack([np.ones_like(x), x])
    y = 2 + 3 * x + rng.normal(0, 0.01, len(x))
    y[:50] -= 5  # 10 % gross outliers (shade, clouds)
    beta = huber_fit(X, y)
    assert beta == pytest.approx([2, 3], abs=0.05)
    assert np.linalg.lstsq(X, y, rcond=None)[0][0] < 1.6  # ordinary least squares is pulled down


def _clear_days(n_days=6, system=SystemConfig()):
    idx = pd.date_range("2021-06-01", periods=96 * n_days, freq="15min", tz=system.tz).tz_convert("UTC")
    frame = prepare(pd.Series(0.0, index=idx), system)
    coef = np.r_[0.1, np.zeros(design(frame[:1]).shape[1] - 1)]
    p_cs = expected(frame, coef)
    return frame, p_cs


def test_expected_power_follows_the_sun():
    frame, p_cs = _clear_days(1)
    assert (p_cs[frame["elevation"].to_numpy() <= 0] == 0).all()
    assert p_cs.max() > 0.7 * SystemConfig().pdc0_kw


def test_day_table_flags_clear_derated_and_dead_days():
    system = SystemConfig()
    frame, p_cs = _clear_days(6, system)
    p = p_cs * 0.8
    day = frame["day"].to_numpy()
    days = np.unique(day)
    noise = 1 + np.random.default_rng(2).normal(0, 0.01, len(p))  # 1 % measurement noise on the flat top
    p = np.where(day == days[1], np.minimum(p, 0.5 * system.pac0_kw) * noise, p)  # derated inverter
    noon = frame["elevation"].to_numpy() > 40
    p = np.where((day == days[2]) & noon, 0.0, p)  # outage at midday
    rng = np.random.default_rng(1)
    p = np.where(day == days[3], p * rng.uniform(0.2, 1.0, len(p)), p)  # broken clouds
    frame = frame.assign(p=p)
    usable = (frame["elevation"].to_numpy() >= 10) & (p_cs > 0)
    t = day_table(frame, p_cs, usable, system, level=0.9, rough=0.03)
    assert t.loc[days[0], "clear"] and t.loc[days[4], "clear"]
    assert t.loc[days[1], "plateau"] and not t.loc[days[1], "clear"]
    assert t.loc[days[2], "dead"] > 0 and not t.loc[days[2], "clear"]
    assert not t.loc[days[3], "clear"]
    assert t.loc[days[0], "scale"] == pytest.approx(0.8, rel=1e-6)
