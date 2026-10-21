import numpy as np
import pandas as pd
import pytest

from pv_shade.fingerprint import (FingerprintConfig, cell_index, estimate_loss, fingerprint, fit_row_profile,
                                  relative_beam, shade_mask, silhouette)
from pv_shade.metrics import score
from pv_shade.obstructions import SCENARIOS, Row
from pv_shade.simulate import simulate
from pv_shade.site import SystemConfig

AZ = np.arange(0, 361, 2.0)
EL = np.arange(0, 91, 2.0)


def test_cell_index_edges():
    i, j = cell_index(np.array([0.0, 1.99, 2.0, 359.9]), np.array([0.0, 3.0, 89.9, 45.0]), AZ, EL)
    assert i.tolist() == [0, 0, 1, 179]
    assert j.tolist() == [0, 1, 44, 22]


def test_mask_needs_depth_days_and_size():
    shape = (len(AZ) - 1, len(EL) - 1)
    cells = {"n_days": np.full(shape, 10), "q_upper": np.zeros(shape), "median": np.zeros(shape)}
    cells["q_upper"][60:64, 5:9] = -0.4  # a 16-cell obstruction
    cells["q_upper"][100, 20] = -0.4  # one noisy cell
    cells["q_upper"][120:122, 5:7] = -0.4
    cells["n_days"][120:122, 5:7] = 2  # deep but seen on too few days
    m = shade_mask(cells, FingerprintConfig())
    assert m[60:64, 5:9].all() and m.sum() == 16


def _column_samples(i, edge, n=120, depth=-0.5, seed=0):
    rng = np.random.default_rng(seed)
    el = rng.uniform(3, 50, n)
    r = np.where(el < edge, depth, 0.0) + rng.normal(0, 0.02, n)
    return pd.DataFrame({"i": i, "j": np.searchsorted(EL, el, side="right") - 1, "el": el, "r": r,
                         "day": np.arange(n)})


def test_silhouette_stump_finds_the_edge_and_flags_censoring():
    shape = (len(AZ) - 1, len(EL) - 1)
    mask = np.zeros(shape, bool)
    observed = np.ones(shape, bool)
    mask[100, 2:10] = True  # edge at 20 deg
    mask[101, 2:25] = True  # the whole observed column is shaded
    samples = pd.concat([_column_samples(100, 20.0), _column_samples(101, 99.0, seed=1)])
    sil = silhouette(samples, mask, observed, AZ, EL, depth=0.1).set_index("i")
    assert sil.loc[100, "height"] == pytest.approx(20.0, abs=1.0)
    assert not sil.loc[100, "censored"]
    assert sil.loc[101, "censored"]


def test_row_fit_recovers_critical_angle_and_gcr():
    row = Row(tilt=30.0, gcr=0.4)
    az = np.arange(110.0, 251.0, 2.0)
    sil = pd.DataFrame({"azimuth": az, "height": row.top(az), "censored": False})
    fit = fit_row_profile(sil, tilt=30.0)
    assert fit["critical_angle_deg"] == pytest.approx(row.critical_angle, abs=0.01)
    assert fit["gcr"] == pytest.approx(0.4, abs=0.001)
    assert fit["row_azimuth_deg"] == pytest.approx(180.0, abs=0.01)


def test_relative_beam_and_loss_formula():
    frame = pd.DataFrame({"p": [100.0, 30.0, 65.0, np.nan]})
    p_ref = np.array([100.0, 100.0, 100.0, 100.0])
    b = relative_beam(frame, p_ref, k_cloud=0.3)
    assert b[:3] == pytest.approx([1.0, 0.0, 0.5])
    system = SystemConfig(pac0_kw=85.0)
    loss = estimate_loss(frame, np.array([0.5, 0.5, 0.0, 0.5]), np.array([1.0, 1.0, 1.0, 1.0]), p_ref, system)
    # clear step: inverter limit 85 kW, half the light gone -> 85 - 50; no depth or no data -> no loss
    assert loss.tolist() == pytest.approx([35.0, 35.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def building_year():
    df, _ = simulate(SCENARIOS["building"], start="2021-01-01", end="2022-01-01", seed=11)
    return df


def test_end_to_end_building(building_year):
    fp = fingerprint(building_year["p_ac"], SystemConfig())
    m = score(fp, SCENARIOS["building"], building_year)
    assert 5.0 < fp.shift_minutes < 10.0  # 15-minute means labelled at the interval start: 7.5 min
    assert m["iou"] > 0.75
    assert m["horizon_rmse_deg"] < 4.0
    assert abs(m["loss_error_pp"]) < 1.5
    assert m["false_cells"] <= 10


def test_end_to_end_no_obstruction():
    df, _ = simulate((), start="2021-01-01", end="2022-01-01", seed=12)
    fp = fingerprint(df["p_ac"], SystemConfig())
    assert fp.mask.sum() <= 5
    assert fp.loss_fraction < 0.001


def test_silhouette_ignores_a_single_bad_day_above_the_edge():
    shape = (len(AZ) - 1, len(EL) - 1)
    mask = np.zeros(shape, bool)
    mask[100, 2:10] = True  # edge at 20 deg
    observed = np.zeros(shape, bool)  # nothing seen clear above the mask: the search is not bounded
    rng = np.random.default_rng(3)
    el = np.r_[rng.uniform(3, 19.5, 40), 28.0, rng.uniform(28.5, 50, 20)]  # no samples between 19.5 and 28
    r = np.where(el < 20, -0.5, 0.0) + rng.normal(0, 0.01, len(el))
    r[40] = -0.3  # one derated day that slipped through the clear-day filter
    sparse = pd.DataFrame({"i": 100, "el": el, "r": r, "day": np.arange(len(el))})
    mask[101, 2:10] = True
    dense = _column_samples(101, 20.0, seed=4)  # the neighbouring column saw the edge
    sil = silhouette(pd.concat([sparse, dense]), mask, observed, AZ, EL, depth=0.1).set_index("i")
    assert sil.loc[100, "low"] < 20.0  # the bad day is not the last shaded sample
    assert sil.loc[100, "height"] == pytest.approx(20.0, abs=1.0)  # gap filled from the neighbour
