import numpy as np
import pandas as pd

from pv_shade.obstructions import SCENARIOS
from pv_shade.simulate import simulate
from pv_shade.site import SystemConfig


def test_columns_grid_and_truth():
    df, events = simulate(SCENARIOS["building"], start="2021-10-01", end="2021-12-01", seed=3)
    assert {"p_ac", "poa", "t_amb", "t_mod", "wind", "p_ac_unshaded", "p_ac_clean", "beam_loss"} <= set(df.columns)
    steps = df.index[1:] - df.index[:-1]
    assert (steps == pd.Timedelta(minutes=15)).all()  # independent of the datetime resolution (us / ns)
    assert str(df.index.tz) == "UTC"
    assert (df["p_ac_unshaded"] >= df["p_ac_clean"] - 1e-9).all()
    assert (df["p_ac_clean"] <= SystemConfig().pac0_kw + 1e-9).all()
    assert df["beam_loss"].max() > 0.5  # the building shades in autumn afternoons
    assert set(events["type"]) <= {"outage", "derate", "snow", "gap"}


def test_no_scene_no_disturbances_is_clean():
    df, events = simulate((), start="2021-06-01", end="2021-06-15", seed=1, disturb=False)
    assert events.empty
    assert df["p_ac"].notna().all()
    assert np.allclose(df["p_ac_unshaded"], df["p_ac_clean"])
    assert (df["beam_loss"] == 0).all()


def test_obstruction_can_appear_later_and_disturbances_start_late():
    df, events = simulate(SCENARIOS["building"], start="2021-10-01", end="2021-12-01", seed=2,
                          appears="2021-11-01", disturb_from="2021-11-01")
    before = df.index < pd.Timestamp("2021-11-01", tz=SystemConfig().tz)
    assert (df.loc[before, "beam_loss"] == 0).all()
    assert df.loc[~before, "beam_loss"].max() > 0.5
    assert (events["start"] >= pd.Timestamp("2021-11-01", tz=SystemConfig().tz)).all()
    assert df.loc[before, "p_ac"].notna().all()


def test_seed_reproducible():
    a, _ = simulate(SCENARIOS["pole"], start="2021-03-01", end="2021-03-08", seed=5)
    b, _ = simulate(SCENARIOS["pole"], start="2021-03-01", end="2021-03-08", seed=5)
    pd.testing.assert_frame_equal(a, b)
