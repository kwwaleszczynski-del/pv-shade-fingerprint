import numpy as np
import pandas as pd
import pytest

from pv_shade.metrics import iou
from pv_shade.pvdaq import SYSTEMS, day_url, load


def _day_file(path, day, power_hw):
    t = pd.date_range(day, periods=96, freq="15min")
    df = pd.DataFrame({"measured_on": t.strftime("%Y-%m-%d %H:%M:%S"), "ac_power_hw__2695": power_hw,
                       "poa_irradiance__2679": 500.0, "system_id": 34})
    df.to_csv(path, index=False)
    return path


def test_load_units_timezone_and_grid(tmp_path):
    files = [_day_file(tmp_path / "a.csv", "2016-06-01", 1000.0), _day_file(tmp_path / "b.csv", "2016-06-03", -3.0)]
    df = load(files, SYSTEMS[34])
    assert df.index[0] == pd.Timestamp("2016-06-01 08:00", tz="UTC")  # local standard time UTC-8
    steps = df.index[1:] - df.index[:-1]
    assert (steps == pd.Timedelta(minutes=15)).all()
    assert df["p_ac"].iloc[0] == pytest.approx(100.0)  # hectowatt -> kW
    assert df["p_ac"].isna().sum() == 96  # the missing day stays empty
    assert (df["p_ac"].dropna() >= 0).all()  # night-time tare clipped


def test_day_url():
    assert day_url(34, pd.Timestamp("2016-02-03")).endswith(
        "system_id=34/year=2016/month=2/day=3/system_34__date_2016_02_03.csv")


def test_iou():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    assert iou(a, b) == pytest.approx(1 / 3)
    assert np.isnan(iou(np.zeros(3, bool), np.zeros(3, bool)))
