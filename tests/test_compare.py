"""The comparison needs torch and the pv-anomaly-autoencoder package; it is skipped where they are missing (CI)."""
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("pv_anomaly")

from pv_shade.compare import _event_mask  # noqa: E402


def test_event_mask():
    idx = pd.date_range("2022-01-01", periods=8, freq="1h", tz="UTC")
    ev = pd.DataFrame({"type": ["outage", "gap"], "start": [idx[2], idx[0]], "end": [idx[4], idx[8 - 1]],
                       "value": [0.0, None]})
    on = _event_mask(idx, ev, ("outage",))
    assert on.tolist() == [False, False, True, True, False, False, False, False]
