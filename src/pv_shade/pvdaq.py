"""NREL PVDAQ public data: one system's daily CSV files, loaded as AC power on a regular grid.

Source: Deline, C., Perry, K., Deceglie, M., Muller, M., Sekulic, W., & Jordan, D. (2021). Photovoltaic Data
Acquisition (PVDAQ) Public Datasets. Open Energy Data Initiative (OEDI), NREL. https://doi.org/10.25984/1846021,
licensed CC BY 4.0. The bucket holds one CSV per system and day under BASE_URL.

Only AC power is used by the shade method. The plane-of-array pyranometer is read too, as an independent check:
if the array is shaded and the sensor is not, the ratio power / sensor shows the same silhouette.

Usage:
    from pv_shade.pvdaq import SYSTEMS, fetch, load
    files = fetch(34, "2016-01-01", "2017-12-31", Path("data/pvdaq"))   # skips files already present
    df = load(sorted(Path("data/pvdaq/system_34").glob("*.csv")), SYSTEMS[34])
"""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import pandas as pd

from pv_shade.site import SystemConfig

log = logging.getLogger(__name__)

BASE_URL = "https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/pvdata"

# PVDAQ system 34 (metadata of the public dataset): school building in Las Vegas NV, 611 x 240 W mono-Si modules on a
# flat roof, 11.2 deg tilt facing south, one 135 kW central inverter. Timestamps are local standard time (UTC-8).
SYSTEMS = {34: SystemConfig(name="PVDAQ system 34", latitude=36.1952, longitude=-115.1582, altitude=620.0,
                            tz="Etc/GMT+8", tilt=11.2, azimuth=180.0, pdc0_kw=146.64, pac0_kw=135.0)}
COLUMNS = {34: {"ac_power_hw__2695": "p_ac", "poa_irradiance__2679": "poa"}}
SCALE = {34: {"p_ac": 0.1, "poa": 1.0}}  # hectowatt -> kW


def day_url(system_id: int, day: pd.Timestamp) -> str:
    return (f"{BASE_URL}/system_id={system_id}/year={day.year}/month={day.month}/day={day.day}/"
            f"system_{system_id}__date_{day:%Y_%m_%d}.csv")


def fetch(system_id: int, start: str, end: str, data_dir: Path) -> list[Path]:
    """Download the daily files from `start` to `end` (inclusive) into data_dir/system_<id>; existing files are
    kept. About 11 kB per day for system 34."""
    out = Path(data_dir) / f"system_{system_id}"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for day in pd.date_range(start, end, freq="D"):
        path = out / f"system_{system_id}__date_{day:%Y_%m_%d}.csv"
        if not path.exists():
            with urllib.request.urlopen(day_url(system_id, day), timeout=60) as r:
                path.write_bytes(r.read())
        paths.append(path)
    log.info("PVDAQ system %d: %d daily files in %s", system_id, len(paths), out)
    return paths


def load(files: list[Path], system: SystemConfig, system_id: int = 34) -> pd.DataFrame:
    """Daily CSV files -> regular 15-minute UTC grid with ``p_ac`` [kW] and ``poa`` [W/m2]. Missing steps stay NaN,
    night-time tare readings are set to 0."""
    cols = COLUMNS[system_id]
    raw = pd.concat([pd.read_csv(f).reindex(columns=["measured_on", *cols]) for f in files], ignore_index=True)
    t = pd.to_datetime(raw["measured_on"]).dt.round("15min").dt.tz_localize(system.tz).dt.tz_convert("UTC")
    raw = raw.rename(columns=cols).set_index(t).drop(columns="measured_on")
    raw = raw[~raw.index.duplicated()]
    grid = pd.date_range(raw.index.min(), raw.index.max(), freq="15min")
    df = raw.reindex(grid)
    for c, k in SCALE[system_id].items():
        df[c] = (df[c] * k).clip(lower=0)
    df.index.name = "time"
    log.info("loaded %d steps %s .. %s, %.1f %% without AC power", len(df), grid[0], grid[-1],
             100 * df["p_ac"].isna().mean())
    return df
