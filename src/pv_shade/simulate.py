"""Synthetic PV system with known obstructions and the usual mess of real data.

The simulator runs at 5-minute resolution and reports 15-minute means labelled at the start of the interval, like
most SCADA exports. It deliberately uses richer physics than the shade method gets, so that the method's
expected-power model is never exact:

    sky         Ineichen clear sky with day-to-day turbidity, a Markov chain of sky regimes (clear, broken
                clouds, overcast) with seasonal probabilities and a two-state sun/cloud process on broken days
    plane       Perez transposition with the Sandia coefficient set (the method uses the all-sites set),
                physical IAM (the method uses ASHRAE)
    modules     SAPM cell temperature with synthetic weather, ADR efficiency (low light, temperature), 3 % DC loss
    inverter    PVWatts efficiency curve and clipping
    shading     beam loss of the scene's obstructions (pv_shade.obstructions), diffuse untouched

Disturbances that a shade method must not mistake for shade (with ``disturb=True``): measurement noise, soiling
that builds up and is washed off by rain, snow cover that slides off during the day, inverter outages (whole days
and a few hours), derating (output capped for days) and data gaps. Each is listed in the returned event table.

Columns of the returned frame (UTC index): ``p_ac`` measured AC power [kW] (NaN in gaps); ``poa`` an unshaded
plane-of-array pyranometer [W/m2]; ``t_amb``, ``t_mod`` [degC], ``wind`` [m/s] (only the autoencoder comparison
uses those); ground truth ``p_ac_unshaded`` and ``p_ac_clean`` (noise-free, without and with shading, all other
effects included), ``beam_loss`` (time-averaged share of beam power lost) and ``regime``.

Usage:
    from pv_shade.obstructions import SCENARIOS
    from pv_shade.simulate import simulate
    df, events = simulate(SCENARIOS["tree"], start="2021-01-01", end="2023-01-01", seed=1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib
from scipy.signal import lfilter

from pv_shade.obstructions import Scene, scene_beam_loss
from pv_shade.site import SystemConfig, plane_irradiance, solar_hour, sun_position

SAPM = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["close_mount_glass_glass"]
# relative efficiency of a generic c-Si module (ADR model, coefficients from the pvlib documentation example)
ADR = {"k_a": 0.99924, "k_d": -5.49097, "tc_d": 0.01918, "k_rs": 0.06999, "k_rsh": 0.26144}
DC_LOSS = 0.03
REGIMES = ("clear", "broken", "overcast")


def _ar1(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary AR(1) series with marginal standard deviation `sigma`."""
    e = rng.normal(0, sigma * np.sqrt(1 - phi**2), n)
    x, _ = lfilter([1.0], [1.0, -phi], e, zi=[phi * rng.normal(0, sigma)])
    return x


def _switching(day_pos: np.ndarray, cover: np.ndarray, rng: np.random.Generator, dwell: float = 3.0) -> np.ndarray:
    """Sun (True) / cloud (False) states: alternating geometric dwell times, mean `dwell` steps for the sun
    scaled so that the cloud share of each day is `cover[day]`."""
    out = np.empty(len(day_pos), bool)
    starts = np.flatnonzero(np.r_[True, np.diff(day_pos) != 0])
    ends = np.r_[starts[1:], len(day_pos)]
    for a, b in zip(starts, ends):
        c = float(np.clip(cover[day_pos[a]], 0.01, 0.99))
        mean = {True: dwell * (1 - c) / 0.5, False: dwell * c / 0.5}
        pos, state = a, bool(rng.random() > c)
        while pos < b:
            n = int(rng.geometric(1 / max(mean[state], 1.0)))
            out[pos:pos + n] = state
            pos, state = pos + n, not state
    return out


def _regimes(doy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Daily sky regime 0 clear, 1 broken clouds, 2 overcast: persistent Markov chain, more clear days in summer."""
    season = np.cos(2 * np.pi * (doy - 196) / 365.25)
    p_clear, p_over = 0.24 + 0.12 * season, 0.30 - 0.15 * season
    out = np.empty(len(doy), int)
    for i in range(len(doy)):
        if i and rng.random() < 0.4:
            out[i] = out[i - 1]
        else:
            out[i] = rng.choice(3, p=[p_clear[i], 1 - p_clear[i] - p_over[i], p_over[i]])
    return out


def _sky(sun: pd.DataFrame, day_pos: np.ndarray, regime: np.ndarray, system: SystemConfig,
         rng: np.random.Generator) -> pd.DataFrame:
    """Horizontal DNI / GHI / DHI for every 5-minute step from the daily regimes."""
    n_days, n = len(regime), len(sun)
    linke = pvlib.clearsky.lookup_linke_turbidity(sun.index, system.latitude, system.longitude)
    linke = linke * np.exp(_ar1(n_days, 0.7, 0.08, rng))[day_pos]
    cs = system.location.get_clearsky(sun.index, model="ineichen", linke_turbidity=linke,
                                      solar_position=sun.rename(columns={"elevation": "apparent_elevation",
                                                                         "zenith": "apparent_zenith"}))
    reg = regime[day_pos]
    cover = rng.uniform(0.15, 0.75, n_days)  # share of time behind clouds on broken days
    level = rng.uniform(0.1, 0.45, n_days)[day_pos]  # overcast clear-sky index
    sunny = _switching(day_pos, cover, rng)
    k_cloud = np.clip(0.35 + _ar1(n, 0.9, 0.1, rng), 0.1, 0.7)
    k_over = np.clip(level + _ar1(n, 0.97, 0.05, rng), 0.05, 0.6)
    beam = np.where(reg == 0, 1.0, np.where(reg == 1, sunny * 1.0, 0.0))
    dhi = np.where(reg == 0, cs["dhi"], np.where(reg == 1, np.where(sunny, 1.2 * cs["dhi"], k_cloud * cs["ghi"]),
                                                 k_over * cs["ghi"]))
    dni = beam * cs["dni"].to_numpy()
    cosz = np.clip(np.cos(np.radians(sun["zenith"].to_numpy())), 0, None)
    return pd.DataFrame({"dni": dni, "dhi": dhi, "ghi": dni * cosz + dhi}, index=sun.index)


def _events(days: pd.DatetimeIndex, rng: np.random.Generator) -> list[dict]:
    """Outages, derates, snow episodes and data gaps, a few per year, on random days."""
    ev = []
    years = max(1, round(len(days) / 365))
    pick = lambda: days[rng.integers(0, max(1, len(days) - 12))]  # noqa: E731
    for _ in range(3 * years):
        if rng.random() < 0.5:
            d0 = pick()
            ev.append({"type": "outage", "start": d0, "end": d0 + pd.Timedelta(days=int(rng.integers(1, 5))),
                       "value": 0.0})
        else:
            t0 = pick() + pd.Timedelta(hours=float(rng.uniform(8, 14)))
            ev.append({"type": "outage", "start": t0, "end": t0 + pd.Timedelta(hours=float(rng.uniform(2, 6))),
                       "value": 0.0})
    for _ in range(2 * years):
        d0 = pick()
        ev.append({"type": "derate", "start": d0, "end": d0 + pd.Timedelta(days=int(rng.integers(3, 11))),
                   "value": float(rng.uniform(0.55, 0.8))})
    winter = days[(days.month == 12) | (days.month <= 2)]
    for _ in range(2 * years if len(winter) > 10 else 0):
        d0 = winter[rng.integers(0, len(winter) - 5)]
        ev.append({"type": "snow", "start": d0, "end": d0 + pd.Timedelta(days=int(rng.integers(2, 6))),
                   "value": float(rng.uniform(3, 6))})  # hours after sunrise until the first day's snow slides off
    for _ in range(2 * years):
        d0 = pick()
        ev.append({"type": "gap", "start": d0, "end": d0 + pd.Timedelta(days=int(rng.integers(1, 8))),
                   "value": np.nan})
    return ev


def simulate(scene: Scene, system: SystemConfig = SystemConfig(), start: str = "2021-01-01",
             end: str = "2023-01-01", seed: int = 0, disturb: bool = True,
             appears: str | None = None, disturb_from: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate [start, end) in local standard time. The obstructions shade from the start, or from `appears` on
    (a new building, a grown tree). Disturbances start at `disturb_from` (default: `start`), so that a healthy
    training year can precede them. Returns the 15-minute frame and the disturbance event table."""
    rng = np.random.default_rng(seed)
    t15 = pd.date_range(start, end, freq="15min", tz=system.tz, inclusive="left")
    t5 = pd.date_range(start, end, freq="5min", tz=system.tz, inclusive="left")
    mid = t5 + pd.Timedelta(minutes=2.5)  # each 5-minute value is evaluated at its centre
    sun = sun_position(mid, system)
    local_day = t5.normalize()
    days = local_day.unique()
    day_pos = np.searchsorted(days, local_day)
    regime = _regimes(days.dayofyear.to_numpy(), rng)
    sky = _sky(sun, day_pos, regime, system, rng)
    poa = plane_irradiance(sun, sky["dni"], sky["ghi"], sky["dhi"], system, perez_set="sandiacomposite1988")

    el, az = sun["elevation"].to_numpy(), sun["azimuth"].to_numpy()
    loss = np.where(el > 0, scene_beam_loss(scene, az, el), 0.0) if scene else np.zeros(len(sun))
    if appears is not None:
        loss = np.where(t5 >= pd.Timestamp(appears, tz=system.tz), loss, 0.0)
    beam = poa["poa_beam"].to_numpy() * np.nan_to_num(pvlib.iam.physical(sun["aoi"].to_numpy()))
    diffuse = 0.95 * poa["poa_diffuse"].to_numpy()

    # weather for the module temperature (and for the autoencoder comparison)
    doy, hour = t5.dayofyear.to_numpy(), solar_hour(mid, system.longitude)
    swing = np.array([6.0, 4.0, 2.0])[regime][day_pos]
    t_amb = (8 - 10 * np.cos(2 * np.pi * (doy - 15) / 365.25) + _ar1(len(days), 0.8, 3.0, rng)[day_pos]
             + swing * np.cos(2 * np.pi * (hour - 15) / 24))
    wind = np.clip(2.5 + _ar1(len(t5), 0.995, 1.2, rng), 0.2, None)
    poa_global = poa["poa_beam"].to_numpy() + poa["poa_diffuse"].to_numpy()
    t_mod = pvlib.temperature.sapm_module(poa_global, t_amb, wind, SAPM["a"], SAPM["b"])
    t_cell = pvlib.temperature.sapm_cell_from_module(t_mod, poa_global, SAPM["deltaT"])

    # soiling builds up between rains (overcast days wash the modules with probability 0.6)
    rate = rng.uniform(0.0005, 0.0015)
    soil, s = np.empty(len(days)), 0.0
    for i in range(len(days)):
        s = 0.0 if (regime[i] == 2 and rng.random() < 0.6) else min(s + rate, 0.12)
        soil[i] = s
    healthy = days < pd.Timestamp(disturb_from or start, tz=system.tz)
    soil[healthy] = 0.0
    factor = 1 - soil[day_pos] if disturb else np.ones(len(t5))
    events = _events(days[~healthy], rng) if disturb else []

    pdc0_inv = system.pac0_kw / 0.96
    out = {}
    for key, lost in (("unshaded", 0.0), ("clean", loss)):
        eff = beam * (1 - lost) + diffuse
        eta = pvlib.pvarray.pvefficiency_adr(eff, t_cell, **ADR)
        pdc = np.where(eff > 0, system.pdc0_kw * eff / 1000 * eta * (1 - DC_LOSS), 0.0) * factor
        out[key] = np.asarray(pvlib.inverter.pvwatts(np.maximum(pdc, 0), pdc0_inv), float).clip(0)

    # events that change the plant's output (snow, derate, outage) apply to both truths
    gap = np.zeros(len(t15), bool)
    sunrise_hour = pd.Series(np.where(el > 0, hour, np.nan), index=t5).groupby(local_day).transform("min").to_numpy()
    for e in events:
        on = (t5 >= e["start"]) & (t5 < e["end"])
        if e["type"] == "snow":
            day_no = ((local_day - e["start"].normalize()) / pd.Timedelta(days=1)).to_numpy()
            clear_at = sunrise_hour + e["value"] - 1.5 * day_no  # the cover slides off earlier every day
            keep = np.clip((hour - clear_at) / 0.5, 0, 1)
            for key in out:
                out[key] = np.where(on, out[key] * keep, out[key])
        elif e["type"] == "derate":
            for key in out:
                out[key] = np.where(on, np.minimum(out[key], e["value"] * system.pac0_kw), out[key])
        elif e["type"] == "outage":
            for key in out:
                out[key] = np.where(on, 0.0, out[key])
        elif e["type"] == "gap":
            gap |= (t15 >= e["start"]) & (t15 < e["end"])

    raw = pd.DataFrame({"p_ac_unshaded": out["unshaded"], "p_ac_clean": out["clean"], "beam_loss": loss,
                        "poa": poa_global, "t_amb": t_amb, "t_mod": t_mod, "wind": wind}, index=t5)
    df = raw.groupby(np.repeat(np.arange(len(t15)), 3)).mean().set_axis(t15)
    df["regime"] = np.array(REGIMES)[regime[np.searchsorted(days, t15.normalize())]]
    p = df["p_ac_clean"].to_numpy()
    meas = p * (1 + rng.normal(0, 0.01, len(p))) + rng.normal(0, 0.002 * system.pac0_kw, len(p))
    meas = np.where(p > 0, meas.clip(0), 0.0)
    if disturb:
        gap |= (rng.random(len(p)) < 0.005) & (t15 >= pd.Timestamp(disturb_from or start, tz=system.tz))
    df.insert(0, "p_ac", np.where(gap, np.nan, meas))
    df["poa"] = (df["poa"] * (1 + rng.normal(0, 0.01, len(p)))).clip(lower=0)
    df.index = df.index.tz_convert("UTC")
    df.index.name = "time"
    ev = pd.DataFrame(events, columns=["type", "start", "end", "value"])
    return df, ev
