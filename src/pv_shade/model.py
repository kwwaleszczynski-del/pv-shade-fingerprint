"""Step 1 of the method: what the system would produce under a clear sky without obstructions.

The expected power is a physical clear-sky chain with a few calibrated coefficients:

    E(t)      = B_cs(t) * IAM(aoi) + D_cs(t)              effective clear-sky irradiance in the module plane [W/m2]
                                                           (Ineichen clear sky, Perez transposition, ASHRAE IAM)
    P_cs(t)   = E * (c0 + c1 E/1000 + c2 ln(E/1000) + c3 tau(t) + c4 D_cs/E + c5 B_cs (1/cos(aoi) - 1)/E)   [kW]
    P_exp(t)  = s_d * P_cs(t)

The bracket is an efficiency curve: c1 and c2 absorb inverter part-load and module low-light behaviour and the
heating of the modules with irradiance, tau(t) = cos(2 pi (solar hour - 15) / 24) stands in for the afternoon
peak of the air temperature, c4 corrects the weight of diffuse light (transposition model, albedo) and c5 the
incidence-angle loss of the beam. None of them can follow a feature that sits at one azimuth only.
s_d is one scale per day: soiling, degradation and the day's turbidity move all of a day up or down together
and must not be mistaken for shade, which only hits part of the day.
The coefficients are fitted by Huber regression (iteratively reweighted least squares) on clear days only and,
once a shade mask exists, only outside it.

Clear days are found from production alone, with a clearness index: s_d relative to the upper envelope of its
neighbours (soiling-proof), the smoothness of the day's normalised power (clouds make it jump) and a test for
flat tops below the inverter rating (a derated inverter is not a clear day). A day with steps of zero output
while the sun is up (snow cover, an outage) is never clear.

Usage:
    from pv_shade.model import prepare, huber_fit
    frame = prepare(p_ac, system, shift_minutes=7.5)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

from pv_shade.site import SystemConfig, clear_sky_plane, solar_hour

E_MIN = 30.0  # W/m2: below this the efficiency curve is not fitted and residuals are not used


def prepare(p_ac: pd.Series, system: SystemConfig, shift_minutes: float = 0.0) -> pd.DataFrame:
    """Measured power plus clear-sky geometry for each step. The sun position is taken `shift_minutes` after the
    timestamp (7.5 for 15-minute means labelled at the interval start)."""
    t = p_ac.index + pd.Timedelta(minutes=shift_minutes)
    sky = clear_sky_plane(t, system)
    E = sky["poa_beam"] * pvlib.iam.ashrae(sky["aoi"]).fillna(0) + sky["poa_diffuse"]
    frame = pd.DataFrame({
        "p": p_ac.to_numpy(float), "elevation": sky["elevation"].to_numpy(), "azimuth": sky["azimuth"].to_numpy(),
        "zenith": sky["zenith"].to_numpy(), "aoi": sky["aoi"].to_numpy(), "ghi_cs": sky["ghi_cs"].to_numpy(),
        "E": E.where(sky["elevation"] > 0, 0.0).to_numpy(), "poa_beam": sky["poa_beam"].to_numpy(),
        "poa_diffuse": sky["poa_diffuse"].to_numpy(),
        "tau": np.cos(2 * np.pi * (solar_hour(t, system.longitude) - 15) / 24),
    }, index=p_ac.index)
    frame["day"] = p_ac.index.tz_convert(system.tz).normalize().tz_localize(None)
    return frame


def design(frame: pd.DataFrame) -> np.ndarray:
    """Regressors of the efficiency curve (the target is P / (s_d * E)): constant, E, ln E, the diurnal
    temperature proxy, the diffuse share of E (diffuse model and albedo errors) and the beam's ASHRAE incidence
    term (IAM slope errors)."""
    E = np.clip(frame["E"].to_numpy(), E_MIN, None)
    e = E / 1000
    inc = np.clip(1 / np.cos(np.radians(np.clip(frame["aoi"].to_numpy(), 0, 89))) - 1, 0, 5)
    return np.column_stack([np.ones(len(e)), e, np.log(e), frame["tau"].to_numpy(),
                            frame["poa_diffuse"].to_numpy() / E, frame["poa_beam"].to_numpy() * inc / E])


def expected(frame: pd.DataFrame, coef: np.ndarray) -> np.ndarray:
    """Clear-sky expected power without the daily scale [kW]."""
    return np.where(frame["E"].to_numpy() >= E_MIN, frame["E"].to_numpy() * (design(frame) @ coef), 0.0)


def huber_fit(X: np.ndarray, y: np.ndarray, k: float = 1.345, iterations: int = 50) -> np.ndarray:
    """Huber M-estimate by iteratively reweighted least squares, scale = 1.4826 * MAD of the residuals."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(iterations):
        r = y - X @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
        w = np.sqrt(np.minimum(1.0, k * s / np.maximum(np.abs(r), 1e-12)))
        new = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
        if np.allclose(new, beta, rtol=1e-7, atol=1e-10):
            return new
        beta = new
    return beta


def day_table(frame: pd.DataFrame, p_cs: np.ndarray, usable: np.ndarray, system: SystemConfig,
              level: float, rough: float, quantile: float = 0.75) -> pd.DataFrame:
    """Per local day: scale s_d, clearness index relative to the envelope, roughness, flat top, clear flag.

    `usable` marks steps that may set the scale: sun high enough, not clipped, outside the current shade mask.
    """
    ratio = frame["p"].to_numpy() / np.where(p_cs > 0, p_cs, np.nan)
    d = pd.DataFrame({"day": frame["day"].to_numpy(), "ratio": np.where(usable, ratio, np.nan),
                      "p": np.where(usable, frame["p"].to_numpy(), np.nan), "p_cs": np.where(usable, p_cs, np.nan)})
    g = d.groupby("day")
    days = pd.DataFrame({"scale": g["ratio"].quantile(quantile), "steps": g["ratio"].count(),
                         "possible": g["p_cs"].count(), "dead": (d["ratio"] < 0.05).groupby(d["day"]).sum()})
    # clearness index: the day's scale against the 90th percentile of its +-15-day neighbourhood
    env = days["scale"].rolling(31, center=True, min_periods=5).quantile(0.9)
    days["clearness"] = days["scale"] / env
    # roughness: median absolute change of the scaled ratio between consecutive usable steps
    norm = d["ratio"] / d["day"].map(days["scale"])
    jump = norm.groupby(d["day"]).diff().abs()
    days["roughness"] = jump.groupby(d["day"]).median()
    # flat top below the rating: >= 5 steps within 3 % of the day's maximum (measurement noise is ~1 %) while the
    # clear-sky shape still changes by > 8 % over them (a derated inverter, not the slow top of a clear day)
    top = d["p"] >= 0.97 * d["day"].map(g["p"].max())
    flat = d[top].groupby("day").agg(n=("p", "size"), lo=("p_cs", "min"), hi=("p_cs", "max"), p=("p", "max"))
    plateau = (flat["n"] >= 5) & (flat["hi"] > 1.08 * flat["lo"]) & (flat["p"] < 0.95 * system.pac0_kw)
    days["plateau"] = plateau.reindex(days.index, fill_value=False)
    # dead steps (no output while the sun is up: snow cover, outage) disqualify a day; an obstruction never
    # takes the diffuse light, so shade alone does not produce them
    days["clear"] = ((days["clearness"] >= level) & (days["roughness"] <= rough) & ~days["plateau"]
                     & (days["dead"] == 0)
                     & (days["steps"] >= 0.8 * days["possible"]) & (days["steps"] >= 8))
    return days
