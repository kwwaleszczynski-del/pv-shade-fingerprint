"""What does each model explain? Shade fingerprint vs. the window autoencoder of pv-anomaly-autoencoder.

The autoencoder (https://github.com/kwwaleszczynski-del/pv-anomaly-autoencoder) learns normal behaviour from a
healthy year of power, irradiance and temperature and flags windows it cannot reconstruct. It is run here exactly
as in its own repository (same features, architecture, smoothing and 99.5 % validation threshold) on the synthetic
plant of this project, in two situations:

    always    the building has stood next to the array all along: it shades the training year too
    appears   the building is put up on 1 January of the scored year: training year clean

The training year is healthy, as the autoencoder's protocol requires; the scored year carries the usual
disturbances (outages, derating, snow, soiling, gaps). The fingerprint gets the scored year's AC power only. Scores:

    shaded steps flagged    share of clear-sky daylight steps with >= 10 % beam loss that the autoencoder alarms
    alarm composition       share of alarmed steps that fall in shade, in a disturbance event, or in neither
    events flagged          outage / derate / snow events with at least one alarm
    sun-space IoU           cells where the alarm rate on clear days exceeds 50 %, against the true shade mask
                            (the autoencoder's score of a window is attributed to the window's middle)

Needs the optional dependencies: ``pip install torch`` and the pv-anomaly-autoencoder package.

Usage:
    from pv_shade.compare import compare
    table, cases = compare(seed=1, data_dir=Path("data/synthetic"))
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pv_shade.fingerprint import Fingerprint, cell_index, fingerprint
from pv_shade.metrics import MATERIAL, iou, score, truth_mask
from pv_shade.obstructions import SCENARIOS
from pv_shade.simulate import simulate
from pv_shade.site import SystemConfig

log = logging.getLogger(__name__)

TRAIN_END = "2022-01-01"
SCENE = "building"


@dataclass
class Comparison:
    situation: str
    df: pd.DataFrame  # scored year
    events: pd.DataFrame
    fp: Fingerprint
    ae_score: pd.Series  # raw window score, attributed to the window's middle
    alarm: pd.Series
    ae_rate: np.ndarray  # (n_az, n_el) alarm rate on clear days, NaN where the sun never was
    metrics: dict


def _autoencoder(df: pd.DataFrame, system: SystemConfig, seed: int):
    """Train and score the pv-anomaly-autoencoder detector as its own repository does. Returns the raw score (at
    the window's end), the smoothed score, the alarm flags and the daylight flags."""
    from pv_anomaly.autoencoder import AEConfig, AutoencoderDetector
    from pv_anomaly.detect import calibrate, smooth
    from pv_anomaly.features import build_features, split_masks
    from pv_anomaly.plant import PlantConfig

    plant = PlantConfig(latitude=system.latitude, longitude=system.longitude, altitude=system.altitude,
                        tz=system.tz, tilt=system.tilt, azimuth=system.azimuth, pdc0_kw=system.pdc0_kw,
                        pac0_kw=system.pac0_kw)
    feats = build_features(df, plant)
    masks = split_masks(df.index, plant.tz, TRAIN_END, val_frac=0.2, seed=seed)
    ae = AutoencoderDetector(AEConfig(seed=seed))
    ae.fit(feats, masks["train"], masks["val"])
    raw = ae.score(feats)
    day = feats["daylight"].to_numpy()
    s = smooth(raw, day, 8, signed=True)
    thr = calibrate(s, masks["val"] & day, 0.995)
    alarm = (s > thr).fillna(False) & masks["test"] & day
    return raw, alarm, ae.config.window


def _event_mask(index: pd.DatetimeIndex, events: pd.DataFrame, kinds) -> np.ndarray:
    on = np.zeros(len(index), bool)
    for e in events[events["type"].isin(kinds)].itertuples():
        on |= (index >= e.start) & (index < e.end)
    return on


def run(situation: str, seed: int, data_dir: Path | None, system: SystemConfig = SystemConfig()) -> Comparison:
    path = None if data_dir is None else Path(data_dir) / f"compare_{situation}_{seed}_v2.parquet"
    if path is not None and path.exists():
        df, events = pd.read_parquet(path), pd.read_parquet(path.with_suffix(".events.parquet"))
    else:
        df, events = simulate(SCENARIOS[SCENE], system, start="2021-01-01", end="2023-01-01", seed=seed,
                              appears=TRAIN_END if situation == "appears" else None, disturb_from=TRAIN_END)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)
            events.to_parquet(path.with_suffix(".events.parquet"))
    raw, alarm, window = _autoencoder(df, system, seed)
    test = df.index >= pd.Timestamp(TRAIN_END, tz=system.tz)
    year, alarm = df[test], alarm[test]
    fp = fingerprint(year["p_ac"], system)
    # attribute each window score to the middle of its window (it ends at its timestamp)
    mid = raw.shift(-(window // 2))[test]

    scene = SCENARIOS[SCENE]
    f = fp.frame
    clear_sky_day = (year["regime"] == "clear").to_numpy()
    events_on = _event_mask(year.index, events, ("outage", "derate", "snow"))
    shaded = (year["beam_loss"].to_numpy() >= MATERIAL) & (f["elevation"].to_numpy() > 5)
    a = alarm.to_numpy(bool)
    lit = f["elevation"].to_numpy() > 5
    target = shaded & clear_sky_day & ~events_on & np.isfinite(year["p_ac"].to_numpy())
    flagged = [e for e in events[events["type"].isin(["outage", "derate", "snow"])].itertuples()
               if e.start >= pd.Timestamp(TRAIN_END, tz=system.tz)
               and a[(year.index >= e.start) & (year.index < e.end)].any()]
    n_events = int(((events["type"].isin(["outage", "derate", "snow"]))
                    & (events["start"] >= pd.Timestamp(TRAIN_END, tz=system.tz))).sum())

    # alarm rate per sun-space cell on clear days (attributed to window middles)
    a_mid = pd.Series(a, index=year.index).shift(-(window // 2), fill_value=False).to_numpy(bool)
    use = clear_sky_day & lit & ~events_on & (f["aoi"].to_numpy() < 80)
    i, j = cell_index(f["azimuth"].to_numpy()[use], f["elevation"].to_numpy()[use], fp.az_edges, fp.el_edges)
    rate = pd.Series(a_mid[use].astype(float)).groupby([i, j]).mean()
    ae_rate = np.full(fp.mask.shape, np.nan)
    ae_rate[rate.index.get_level_values(0), rate.index.get_level_values(1)] = rate.to_numpy()
    ae_map = np.nan_to_num(ae_rate) > 0.5
    obs = fp.observed
    tm = truth_mask(fp, scene) & obs

    fp_m = score(fp, scene, year)
    metrics = {
        "situation": situation, "seed": seed,
        "ae_shaded_steps_flagged": float(a[target].mean()) if target.any() else np.nan,
        "ae_alarm_steps": int(a.sum()),
        "ae_alarms_in_shade": float((a & shaded).sum() / max(a.sum(), 1)),
        "ae_alarms_in_events": float((a & events_on & ~shaded).sum() / max(a.sum(), 1)),
        "ae_alarms_other": float((a & ~events_on & ~shaded).sum() / max(a.sum(), 1)),
        "ae_events_flagged": f"{len(flagged)}/{n_events}",
        "ae_sun_space_iou": iou(ae_map & obs, tm),
        "fp_iou": fp_m["iou"], "fp_horizon_rmse_deg": fp_m["horizon_rmse_deg"],
        "fp_true_loss_pct": fp_m["true_loss_pct"], "fp_estimated_loss_pct": fp_m["estimated_loss_pct"],
    }
    log.info("%s seed %d: autoencoder flags %.0f %% of shaded clear-sky steps, sun-space IoU %.2f; fingerprint IoU "
             "%.2f, loss %.2f %% (true %.2f %%)", situation, seed, 100 * metrics["ae_shaded_steps_flagged"],
             metrics["ae_sun_space_iou"], metrics["fp_iou"], metrics["fp_estimated_loss_pct"],
             metrics["fp_true_loss_pct"])
    return Comparison(situation, year, events, fp, mid, pd.Series(a_mid, index=year.index), ae_rate, metrics)


def compare(seed: int = 1, data_dir: Path | None = None) -> tuple[pd.DataFrame, list[Comparison]]:
    cases = [run(s, seed, data_dir) for s in ("always", "appears")]
    return pd.DataFrame([c.metrics for c in cases]), cases
