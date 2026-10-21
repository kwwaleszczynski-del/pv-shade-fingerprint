"""Sun-path residual fingerprinting: find, draw and cost the shading of a PV system from its power alone.

A fixed obstruction always blocks the sun at the same place in the sky, whatever the date or the time. Clouds,
outages and soiling do not care where the sun is. So if the shortfall of each clear-sky measurement is plotted not
against time but against the sun's position, shade piles up in one place while everything else spreads out. The
method has four steps:

1. Expected power (pv_shade.model): clear-sky physics, a daily scale s_d and a robustly calibrated efficiency
   curve, fitted on clear days outside the current shade mask. The clock offset of the data (interval start or
   end labels, logger drift) is estimated by trying shifts and keeping the one with the smallest residuals.
2. Residuals in sun space: r = P / (s_d * P_cs) - 1 on clear days, binned on an azimuth x elevation grid
   (2 x 2 deg). Each cell keeps robust statistics over all days the sun passed it: count of days, median and
   upper quartile q75. Clouds that slipped through the clear-day filter pull single samples down; q75 ignores
   them unless they happen on more than a quarter of the visits.
3. Shade mask: cells where q75 < -depth (even the better visits are short of power) seen on enough days, small
   isolated patches removed. Steps 1-4 are iterated: the mask and everything below the silhouette are left out
   of the next calibration and daily scales, so the shade itself does not bend the expected-power model.
4. Silhouette and loss. For each 2-deg azimuth column the obstruction's height is the elevation that best
   separates shaded samples (r < -depth / 2) below from unshaded ones above (a decision stump). Censored
   columns, where the sun was never seen above the obstruction, are flagged. The energy loss of every step that
   falls in the mask, cloudy steps included, is depth(cell) x P_ref(t) x b(t): the clear-sky unshaded power times
   the share of the clear-sky beam that was there. An obstruction only takes the beam, so it costs little when
   clouds have taken the beam already. b(t) is read from the step's unshaded neighbours on the same day.

Usage:
    from pv_shade.fingerprint import FingerprintConfig, fingerprint
    fp = fingerprint(df["p_ac"], system)
    fp.mask, fp.silhouette, fp.loss_fraction
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage

from pv_shade.model import E_MIN, day_table, design, expected, huber_fit, prepare
from pv_shade.site import SystemConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FingerprintConfig:
    az_step: float = 2.0  # grid resolution [deg]
    el_step: float = 2.0
    pool: int = 1  # cells pooled above and below (elevation only: azimuth keeps full resolution)
    min_elevation: float = 3.0
    max_aoi: float = 80.0  # sun (nearly) in or behind the module plane: almost no beam to shade
    quantile: float = 0.75  # cell statistic for the mask: upper quartile of the residual
    depth: float = 0.10  # a cell is shaded when its q75 residual is below -depth
    min_days: int = 3  # clear days needed in a pooled cell to judge it
    min_patch: int = 3  # smallest shaded patch kept [cells]
    clear_level: float = 0.9  # clearness index (day scale / envelope) of a clear day
    clear_rough: float = 0.03  # max median step-to-step change of a clear day's normalised power
    clip: float = 0.97  # steps at or above this share of the inverter rating are not used
    iterations: int = 3
    k_cloud: float = 0.3  # clearness index of a step with the sun behind a cloud (loss estimate)
    shift_minutes: float | None = None  # clock offset; None = estimate from the data
    shift_grid: tuple[float, ...] = tuple(np.arange(-15.0, 22.6, 2.5))


@dataclass
class Fingerprint:
    config: FingerprintConfig
    system: SystemConfig
    shift_minutes: float
    coef: np.ndarray
    frame: pd.DataFrame  # per step: p, geometry, p_exp, residual, cell, clear flag, estimated shade loss
    days: pd.DataFrame
    az_edges: np.ndarray
    el_edges: np.ndarray
    cells: dict[str, np.ndarray]  # n_days, median, q_upper: (n_az, n_el) arrays, pooled
    mask: np.ndarray  # (n_az, n_el) bool
    silhouette: pd.DataFrame  # per azimuth column: height, censored, samples
    history: list[dict] = field(default_factory=list)

    @property
    def az_centers(self) -> np.ndarray:
        return (self.az_edges[:-1] + self.az_edges[1:]) / 2

    @property
    def el_centers(self) -> np.ndarray:
        return (self.el_edges[:-1] + self.el_edges[1:]) / 2

    @property
    def observed(self) -> np.ndarray:
        return self.cells["n_days"] >= self.config.min_days

    @property
    def loss_fraction(self) -> float:
        """Estimated shade loss as a share of what the system would have produced unshaded (steps with data)."""
        f = self.frame
        lost = f["shade_loss"].sum()
        return float(lost / (f["p"].sum() + lost))

    def monthly_loss(self) -> pd.DataFrame:
        f = self.frame
        month = f.index.tz_convert(self.system.tz).tz_localize(None).to_period("M")
        g = f.groupby(month)
        return pd.DataFrame({"energy_kwh": g["p"].sum() * self._hours, "loss_kwh": g["shade_loss"].sum() * self._hours})

    @property
    def _hours(self) -> float:
        return (self.frame.index[1] - self.frame.index[0]) / pd.Timedelta(hours=1)


def _grid(cfg: FingerprintConfig) -> tuple[np.ndarray, np.ndarray]:
    return np.arange(0, 360 + cfg.az_step / 2, cfg.az_step), np.arange(0, 90 + cfg.el_step / 2, cfg.el_step)


def cell_index(az: np.ndarray, el: np.ndarray, az_edges, el_edges) -> tuple[np.ndarray, np.ndarray]:
    i = np.clip(np.searchsorted(az_edges, az, side="right") - 1, 0, len(az_edges) - 2)
    j = np.clip(np.searchsorted(el_edges, el, side="right") - 1, 0, len(el_edges) - 2)
    return i, j


def cell_stats(samples: pd.DataFrame, az_edges, el_edges, pool: int, q: float) -> dict[str, np.ndarray]:
    """Pooled per-cell statistics of residual samples (columns i, j, day, r)."""
    shape = (len(az_edges) - 1, len(el_edges) - 1)
    parts = [samples.assign(j=samples["j"] + dj) for dj in range(-pool, pool + 1)]
    s = pd.concat(parts, ignore_index=True)
    s = s[(s["j"] >= 0) & (s["j"] < shape[1])]
    g = s.groupby(["i", "j"])["r"]
    stats = pd.DataFrame({"n_days": s.groupby(["i", "j"])["day"].nunique(), "median": g.median(),
                          "q_upper": g.quantile(q)})
    out = {k: np.full(shape, np.nan) for k in stats.columns}
    ii, jj = stats.index.get_level_values(0), stats.index.get_level_values(1)
    for k in stats.columns:
        out[k][ii, jj] = stats[k].to_numpy()
    out["n_days"] = np.nan_to_num(out["n_days"]).astype(int)
    return out


def shade_mask(cells: dict[str, np.ndarray], cfg: FingerprintConfig) -> np.ndarray:
    """Cells persistently short of power, without patches smaller than `min_patch`."""
    raw = (cells["n_days"] >= cfg.min_days) & (np.nan_to_num(cells["q_upper"], nan=0.0) < -cfg.depth)
    lab, n = ndimage.label(raw, structure=np.ones((3, 3)))
    if n == 0:
        return raw
    size = ndimage.sum(raw, lab, index=np.arange(1, n + 1))
    return np.isin(lab, 1 + np.flatnonzero(size >= cfg.min_patch))


def shade_region(mask: np.ndarray, sil: pd.DataFrame, el_edges) -> np.ndarray:
    """Mask plus every cell below the reconstructed silhouette: left out of the daily scale and the calibration
    in the next iteration, so that partly shaded days do not pull the expected power down."""
    region = mask.copy()
    el_mid = (el_edges[:-1] + el_edges[1:]) / 2
    for row in sil.itertuples():
        if np.isfinite(row.height):
            region[row.i] |= el_mid < row.height
    return region


def silhouette(samples: pd.DataFrame, mask: np.ndarray, observed: np.ndarray, az_edges, el_edges, depth: float,
               pool: int = 1) -> pd.DataFrame:
    """Obstruction height per azimuth column from a decision stump on the column's samples.

    The height h minimises (#unshaded samples below h) + (#shaded samples above h), a sample being shaded when
    r < -depth / 2. Above the mask, a shaded sample needs a shaded sample of another day within one cell, else it
    is dropped: one bad day must not lift the edge where the 15-minute sampling leaves no other evidence. Only columns with at least one masked cell get a height, and h is searched between the pooling
    blur below the mask's upper edge and the first cell above it that was observed clear (the mask decides where
    shade is, the stump how high it reaches); the middle of equally good splits is taken. ``censored`` is True
    when no unshaded sample lies above h: the obstruction may be taller than anything the sun showed.
    """
    rows = []
    step = el_edges[1] - el_edges[0]
    for i in np.flatnonzero(mask.any(axis=1)):
        col = samples[samples["i"] == i].sort_values("el")
        el, shaded = col["el"].to_numpy(dtype=float, copy=True), (col["r"] < -depth / 2).to_numpy(copy=True)
        top_masked = el_edges[np.flatnonzero(mask[i])[-1] + 1]
        if len(el) == 0:
            continue
        # above the mask, a shaded sample counts only if a shaded sample of another day lies within one cell of it
        # (the persistence the mask asks for); a single bad day is left out of the count
        day = col["day"].to_numpy()
        for n in np.flatnonzero(shaded & (el > top_masked)):
            if not (shaded & (day != day[n]) & (np.abs(el - el[n]) <= step)).any():
                shaded[n] = False
                el[n] = np.nan
        keep = np.isfinite(el)
        el, shaded = el[keep], shaded[keep]
        # cost of a split after k samples (k = 0 .. n): unshaded below + shaded above
        cost = (np.concatenate([[0], np.cumsum(~shaded)])
                + shaded.sum() - np.concatenate([[0], np.cumsum(shaded)])).astype(float)
        split_el = np.r_[el[0] - 0.01, (el[:-1] + el[1:]) / 2, el[-1] + 0.01]
        top_j = np.flatnonzero(mask[i])[-1]
        clear_above = np.flatnonzero(observed[i, top_j + 1:] & ~mask[i, top_j + 1:])
        upper = el_edges[top_j + 2 + clear_above[0]] if len(clear_above) else np.inf
        near = (split_el >= top_masked - (pool + 1) * step) & (split_el <= upper)
        if near.any():
            cost[~near] = np.inf
        best = np.flatnonzero(cost == cost.min())
        k = int(best[len(best) // 2])
        h = np.nan if k == 0 else (el[-1] if k == len(el) else split_el[k])
        rows.append({"azimuth": (az_edges[i] + az_edges[i + 1]) / 2, "i": i, "height": h,
                     "low": el[k - 1] if k > 0 else np.nan, "high": el[k] if 0 < k < len(el) else np.nan,
                     "censored": bool(k > 0 and not (~shaded[k:]).any()), "samples": len(el),
                     "mask_top": top_masked})
    out = pd.DataFrame(rows, columns=["azimuth", "i", "height", "low", "high", "censored", "samples", "mask_top"])
    # the edge lies anywhere between the column's last shaded and first unshaded sample; where that gap is wide
    # (the 15-minute sampling leaves some columns without samples near the edge), take the neighbours' median
    # height, kept inside the gap
    first = out["height"].to_numpy(float)
    pos = {i: n for n, i in enumerate(out["i"])}
    for n, row in enumerate(out.itertuples()):
        if np.isfinite(row.high) and row.high - row.low > 2 * step:
            near = [first[pos[j]] for j in (row.i - 1, row.i + 1) if j in pos and np.isfinite(first[pos[j]])]
            if near:
                out.loc[n, "height"] = float(np.clip(np.median(near), row.low, row.high))
    return out


def relative_beam(frame: pd.DataFrame, p_ref: np.ndarray, k_cloud: float = 0.3) -> np.ndarray:
    """Share of the clear-sky beam present in each step, 0..1, from its production-based clearness index
    k = P / P_ref. Two-state sky: in a 15-minute step the sun is either out (k ~ 1) or behind a cloud
    (k ~ k_cloud, diffuse only), so the share of time it was out is (k - k_cloud) / (1 - k_cloud)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        k = frame["p"].to_numpy() / np.where(p_ref > 0, p_ref, np.nan)
    return np.clip((k - k_cloud) / (1 - k_cloud), 0, 1)


def sky_beam(frame: pd.DataFrame, beam_rel: np.ndarray, shaded: np.ndarray, window: int = 9) -> np.ndarray:
    """Expected relative beam of each step, read from its unshaded neighbours on the same day: centred rolling
    mean over `window` steps with shaded steps left out (on a day of broken clouds, the share of sunny moments);
    the day's mean where no neighbour is left."""
    b = pd.Series(np.where(shaded, np.nan, beam_rel), index=frame.index)
    day = frame["day"].to_numpy()
    near = b.groupby(day).transform(lambda s: s.rolling(window, center=True, min_periods=1).mean())
    return near.fillna(b.groupby(day).transform("mean")).fillna(0.0).to_numpy()


def estimate_loss(frame: pd.DataFrame, depth_of_step: np.ndarray, beam: np.ndarray, p_ref: np.ndarray,
                  system: SystemConfig) -> np.ndarray:
    """Expected shade loss [kW] of each step with data, all weather:

        loss = beam * (min(P_ref, P_ac0) - min(P_ref * (1 - depth), P_ac0))

    depth_of_step is the clear-sky relative deficit of the step's cell (>= 0), beam the expected beam share
    relative to a clear sky (0..1, read from unshaded neighbours), P_ref the clear-sky unshaded power. The
    inverter limit matters: shade on an oversized array costs nothing while the inverter clips.
    """
    ok = (depth_of_step > 0) & np.isfinite(frame["p"].to_numpy()) & (p_ref > 0)
    cap = system.pac0_kw
    full = np.minimum(p_ref, cap)
    shaded = np.minimum(p_ref * (1 - np.clip(depth_of_step, 0, 1)), cap)
    return np.where(ok, np.clip(beam, 0, 1) * (full - shaded), 0.0)


def _calibrate(frame, mask_of_step, cfg, system, coef=None):
    """Alternate daily scales and the efficiency curve until stable. Returns coef, days, p_cs."""
    p = frame["p"].to_numpy()
    high = (frame["elevation"].to_numpy() >= 10) & (frame["E"].to_numpy() >= 100)
    if coef is None:
        ok = high & np.isfinite(p) & (p > 0)
        coef = np.r_[np.nanquantile(p[ok] / frame["E"].to_numpy()[ok], 0.9), np.zeros(design(frame[:1]).shape[1] - 1)]
    for _ in range(4):
        p_cs = expected(frame, coef)
        usable = high & ~mask_of_step & (p_cs > 0) & ~(p >= cfg.clip * system.pac0_kw)
        days = day_table(frame, p_cs, usable, system, cfg.clear_level, cfg.clear_rough)
        scale = frame["day"].map(days["scale"]).to_numpy()
        clear = frame["day"].map(days["clear"]).fillna(False).to_numpy(bool)
        fit = (clear & ~mask_of_step & np.isfinite(p) & (frame["E"].to_numpy() >= E_MIN)
               & (frame["elevation"].to_numpy() >= cfg.min_elevation) & (scale * p_cs < cfg.clip * system.pac0_kw)
               & ~(p >= cfg.clip * system.pac0_kw))
        y = p[fit] / (scale[fit] * frame["E"].to_numpy()[fit])
        coef = huber_fit(design(frame[fit]), y)
    p_cs = expected(frame, coef)
    usable = high & ~mask_of_step & (p_cs > 0) & ~(p >= cfg.clip * system.pac0_kw)
    days = day_table(frame, p_cs, usable, system, cfg.clear_level, cfg.clear_rough)
    return coef, days, p_cs


def _residual_samples(frame, days, p_cs, cfg, system, az_edges, el_edges) -> tuple[pd.DataFrame, np.ndarray]:
    p = frame["p"].to_numpy()
    scale = frame["day"].map(days["scale"]).to_numpy()
    clear = frame["day"].map(days["clear"]).fillna(False).to_numpy(bool)
    p_exp = scale * p_cs
    with np.errstate(divide="ignore", invalid="ignore"):
        r = p / p_exp - 1
    use = (clear & np.isfinite(r) & (frame["elevation"].to_numpy() >= cfg.min_elevation)
           & (frame["E"].to_numpy() >= E_MIN) & (p_exp < cfg.clip * system.pac0_kw)
           & (p < cfg.clip * system.pac0_kw) & (p_exp > 0.01 * system.pdc0_kw)
           & (frame["aoi"].to_numpy() < cfg.max_aoi))
    i, j = cell_index(frame["azimuth"].to_numpy(), frame["elevation"].to_numpy(), az_edges, el_edges)
    samples = pd.DataFrame({"i": i[use], "j": j[use], "el": frame["elevation"].to_numpy()[use],
                            "az": frame["azimuth"].to_numpy()[use], "day": frame["day"].to_numpy()[use],
                            "r": np.clip(r[use], -1, 1)}, index=frame.index[use])
    return samples, np.where(use, r, np.nan)


def _mask_of_steps(frame, mask, az_edges, el_edges) -> np.ndarray:
    i, j = cell_index(frame["azimuth"].to_numpy(), frame["elevation"].to_numpy(), az_edges, el_edges)
    return mask[i, j] & (frame["elevation"].to_numpy() > 0)


def estimate_shift(p_ac: pd.Series, system: SystemConfig, clear_days, cfg: FingerprintConfig) -> float:
    """Clock offset [min] that minimises the median absolute clear-day residual (parabola through the best)."""
    sub = p_ac[p_ac.index.tz_convert(system.tz).normalize().tz_localize(None).isin(clear_days)]
    costs = []
    for s in cfg.shift_grid:
        f = prepare(sub, system, s)
        p = f["p"].to_numpy()
        ok = (np.isfinite(p) & (f["elevation"].to_numpy() >= 5) & (f["E"].to_numpy() >= 100)
              & (p > 0) & (p < cfg.clip * system.pac0_kw))
        f, p = f[ok], p[ok]
        c = huber_fit(design(f), p / f["E"].to_numpy())
        ratio = p / (f["E"].to_numpy() * (design(f) @ c))
        r = ratio / pd.Series(ratio).groupby(f["day"].to_numpy()).transform("median").to_numpy() - 1
        r = r[np.isfinite(r)]
        costs.append(np.median(np.abs(r)) if len(r) else np.inf)
    costs = np.array(costs)
    log.debug("clock offset cost: %s", dict(zip(cfg.shift_grid, np.round(costs, 5))))
    b = int(np.argmin(costs))
    grid = np.asarray(cfg.shift_grid)
    if 0 < b < len(grid) - 1:
        y0, y1, y2 = costs[b - 1:b + 2]
        denom = y0 - 2 * y1 + y2
        offset = 0.5 * (y0 - y2) / denom if denom > 0 else 0.0
        return float(grid[b] + offset * (grid[1] - grid[0]))
    return float(grid[b])


def fingerprint(p_ac: pd.Series, system: SystemConfig, cfg: FingerprintConfig = FingerprintConfig()) -> Fingerprint:
    """Run the four steps on a series of AC power [kW] with a tz-aware regular index."""
    az_edges, el_edges = _grid(cfg)
    shift = 0.0 if cfg.shift_minutes is None else cfg.shift_minutes
    frame = prepare(p_ac, system, shift)
    coef, days, _ = _calibrate(frame, np.zeros(len(frame), bool), cfg, system)
    if cfg.shift_minutes is None:
        shift = estimate_shift(p_ac, system, days.index[days["clear"]], cfg)
        log.info("clock offset: data are best explained with the sun %.1f min after the timestamp", shift)
        frame = prepare(p_ac, system, shift)

    mask = np.zeros((len(az_edges) - 1, len(el_edges) - 1), bool)
    region = mask
    history = []
    for it in range(cfg.iterations):
        on_mask = _mask_of_steps(frame, region, az_edges, el_edges)
        coef, days, p_cs = _calibrate(frame, on_mask, cfg, system, coef)
        samples, r = _residual_samples(frame, days, p_cs, cfg, system, az_edges, el_edges)
        cells = cell_stats(samples, az_edges, el_edges, cfg.pool, cfg.quantile)
        new = shade_mask(cells, cfg)
        history.append({"iteration": it + 1, "clear_days": int(days["clear"].sum()), "samples": len(samples),
                        "shaded_cells": int(new.sum()), "coef": np.round(coef, 5).tolist()})
        log.info("iteration %d: %d clear days, %d residual samples, %d shaded cells", it + 1,
                 days["clear"].sum(), len(samples), new.sum())
        if np.array_equal(new, mask):
            break
        mask = new
        region = shade_region(mask, silhouette(samples, mask, cells["n_days"] >= cfg.min_days, az_edges,
                                                          el_edges, cfg.depth, cfg.pool), el_edges)

    sil = silhouette(samples, mask, cells["n_days"] >= cfg.min_days, az_edges, el_edges, cfg.depth, cfg.pool)
    # loss: depth of each step's cell. Masked cells use their median residual; below the silhouette, observed
    # cells cut by the edge (median clearly negative, q75 not) use theirs too, and cells not observed on enough
    # clear days take the median depth of their column's masked cells
    depth_map = np.where(mask, -np.nan_to_num(cells["median"]), 0.0).clip(0, 1)
    observed = cells["n_days"] >= cfg.min_days
    el_mid = (el_edges[:-1] + el_edges[1:]) / 2
    for row in sil.itertuples():
        col, med = depth_map[row.i], np.nan_to_num(cells["median"][row.i])
        below = el_mid < (row.height if np.isfinite(row.height) else 0.0)
        edge = below & observed[row.i] & ~mask[row.i] & (med < -cfg.depth / 2)
        col[edge] = np.clip(-med[edge], 0, 1)
        col[below & ~observed[row.i]] = np.median(col[mask[row.i]]) if mask[row.i].any() else 0.0
    i, j = cell_index(frame["azimuth"].to_numpy(), frame["elevation"].to_numpy(), az_edges, el_edges)
    step_depth = np.where((frame["elevation"].to_numpy() > 0) & (frame["aoi"].to_numpy() < 90), depth_map[i, j], 0.0)
    clear_scale = days["scale"].where(days["clear"])
    scale_ref = clear_scale.rolling(31, center=True, min_periods=1).median().reindex(days.index)
    scale_ref = scale_ref.ffill().bfill().fillna(1.0)
    frame["p_exp_cs"] = p_cs
    frame["residual"] = r
    frame["clear"] = frame["day"].map(days["clear"]).fillna(False).to_numpy(bool)
    p_ref = frame["day"].map(scale_ref).to_numpy() * p_cs
    frame["beam"] = sky_beam(frame, relative_beam(frame, p_ref, cfg.k_cloud), step_depth > 0)
    frame["shade_loss"] = estimate_loss(frame, step_depth, frame["beam"].to_numpy(), p_ref, system)
    fp = Fingerprint(cfg, system, shift, coef, frame, days, az_edges, el_edges, cells, mask, sil, history)
    log.info("shade mask %d cells, silhouette over %d azimuth columns, estimated loss %.2f %%", mask.sum(),
             len(sil), 100 * fp.loss_fraction)
    return fp


def fit_row_profile(sil: pd.DataFrame, tilt: float) -> dict:
    """Test whether a silhouette is the row in front: fit tan(h) = tan(a_c) * cos(az - az_0), linear in
    (tan(a_c) cos az_0, tan(a_c) sin az_0). a_c is the critical profile angle below which the front row casts a
    shadow; with the module tilt it gives the ground coverage ratio GCR = tan(a_c) / (sin b + tan(a_c) cos b)."""
    s = sil[np.isfinite(sil["height"]) & ~sil["censored"]]
    if len(s) < 4:
        return {"points": len(s)}
    az = np.radians(s["azimuth"].to_numpy())
    X = np.column_stack([np.cos(az), np.sin(az)])
    (a, b), *_ = np.linalg.lstsq(X, np.tan(np.radians(s["height"].to_numpy())), rcond=None)
    t, az0 = np.hypot(a, b), np.degrees(np.arctan2(b, a)) % 360
    fitted = np.degrees(np.arctan(np.clip(X @ [a, b], 0, None)))
    beta = np.radians(tilt)
    return {"points": len(s), "critical_angle_deg": float(np.degrees(np.arctan(t))), "row_azimuth_deg": float(az0),
            "rmse_deg": float(np.sqrt(np.mean((fitted - s["height"].to_numpy()) ** 2))),
            "gcr": float(t / (np.sin(beta) + t * np.cos(beta))),
            "azimuth_span_deg": float(s["azimuth"].max() - s["azimuth"].min())}
