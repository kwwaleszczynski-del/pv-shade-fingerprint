"""Scoring a fingerprint against the known obstructions of a simulated scene.

    truth_mask        cells whose centre is behind an obstruction that removes >= `material` of the beam
    iou               |estimated & true| / |estimated | true|, both restricted to cells observed on enough days
    horizon_rmse      RMSE [deg] of the reconstructed silhouette over the azimuth columns where the true edge is
                      inside the elevation range the sun showed (identifiable); censored columns are left out
    azimuth coverage  share of identifiable obstruction columns that got a height at all
    loss error        estimated minus true annual shade loss, in percentage points of the unshaded yield

Usage:
    from pv_shade.metrics import score
    row = score(fp, scene, truth_df)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pv_shade.fingerprint import Fingerprint
from pv_shade.obstructions import Scene, scene_beam_loss, scene_top

MATERIAL = 0.10  # beam loss that counts as shade for the true mask


def truth_mask(fp: Fingerprint, scene: Scene, material: float = MATERIAL) -> np.ndarray:
    az, el = np.meshgrid(fp.az_centers, fp.el_centers, indexing="ij")
    if not scene:
        return np.zeros(az.shape, bool)
    return scene_beam_loss(scene, az, el) >= material


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else np.nan


def observed_range(fp: Fingerprint) -> pd.DataFrame:
    """Lowest and highest clear-day sample elevation per azimuth column."""
    f = fp.frame[np.isfinite(fp.frame["residual"])]
    i = np.clip(np.searchsorted(fp.az_edges, f["azimuth"].to_numpy(), side="right") - 1, 0, len(fp.az_centers) - 1)
    g = pd.Series(f["elevation"].to_numpy()).groupby(i)
    return pd.DataFrame({"el_min": g.min(), "el_max": g.max()}).reindex(range(len(fp.az_centers)))


def horizon_table(fp: Fingerprint, scene: Scene) -> pd.DataFrame:
    """True and estimated silhouette per azimuth column, with the column's observable elevation range."""
    rng = observed_range(fp)
    t = pd.DataFrame({"azimuth": fp.az_centers, "true": scene_top(scene, fp.az_centers) if scene else np.nan,
                      "el_min": rng["el_min"].to_numpy(), "el_max": rng["el_max"].to_numpy()})
    est = fp.silhouette.set_index("i")
    t["estimated"] = est["height"].reindex(t.index).to_numpy(float)
    t["censored"] = est["censored"].reindex(t.index).fillna(False).astype(bool).to_numpy()
    t["identifiable"] = (t["true"] > t["el_min"] + 1) & (t["true"] < t["el_max"] - 1)
    return t


def score(fp: Fingerprint, scene: Scene, truth: pd.DataFrame) -> dict:
    obs = fp.observed
    tm = truth_mask(fp, scene) & obs
    em = fp.mask & obs
    h = horizon_table(fp, scene)
    ident = h[h["identifiable"]]
    hit = ident[np.isfinite(ident["estimated"]) & ~ident["censored"]]
    err = hit["estimated"] - hit["true"]
    lost = (truth["p_ac_unshaded"] - truth["p_ac_clean"]).sum()
    true_loss = lost / truth["p_ac_unshaded"].sum()
    est_loss = fp.loss_fraction
    false_cols = h[np.isfinite(h["estimated"]) & ~np.isfinite(h["true"])]
    return {
        "iou": iou(em, tm) if tm.any() else np.nan,
        "true_cells": int(tm.sum()), "estimated_cells": int(em.sum()),
        "false_cells": int((em & ~tm).sum()), "missed_cells": int((tm & ~em).sum()),
        "horizon_rmse_deg": float(np.sqrt(np.mean(err**2))) if len(err) else np.nan,
        "horizon_bias_deg": float(err.mean()) if len(err) else np.nan,
        "azimuth_coverage": len(hit) / len(ident) if len(ident) else np.nan,
        "false_columns": len(false_cols),
        "true_loss_pct": 100 * true_loss, "estimated_loss_pct": 100 * est_loss,
        "loss_error_pp": 100 * (est_loss - true_loss),
        "clear_days": int(fp.days["clear"].sum()), "shift_minutes": fp.shift_minutes,
    }
