"""Synthetic validation: simulate each obstruction scenario, fingerprint it from power alone, score it.

Design choices (grid, thresholds, clear-day filter) were made on development seeds 101 and 102; the reported
numbers come from seeds 1-5, run once with those settings.

Usage:
    from pv_shade.validate import run_case, validate
    case = run_case("tree", seed=1, data_dir=Path("data/synthetic"))
    table = validate(["tree", "building"], seeds=[1, 2], data_dir=Path("data/synthetic"))
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pv_shade.fingerprint import Fingerprint, FingerprintConfig, fingerprint
from pv_shade.metrics import score
from pv_shade.obstructions import SCENARIOS
from pv_shade.simulate import simulate
from pv_shade.site import SystemConfig

log = logging.getLogger(__name__)


@dataclass
class Case:
    scenario: str
    seed: int
    df: pd.DataFrame
    events: pd.DataFrame
    fp: Fingerprint
    metrics: dict


def simulated(scenario: str, seed: int, data_dir: Path | None, start: str = "2021-01-01", end: str = "2023-01-01",
              refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a scenario, cached as Parquet in data_dir (None = no cache). The file name carries a hash of the
    scene and system definitions, so a changed obstruction never reuses an old simulation."""
    key = hashlib.sha1(repr((SCENARIOS[scenario], SystemConfig())).encode()).hexdigest()[:8]
    path = None if data_dir is None else Path(data_dir) / f"{scenario}_{seed}_{start}_{end}_{key}.parquet"
    if path is not None and path.exists() and not refresh:
        events = pd.read_parquet(path.with_suffix(".events.parquet"))
        return pd.read_parquet(path), events
    df, events = simulate(SCENARIOS[scenario], SystemConfig(), start=start, end=end, seed=seed)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        events.to_parquet(path.with_suffix(".events.parquet"))
    return df, events


def run_case(scenario: str, seed: int, data_dir: Path | None = None, cfg: FingerprintConfig = FingerprintConfig(),
             **sim) -> Case:
    t0 = time.time()
    df, events = simulated(scenario, seed, data_dir, **sim)
    fp = fingerprint(df["p_ac"], SystemConfig(), cfg)
    m = {"scenario": scenario, "seed": seed, **score(fp, SCENARIOS[scenario], df),
         "seconds": round(time.time() - t0, 1)}
    log.info("%-14s seed %3d: IoU %.2f, horizon RMSE %.1f deg, loss %.2f %% (true %.2f %%), %d false cells",
             scenario, seed, m["iou"], m["horizon_rmse_deg"], m["estimated_loss_pct"], m["true_loss_pct"],
             m["false_cells"])
    return Case(scenario, seed, df, events, fp, m)


def validate(scenarios, seeds, data_dir: Path | None = None, cfg: FingerprintConfig = FingerprintConfig(),
             keep: bool = False) -> tuple[pd.DataFrame, list[Case]]:
    """Metrics table over scenarios x seeds (and the cases themselves when `keep`)."""
    rows, cases = [], []
    for sc in scenarios:
        for seed in seeds:
            case = run_case(sc, seed, data_dir, cfg)
            rows.append(case.metrics)
            if keep:
                cases.append(case)
    return pd.DataFrame(rows), cases


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation per scenario of the headline metrics."""
    cols = ["iou", "horizon_rmse_deg", "azimuth_coverage", "true_loss_pct", "estimated_loss_pct", "loss_error_pp",
            "false_cells"]
    g = table.groupby("scenario", sort=False)[cols]
    out = g.mean().round(3).join(g.std().round(3), rsuffix="_sd")
    return out[[c for pair in zip(cols, [c + "_sd" for c in cols]) for c in pair]]
