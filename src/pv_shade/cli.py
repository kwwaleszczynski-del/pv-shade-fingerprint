"""Command line interface.

Usage:
    python -m pv_shade validate                          # synthetic scenarios x seeds 1-5: metrics and charts
    python -m pv_shade validate --scenarios tree pole --seeds 101 102
    python -m pv_shade real --pvdaq-dir data/pvdaq       # NREL PVDAQ system 34 (daily CSVs, --fetch downloads 8 MB)
    python -m pv_shade compare                           # vs. the pv-anomaly-autoencoder model (needs torch)
    python -m pv_shade all
    python -m pv_shade validate --help

Simulations are cached in ./data/synthetic (override with --data-dir or PV_SHADE_DATA_DIR), results go to
./results and charts to ./docs/img.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pv_shade import plots, pvdaq
from pv_shade.fingerprint import FingerprintConfig, cell_index, fingerprint, fit_row_profile
from pv_shade.metrics import iou
from pv_shade.obstructions import SCENARIOS, scene_top
from pv_shade.validate import summarize, validate

log = logging.getLogger("pv_shade")

TEST_SEEDS = [1, 2, 3, 4, 5]


def _top(name: str):
    scene = SCENARIOS[name]
    return (lambda az: scene_top(scene, az)) if scene else None


def cmd_validate(args):
    t0 = time.time()
    cfg = FingerprintConfig()
    table, cases = validate(args.scenarios, args.seeds, args.data_dir / "synthetic", cfg, keep=True)
    summary = summarize(table)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(args.results_dir / "validation.csv", index=False)
    summary.to_csv(args.results_dir / "validation_summary.csv")
    log.info("validation summary over seeds %s:\n%s", args.seeds, summary.to_string())

    first = [c for c in cases if c.seed == args.seeds[0]]
    tops = {c.scenario: _top(c.scenario) for c in first}
    plots.scenario_grid(first, args.img_dir / "scenarios.png", tops)
    plots.loss_curves([c for c in first if c.scenario not in ("none", "row-dense")], args.img_dir / "loss_curves.png")
    plots.validation_bars(summary, args.img_dir / "validation.png")
    hero = next((c for c in first if c.scenario == "building+pole"), first[0])
    plots.time_vs_sun(hero.fp, tops.get(hero.scenario), args.img_dir / "time_vs_sun.png")
    fig = plots.sun_path_polar(hero.fp, truth_top=tops.get(hero.scenario),
                               title=f"Sun-path diagram: synthetic '{hero.scenario}' scene")
    plots._save(fig, args.img_dir / "sunpath_synthetic.png")
    rows = [{"scenario": c.scenario, "seed": c.seed, **fit_row_profile(c.fp.silhouette, c.fp.system.tilt)}
            for c in cases if c.scenario.startswith("row")]
    if rows:
        pd.DataFrame(rows).round(3).to_csv(args.results_dir / "row_fit.csv", index=False)
    (args.results_dir / "validation_run.json").write_text(json.dumps({
        "scenarios": args.scenarios, "seeds": args.seeds, "years": ["2021-01-01", "2023-01-01"],
        "config": {k: v for k, v in vars(cfg).items() if k != "shift_grid"},
        "runtime_seconds": round(time.time() - t0, 1)}, indent=2, default=str))
    log.info("results in %s, charts in %s (%.0f s)", args.results_dir, args.img_dir, time.time() - t0)


def cmd_real(args):
    t0 = time.time()
    system = pvdaq.SYSTEMS[args.system]
    src = args.pvdaq_dir / f"system_{args.system}"
    if args.fetch:
        files = pvdaq.fetch(args.system, args.start, args.end, args.pvdaq_dir)
    else:
        days = pd.date_range(args.start, args.end, freq="D")
        files = [src / f"system_{args.system}__date_{d:%Y_%m_%d}.csv" for d in days]
        missing = [f for f in files if not f.exists()]
        if missing:
            raise SystemExit(f"{len(missing)} of {len(files)} daily files missing in {src}; "
                             f"run with --fetch to download them (about 8 MB for two years)")
    df = pvdaq.load(files, system, args.system)
    out = args.results_dir / f"pvdaq_{args.system}"
    out.mkdir(parents=True, exist_ok=True)

    fp = fingerprint(df["p_ac"], system)
    years = sorted(set(df.index.tz_convert(system.tz).year))
    per_year = {y: fingerprint(df["p_ac"][df.index.tz_convert(system.tz).year == y], system) for y in years}
    row = fit_row_profile(fp.silhouette, system.tilt)

    # independent check with the plane-of-array pyranometer, which the method never saw: the ratio power / sensor
    # on clear days, normalised by its median with the sun above 20 deg, inside and outside the shade mask
    f = fp.frame
    k = df["p_ac"] / (df["poa"] * system.pdc0_kw / 1000)
    ok = (f["clear"] & (df["poa"] > 50) & (f["aoi"] < 80) & (f["elevation"] > 3) & np.isfinite(k)).to_numpy()
    high = ok & (f["elevation"] > 20).to_numpy()
    ref = k[high].groupby(f["day"][high]).median()
    ratio = (k / f["day"].map(ref).to_numpy() - 1)[ok]
    i, j = cell_index(f["azimuth"].to_numpy()[ok], f["elevation"].to_numpy()[ok], fp.az_edges, fp.el_edges)
    inside = fp.mask[i, j]
    low = f["elevation"].to_numpy()[ok] < 20
    sensor = {"power_over_sensor_in_mask": float(np.median(ratio[inside])),
              "power_over_sensor_outside_low_sun": float(np.median(ratio[~inside & low])),
              "power_over_model_in_mask": float(np.nanmedian(f["residual"].to_numpy()[ok][inside]))}

    a, b = per_year[years[0]], per_year[years[-1]]
    both = a.observed & b.observed
    sa, sb = a.silhouette.set_index("azimuth")["height"], b.silhouette.set_index("azimuth")["height"]
    common = sa.index.intersection(sb.index)
    summary = {
        "system": args.system, "name": system.name, "period": [args.start, args.end], "files": len(files),
        "missing_ac_share": round(float(df["p_ac"].isna().mean()), 4),
        "clock_offset_minutes": round(fp.shift_minutes, 2), "clear_days": int(fp.days["clear"].sum()),
        "shaded_cells": int(fp.mask.sum()), "silhouette_columns": int(np.isfinite(fp.silhouette["height"]).sum()),
        "estimated_loss_pct": round(100 * fp.loss_fraction, 3),
        "per_year": {str(y): {"clear_days": int(p.days["clear"].sum()), "shaded_cells": int(p.mask.sum()),
                              "estimated_loss_pct": round(100 * p.loss_fraction, 3),
                              "clock_offset_minutes": round(p.shift_minutes, 2)} for y, p in per_year.items()},
        "year_to_year": {"mask_iou": round(iou(a.mask & both, b.mask & both), 3),
                         "silhouette_rms_difference_deg": round(float(np.sqrt(np.mean((sa[common] - sb[common]) ** 2))),
                                                                2),
                         "common_columns": len(common)},
        "row_fit": {k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in row.items()},
        "sensor_check": {k_: round(v, 3) for k_, v in sensor.items()},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    fp.silhouette.round(2).to_csv(out / "silhouette.csv", index=False)
    pd.DataFrame({str(y): p.silhouette.set_index("azimuth")["height"] for y, p in per_year.items()}).round(2).to_csv(
        out / "silhouette_by_year.csv")
    ml = fp.monthly_loss()
    ml.index = ml.index.astype(str)
    ml.round(2).to_csv(out / "monthly_loss.csv")
    (out / "run.json").write_text(json.dumps(summary, indent=2, default=str))
    plots.real_overview(fp, per_year, row, system, args.img_dir / f"pvdaq{args.system}_sunpath.png")
    plots.real_years(per_year, row, args.img_dir / f"pvdaq{args.system}_years.png")
    log.info("PVDAQ %d: %s", args.system, json.dumps(summary, default=str))
    log.info("results in %s, charts in %s (%.0f s)", out, args.img_dir, time.time() - t0)


def cmd_compare(args):
    try:
        from pv_shade.compare import compare
    except ImportError as e:  # pragma: no cover
        raise SystemExit(f"compare needs torch and the pv-anomaly-autoencoder package ({e})")
    t0 = time.time()
    table, cases = compare(seed=args.seed, data_dir=args.data_dir / "synthetic")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(args.results_dir / "compare_autoencoder.csv", index=False)
    plots.compare_maps(cases, _top("building"), args.img_dir / "compare_autoencoder.png")
    log.info("comparison:\n%s\n(%.0f s)", table.round(3).to_string(), time.time() - t0)


def cmd_all(args):
    cmd_validate(args)
    cmd_real(args)
    cmd_compare(args)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pv-shade", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("validate", "real", "compare", "all"):
        p = sub.add_parser(name)
        p.add_argument("--data-dir", type=Path, default=Path(os.environ.get("PV_SHADE_DATA_DIR", "data")))
        p.add_argument("--results-dir", type=Path, default=Path("results"))
        p.add_argument("--img-dir", type=Path, default=Path("docs/img"))
        if name in ("validate", "all"):
            p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
            p.add_argument("--seeds", type=int, nargs="+", default=TEST_SEEDS,
                           help="simulation seeds (design choices were made on 101 and 102)")
        if name in ("real", "all"):
            p.add_argument("--system", type=int, default=34, choices=list(pvdaq.SYSTEMS))
            p.add_argument("--pvdaq-dir", type=Path, default=Path("data/pvdaq"),
                           help="folder holding system_<id>/system_<id>__date_YYYY_MM_DD.csv")
            p.add_argument("--start", default="2016-01-01")
            p.add_argument("--end", default="2017-12-31")
            p.add_argument("--fetch", action="store_true", help="download missing daily files from the OEDI bucket")
        if name in ("compare", "all"):
            p.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    {"validate": cmd_validate, "real": cmd_real, "compare": cmd_compare, "all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    main()
