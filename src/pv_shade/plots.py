"""Charts: sun-path diagrams with the residual map, silhouettes, the time-vs-sun-space view and loss curves.

Colours: residuals use a diverging red (power shortfall) - grey (as expected) - blue (surplus) scale, magnitudes a
single blue ramp, series a fixed categorical order. The true silhouette is always the dashed dark line, the
reconstruction the solid orange one.

Usage:
    from pv_shade import plots
    plots.sun_path_polar(fp, path=Path("docs/img/sunpath.png"), truth_top=scene_top_fn)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from pv_shade.fingerprint import Fingerprint  # noqa: E402
from pv_shade.site import sun_position  # noqa: E402

INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                           "#008300", "#6250d6", "#e34948")
RESID = LinearSegmentedColormap.from_list("resid", ["#9c2a2a", RED, "#f2a7a0", "#f0efec", "#9ec5f4", BLUE, "#184f95"])
RESID.set_bad(SURFACE)
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"])
VLIM = 0.5
NL = "\n"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2,
})


def _save(fig, path: Path | None):
    if path is None:
        return fig
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def sun_paths(fp: Fingerprint, year: int = 2021) -> dict[str, pd.DataFrame]:
    """Sun position on the solstices and the equinox (5-minute steps, sun above the horizon)."""
    out = {}
    for label, day in (("21 Jun", f"{year}-06-21"), ("21 Mar / 23 Sep", f"{year}-03-20"), ("21 Dec", f"{year}-12-21")):
        t = pd.date_range(day, periods=288, freq="5min", tz=fp.system.tz)
        s = sun_position(t, fp.system)
        out[label] = s[s["elevation"] > 0]
    return out


def hour_marks(fp: Fingerprint, year: int = 2021) -> pd.DataFrame:
    """Sun position at full local solar-ish hours over a year (for the hour lines of the diagram)."""
    t = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="7D", tz=fp.system.tz)
    rows = []
    for h in range(4, 21):
        s = sun_position(t + pd.Timedelta(hours=h), fp.system)
        rows.append(s[s["elevation"] > 0].assign(hour=h))
    return pd.concat(rows)


def _cell_image(fp: Fingerprint, value: str) -> np.ndarray:
    img = np.where(fp.observed, fp.cells[value], np.nan)
    return np.ma.masked_invalid(img)


def row_top(row: dict):
    """Silhouette of a fitted front row (pv_shade.fingerprint.fit_row_profile) as a function of azimuth."""
    t, az0 = np.tan(np.radians(row["critical_angle_deg"])), row["row_azimuth_deg"]
    return lambda az: np.where(np.cos(np.radians(np.asarray(az) - az0)) > 0,
                               np.degrees(np.arctan(t * np.cos(np.radians(np.asarray(az) - az0)))), np.nan)


def _silhouette_lines(ax, fp: Fingerprint, truth_top, polar: bool, model_top=None, model_label: str = ""):
    conv = (lambda az, el: (np.radians(az), 90 - el)) if polar else (lambda az, el: (az, el))
    if model_top is not None:
        az = np.arange(0, 360.01, 0.25)
        ax.plot(*conv(az, model_top(az)), color=VIOLET, lw=1.4, ls=(0, (1, 1.5)), label=model_label, zorder=5)
    if truth_top is not None:
        az = np.arange(0, 360.01, 0.25)
        top = truth_top(az)
        ax.plot(*conv(az, top), color=INK, lw=1.4, ls=(0, (4, 2)), label="true silhouette", zorder=8)
    s = fp.silhouette.dropna(subset=["height"])
    if len(s):
        h = s["height"].to_numpy()
        az = s["azimuth"].to_numpy()
        # draw contiguous runs of azimuth columns as lines, isolated columns as markers
        breaks = np.flatnonzero(np.diff(az) > fp.config.az_step * 1.5) + 1
        for k, (a, b) in enumerate(zip(np.r_[0, breaks], np.r_[breaks, len(az)])):
            ax.plot(*conv(az[a:b], h[a:b]), color=ORANGE, lw=2, marker="o", ms=3,
                    label="reconstructed" if k == 0 else None, zorder=6)
        cens = s[s["censored"]]
        if len(cens):
            ax.plot(*conv(cens["azimuth"].to_numpy(), cens["height"].to_numpy()), ls="none", marker="^", ms=6,
                    mfc="none", mec=ORANGE, mew=1.2, label="at least this tall (censored)", zorder=7)


def sun_map(fp: Fingerprint, ax=None, value: str = "median", truth_top=None, title: str | None = None,
            az_range=(40, 320), el_max: float = 65, colorbar: bool = True, legend: bool = True, model_top=None,
            model_label: str = ""):
    """Azimuth x elevation view ('unrolled fisheye'): residual map, mask outline, sun paths, silhouettes."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3.8))
    img = _cell_image(fp, value)
    m = ax.pcolormesh(fp.az_edges, fp.el_edges, img.T, cmap=RESID, vmin=-VLIM, vmax=VLIM, shading="flat",
                      rasterized=True)
    if fp.mask.any():
        ax.contour(fp.az_centers, fp.el_centers, fp.mask.T.astype(float), levels=[0.5], colors=[INK2],
                   linewidths=0.8)
    for label, s in sun_paths(fp).items():
        ax.plot(s["azimuth"], s["elevation"], color=MUTED, lw=0.8, zorder=3)
        k = int(np.argmax(s["elevation"].to_numpy()))
        if s["elevation"].iloc[k] + 3 < el_max:
            ax.text(s["azimuth"].iloc[k], s["elevation"].iloc[k] + 1.2, label, color=INK2, fontsize=7, ha="center")
    _silhouette_lines(ax, fp, truth_top, polar=False, model_top=model_top, model_label=model_label)
    ticks = [(t, n) for t, n in zip(range(45, 316, 45), ["NE", "E", "SE", "S", "SW", "W", "NW"])
             if az_range[0] <= t <= az_range[1]]
    ax.set_xticks([t for t, _ in ticks])
    ax.set_xticklabels([f"{n} {t}" for t, n in ticks])
    ax.set_xlim(*az_range)
    ax.set_ylim(0, el_max)
    ax.set_ylabel("sun elevation [deg]")
    ax.set_xlabel("sun azimuth [deg]")
    ax.grid(color=GRID, lw=0.5)
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(loc="upper left", ncol=1)
    if colorbar:
        cb = plt.colorbar(m, ax=ax, pad=0.01, fraction=0.035)
        cb.set_label("clear-sky residual P/P_exp - 1", color=INK2)
        cb.outline.set_visible(False)
    return fig if fig is not None else m


def sun_path_polar(fp: Fingerprint, ax=None, value: str = "median", truth_top=None, title: str | None = None,
                   hours: bool = True, colorbar: bool = True, legend: bool = True, model_top=None,
                   model_label: str = ""):
    """Classic sun-path diagram (north up, clockwise, centre = zenith) with the residual map behind it."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 6.2), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    img = _cell_image(fp, value)
    th, r = np.radians(fp.az_edges), 90 - fp.el_edges
    m = ax.pcolormesh(th, r, img.T, cmap=RESID, vmin=-VLIM, vmax=VLIM, shading="flat", rasterized=True)
    if fp.mask.any():
        ax.contour(np.radians(fp.az_centers), 90 - fp.el_centers, fp.mask.T.astype(float), levels=[0.5],
                   colors=[INK2], linewidths=0.8)
    for label, s in sun_paths(fp).items():
        ax.plot(np.radians(s["azimuth"]), 90 - s["elevation"], color=MUTED, lw=0.9, zorder=3)
    if hours:
        hm = hour_marks(fp)
        for h, g in hm.groupby("hour"):
            ax.plot(np.radians(g["azimuth"]), 90 - g["elevation"], color=GRID, lw=0.7, zorder=3)
            top = g.loc[g["elevation"].idxmax()]
            ax.text(np.radians(top["azimuth"]), 90 - top["elevation"] - 3, f"{h}h", color=INK2, fontsize=6.5,
                    ha="center", va="center")
    _silhouette_lines(ax, fp, truth_top, polar=True, model_top=model_top, model_label=model_label)
    ax.set_ylim(0, 90)
    ax.set_yticks([30, 60, 90])
    ax.set_yticklabels(["60", "30", "0"], color=MUTED, fontsize=7)
    ax.set_xticks(np.radians(np.arange(0, 360, 45)))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    ax.grid(color=GRID, lw=0.5)
    ax.spines["polar"].set_color(MUTED)
    if title:
        ax.set_title(title, pad=14)
    if legend:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    if colorbar:
        cb = plt.colorbar(m, ax=ax, pad=0.08, fraction=0.04, shrink=0.7)
        cb.set_label("clear-sky residual P/P_exp - 1", color=INK2)
        cb.outline.set_visible(False)
    return fig if fig is not None else m


def time_vs_sun(fp: Fingerprint, truth_top, path: Path | None = None, title: str = ""):
    """The same residuals twice: against date and time of day (shade wanders), and against the sun's position
    (shade stands still)."""
    f = fp.frame[np.isfinite(fp.frame["residual"])]
    local = f.index.tz_convert(fp.system.tz)
    fig = plt.figure(figsize=(12, 4.4))
    ax1 = fig.add_axes([0.05, 0.12, 0.36, 0.76])
    ax2 = fig.add_axes([0.47, 0.12, 0.47, 0.76])
    hour = local.hour + local.minute / 60
    ax1.scatter(local.dayofyear, hour, c=f["residual"].clip(-VLIM, VLIM), cmap=RESID, vmin=-VLIM, vmax=VLIM, s=4,
                lw=0, rasterized=True)
    ax1.set_xlabel("day of year (clear days only)")
    ax1.set_ylabel("local standard time [h]")
    ax1.set_ylim(4, 21)
    ax1.set_xlim(0, 366)
    ax1.grid(color=GRID, lw=0.5)
    ax1.set_title("In time: a fixed obstruction draws wandering bands")
    m = sun_map(fp, ax=ax2, truth_top=truth_top, colorbar=False, legend=True,
                title="In sun space: it becomes a silhouette")
    cb = fig.colorbar(m, ax=ax2, pad=0.01, fraction=0.03)
    cb.set_label("P/P_exp - 1", color=INK2)
    cb.outline.set_visible(False)
    if title:
        fig.suptitle(title, x=0.05, ha="left", fontweight="bold")
    return _save(fig, path)


def scenario_grid(cases, path: Path | None = None, truth_tops=None):
    """One sun-space panel per scenario: residual map, true and reconstructed silhouette, metrics in the title."""
    n = len(cases)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.9 * n), squeeze=False)
    for ax, case in zip(axes[:, 0], cases):
        m = case.metrics
        txt = (f"{case.scenario}: IoU {m['iou']:.2f}, horizon RMSE {m['horizon_rmse_deg']:.1f} deg, "
               f"loss {m['estimated_loss_pct']:.2f} % (true {m['true_loss_pct']:.2f} %)")
        if case.scenario == "none":
            txt = f"none (control): {m['false_cells']} false shaded cells, loss {m['estimated_loss_pct']:.2f} %"
        mesh = sun_map(case.fp, ax=ax, truth_top=(truth_tops or {}).get(case.scenario), title=txt,
                       colorbar=False, legend=ax is axes[0, 0], el_max=62)
        if ax is not axes[-1, 0]:
            ax.set_xlabel("")
    cb = fig.colorbar(mesh, ax=axes[:, 0], pad=0.01, fraction=0.02, shrink=0.5)
    cb.set_label("clear-sky residual P/P_exp - 1", color=INK2)
    cb.outline.set_visible(False)
    return _save(fig, path)


def loss_curves(cases, path: Path | None = None):
    """Monthly shade loss, estimated vs true, one small panel per shaded scenario."""
    cases = [c for c in cases if c.scenario != "none"]
    fig, axes = plt.subplots(1, len(cases), figsize=(2.6 * len(cases), 2.9), sharey=False, squeeze=False)
    for ax, c in zip(axes[0], cases):
        est = c.fp.monthly_loss()
        truth = c.df.assign(lost=(c.df["p_ac_unshaded"] - c.df["p_ac_clean"]) * 0.25)
        month = truth.index.tz_convert(c.fp.system.tz).tz_localize(None).to_period("M")
        true_m = truth.groupby(month)["lost"].sum()
        moy_est = est["loss_kwh"].groupby(est.index.month).mean()
        moy_true = true_m.groupby(true_m.index.month).mean()
        x = np.arange(1, 13)
        ax.bar(x - 0.2, moy_true.reindex(x).to_numpy(), 0.38, color=INK2, label="true", edgecolor=SURFACE, lw=1)
        ax.bar(x + 0.2, moy_est.reindex(x).to_numpy(), 0.38, color=ORANGE, label="estimated", edgecolor=SURFACE,
               lw=1)
        ax.set_xticks([1, 3, 5, 7, 9, 11])
        ax.set_xticklabels(["Jan", "Mar", "May", "Jul", "Sep", "Nov"])
        ax.set_title(c.scenario, fontsize=10)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("shade loss [kWh / month]")
    axes[0, 0].legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


def validation_bars(summary: pd.DataFrame, path: Path | None = None):
    """Headline metrics per scenario with seed-to-seed spread."""
    s = summary.drop(index="none", errors="ignore")
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9))
    specs = [("iou", "mask IoU (1 = perfect)", BLUE, (0, 1)),
             ("horizon_rmse_deg", "silhouette RMSE [deg]", BLUE, None),
             ("loss_error_pp", "loss error [pp of annual yield]", BLUE, None)]
    y = np.arange(len(s))
    for ax, (col, label, color, lim) in zip(axes, specs):
        ax.barh(y, s[col], xerr=s[col + "_sd"], color=color, height=0.55, error_kw={"ecolor": INK2, "lw": 1})
        ax.set_yticks(y)
        ax.set_yticklabels(s.index if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_title(label, fontsize=10)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        if lim:
            ax.set_xlim(*lim)
        for yi, v in zip(y, s[col]):
            ax.text(v, yi - 0.36, f" {v:.2f} " if v >= 0 else f"{v:.2f} ", color=INK, fontsize=8, va="bottom",
                    ha="left" if v >= 0 else "right")
    fig.tight_layout()
    return _save(fig, path)


def real_overview(fp: Fingerprint, per_year: dict, row: dict, system, path: Path | None = None):
    """Real system: sun-path diagram over the whole period with the reconstructed silhouette and the row fit."""
    label = (f"front-row fit: critical angle {row['critical_angle_deg']:.1f} deg, GCR {row['gcr']:.2f}"
             if "gcr" in row else "")
    fig = sun_path_polar(fp, model_top=row_top(row) if "gcr" in row else None, model_label=label,
                         title=f"{system.name}: clear-sky residuals on the sun-path diagram")
    return _save(fig, path)


def real_years(per_year: dict, row: dict, path: Path | None = None):
    """Each year fingerprinted on its own: the same silhouette twice is the reproducibility check."""
    years = list(per_year)
    fig, axes = plt.subplots(len(years), 1, figsize=(9.5, 2.9 * len(years)), squeeze=False)
    top = row_top(row) if "gcr" in row else None
    for ax, y in zip(axes[:, 0], years):
        fp = per_year[y]
        mesh = sun_map(fp, ax=ax, title=f"{y} alone: {fp.days['clear'].sum()} clear days, "
                       f"{fp.mask.sum()} shaded cells, estimated loss {100 * fp.loss_fraction:.2f} %",
                       az_range=(80, 280), el_max=40, colorbar=False, legend=ax is axes[0, 0], model_top=top,
                       model_label="front-row fit (both years)")
        if ax is not axes[-1, 0]:
            ax.set_xlabel("")
    cb = fig.colorbar(mesh, ax=axes[:, 0], pad=0.01, fraction=0.025)
    cb.set_label("clear-sky residual P/P_exp - 1", color=INK2)
    cb.outline.set_visible(False)
    return _save(fig, path)


def compare_maps(cases, truth_top, path: Path | None = None):
    """Per situation: the fingerprint's residual map and the autoencoder's alarm rate on the same sun grid."""
    rate_cmap = LinearSegmentedColormap.from_list("rate", ["#f0efec", "#86b6ef", "#2a78d6", "#0d366b"])
    rate_cmap.set_bad(SURFACE)
    fig, axes = plt.subplots(len(cases), 2, figsize=(13, 3.3 * len(cases)), squeeze=False)
    names = {"always": "building there all along (also in the training year)",
             "appears": "building put up at the start of the scored year"}
    for (ax1, ax2), c in zip(axes, cases):
        m = c.metrics
        sun_map(c.fp, ax=ax1, truth_top=truth_top, az_range=(90, 270), el_max=45, colorbar=False,
                legend=ax1 is axes[0, 0], title=f"Fingerprint - {names.get(c.situation, c.situation)}{NL}"
                f"IoU {m['fp_iou']:.2f}, loss {m['fp_estimated_loss_pct']:.2f} % (true {m['fp_true_loss_pct']:.2f} %)")
        img = np.ma.masked_invalid(np.where(c.fp.observed, c.ae_rate, np.nan))
        mesh = ax2.pcolormesh(c.fp.az_edges, c.fp.el_edges, img.T, cmap=rate_cmap, vmin=0, vmax=1,
                              shading="flat", rasterized=True)
        az = np.arange(0, 360.01, 0.25)
        ax2.plot(az, truth_top(az), color=INK, lw=1.6, ls=(0, (4, 2)))
        for label, s in sun_paths(c.fp).items():
            ax2.plot(s["azimuth"], s["elevation"], color=MUTED, lw=0.8)
        ax2.set_xlim(90, 270)
        ax2.set_ylim(0, 45)
        ax2.set_xticks(np.arange(90, 271, 45))
        ax2.set_xticklabels(["E 90", "SE 135", "S 180", "SW 225", "W 270"])
        ax2.grid(color=GRID, lw=0.5)
        ax2.set_title(f"Autoencoder alarm rate on clear days{NL}"
                      f"flags {100 * m['ae_shaded_steps_flagged']:.0f} % of "
                      f"shaded steps, sun-space IoU {m['ae_sun_space_iou']:.2f}")
        cb = fig.colorbar(mesh, ax=ax2, pad=0.01, fraction=0.035)
        cb.set_label("share of steps alarmed", color=INK2)
        cb.outline.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("sun azimuth [deg]")
    axes[0, 1].set_ylabel("sun elevation [deg]")
    fig.tight_layout()
    return _save(fig, path)
