"""PV system definition, sun position and clear-sky irradiance in the module plane.

Everything the shade method knows about a system is in `SystemConfig`: location, orientation and nameplate.
`clear_sky_plane` gives the sun's position and what a cloudless sky would deliver to the modules, split into beam
and diffuse, because only the beam can be blocked by an obstruction.

Usage:
    from pv_shade.site import SystemConfig, clear_sky_plane
    system = SystemConfig()
    sky = clear_sky_plane(index, system)          # elevation, azimuth, aoi, poa_beam, poa_diffuse, ...
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pvlib


@dataclass(frozen=True)
class SystemConfig:
    """A fixed-tilt PV system. The default is the synthetic test system: 100 kWp on a flat roof in central Europe,
    tilted 30 deg towards south, with an 85 kW inverter (DC/AC 1.18)."""

    name: str = "synthetic"
    latitude: float = 52.2
    longitude: float = 21.0
    altitude: float = 100.0
    tz: str = "Etc/GMT-1"  # local standard time, no daylight saving
    tilt: float = 30.0
    azimuth: float = 180.0
    pdc0_kw: float = 100.0
    pac0_kw: float = 85.0

    @property
    def location(self) -> pvlib.location.Location:
        return pvlib.location.Location(self.latitude, self.longitude, altitude=self.altitude)


def sun_position(index: pd.DatetimeIndex, system: SystemConfig) -> pd.DataFrame:
    """Apparent elevation, azimuth and zenith [deg] and the angle of incidence on the modules."""
    sp = system.location.get_solarposition(index)
    aoi = pvlib.irradiance.aoi(system.tilt, system.azimuth, sp["apparent_zenith"], sp["azimuth"])
    return pd.DataFrame({"elevation": sp["apparent_elevation"], "azimuth": sp["azimuth"],
                         "zenith": sp["apparent_zenith"], "aoi": aoi}, index=index)


def plane_irradiance(sun: pd.DataFrame, dni, ghi, dhi, system: SystemConfig, model: str = "perez",
                     perez_set: str = "allsitescomposite1990") -> pd.DataFrame:
    """Beam and diffuse (sky + ground) irradiance in the module plane [W/m2] for given horizontal components."""
    dni_extra = pvlib.irradiance.get_extra_radiation(sun.index)
    airmass = pvlib.atmosphere.get_relative_airmass(sun["zenith"])
    poa = pvlib.irradiance.get_total_irradiance(system.tilt, system.azimuth, sun["zenith"], sun["azimuth"],
                                                dni, ghi, dhi, dni_extra=dni_extra, airmass=airmass, albedo=0.2,
                                                model=model, model_perez=perez_set)
    return pd.DataFrame({"poa_beam": poa["poa_direct"].fillna(0.0).clip(lower=0),
                         "poa_diffuse": poa["poa_diffuse"].fillna(0.0).clip(lower=0)}, index=sun.index)


def clear_sky_plane(index: pd.DatetimeIndex, system: SystemConfig, linke: float | None = None) -> pd.DataFrame:
    """Sun position plus clear-sky irradiance (Ineichen with the Linke turbidity climatology bundled with pvlib,
    or a fixed `linke`), horizontal and in the module plane (Perez transposition, all-sites coefficients)."""
    sun = sun_position(index, system)
    loc = system.location
    if linke is None:
        cs = loc.get_clearsky(index, model="ineichen")
    else:
        cs = loc.get_clearsky(index, model="ineichen", linke_turbidity=linke)
    poa = plane_irradiance(sun, cs["dni"], cs["ghi"], cs["dhi"], system)
    out = pd.concat([sun, cs.rename(columns=lambda c: f"{c}_cs"), poa], axis=1)
    night = out["elevation"] <= 0
    out.loc[night, ["poa_beam", "poa_diffuse"]] = 0.0
    return out


def solar_hour(index: pd.DatetimeIndex, longitude: float) -> np.ndarray:
    """Local apparent solar time in hours (12 = solar noon)."""
    utc = index.tz_convert("UTC")
    hours = utc.hour + utc.minute / 60 + utc.second / 3600
    eot = pvlib.solarposition.equation_of_time_spencer71(utc.dayofyear)  # minutes
    return np.mod(np.asarray(hours) + longitude / 15 + np.asarray(eot) / 60, 24)
