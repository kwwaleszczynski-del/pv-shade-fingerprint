"""Obstructions as seen from the array: a silhouette on the sky and how much beam light they take away.

Every obstruction answers two questions for a sun position (azimuth, elevation in degrees):

    top(az)              the silhouette: elevation of the obstruction's upper edge at that azimuth (NaN where the
                         obstruction is not), what a horizon profile in PVsyst describes
    beam_loss(az, el)    the fraction of the array's beam (direct) power lost when the sun stands there, 0..1

The beam loss is (share of the array in the shadow) x (opacity), with an electrical mismatch factor: a shadow on a
few cells of a module bypasses a whole substring, so a partial shadow costs more than its area. Diffuse light is
left untouched. Four shapes cover the usual cases on a site visit:

    Tree      elliptical crown with partial transparency on a narrow opaque trunk
    Building  a wall whose angular height falls off away from its normal (flat roof edge)
    Row       the row in front on a tilted rack: shaded strip grows as the sun's profile angle drops
    Pole      a thin opaque vertical object (lamp post, lightning rod, chimney)

Usage:
    from pv_shade.obstructions import SCENARIOS, scene_beam_loss, scene_top
    scene = SCENARIOS["tree"]
    loss = scene_beam_loss(scene, sun["azimuth"], sun["elevation"])
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _dist(az, center: float) -> np.ndarray:
    """Signed angular distance az - center in (-180, 180]."""
    return (np.asarray(az, float) - center + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class Tree:
    azimuth: float = 135.0
    half_width: float = 12.0
    crown_top: float = 22.0
    crown_base: float = 7.0
    trunk_half_width: float = 1.5
    transmittance: float = 0.4
    coverage: float = 0.6  # share of the array that falls into the crown's shadow
    mismatch: float = 1.3
    name: str = "tree"

    def top(self, az) -> np.ndarray:
        u = _dist(az, self.azimuth) / self.half_width
        mid, semi = (self.crown_top + self.crown_base) / 2, (self.crown_top - self.crown_base) / 2
        crown = np.where(np.abs(u) <= 1, mid + semi * np.sqrt(np.clip(1 - u**2, 0, 1)), np.nan)
        trunk = np.where(np.abs(_dist(az, self.azimuth)) <= self.trunk_half_width, self.crown_base, np.nan)
        return np.fmax(crown, trunk)

    def beam_loss(self, az, el) -> np.ndarray:
        d, el = _dist(az, self.azimuth), np.asarray(el, float)
        mid, semi = (self.crown_top + self.crown_base) / 2, (self.crown_top - self.crown_base) / 2
        in_crown = (d / self.half_width) ** 2 + ((el - mid) / semi) ** 2 <= 1
        in_trunk = (np.abs(d) <= self.trunk_half_width) & (el <= mid)
        share = min(1.0, self.coverage * self.mismatch)
        return np.where(in_trunk, share, np.where(in_crown, share * (1 - self.transmittance), 0.0))


@dataclass(frozen=True)
class Building:
    azimuth: float = 215.0  # direction of the wall's normal (where it looks tallest)
    half_width: float = 20.0
    height: float = 22.0  # angular height at the normal [deg]
    coverage: float = 1.0
    name: str = "building"

    def top(self, az) -> np.ndarray:
        d = _dist(az, self.azimuth)
        h = np.degrees(np.arctan(np.tan(np.radians(self.height)) * np.cos(np.radians(d))))
        return np.where(np.abs(d) <= self.half_width, h, np.nan)

    def beam_loss(self, az, el) -> np.ndarray:
        return np.where(np.asarray(el, float) < np.nan_to_num(self.top(az), nan=-90.0), self.coverage, 0.0)


@dataclass(frozen=True)
class Row:
    """Front row of a fixed-tilt rack. Shading starts when the sun's profile angle (elevation projected on the
    plane normal to the rows) drops below the critical angle set by tilt and ground coverage ratio."""

    azimuth: float = 180.0
    tilt: float = 30.0
    gcr: float = 0.35  # collector width / row pitch
    rows_behind: float = 0.9  # share of the array that has a row in front of it
    mismatch: float = 3.0  # portrait modules: a strip along the bottom edge cuts every substring
    name: str = "row"

    @property
    def critical_angle(self) -> float:
        b = np.radians(self.tilt)
        return float(np.degrees(np.arctan(self.gcr * np.sin(b) / (1 - self.gcr * np.cos(b)))))

    def _profile(self, az, el) -> np.ndarray:
        d = np.radians(_dist(az, self.azimuth))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.degrees(np.arctan2(np.tan(np.radians(np.asarray(el, float))), np.cos(d)))

    def top(self, az) -> np.ndarray:
        d = _dist(az, self.azimuth)
        h = np.degrees(np.arctan(np.tan(np.radians(self.critical_angle)) * np.cos(np.radians(d))))
        return np.where(np.abs(d) < 90, h, np.nan)

    def beam_loss(self, az, el) -> np.ndarray:
        pa = np.radians(self._profile(az, el))
        b = np.radians(self.tilt)
        with np.errstate(divide="ignore", invalid="ignore"):
            shaded = 1 - np.sin(pa) / (self.gcr * np.sin(pa + b))  # shaded fraction of the collector width
        shaded = np.where((pa > 0) & (np.abs(_dist(az, self.azimuth)) < 90), np.clip(shaded, 0, 1), 0.0)
        return self.rows_behind * np.minimum(1.0, self.mismatch * shaded)


@dataclass(frozen=True)
class Pole:
    azimuth: float = 160.0
    half_width: float = 1.5
    height: float = 30.0
    coverage: float = 0.08
    mismatch: float = 3.0
    name: str = "pole"

    def top(self, az) -> np.ndarray:
        return np.where(np.abs(_dist(az, self.azimuth)) <= self.half_width, self.height, np.nan)

    def beam_loss(self, az, el) -> np.ndarray:
        inside = (np.abs(_dist(az, self.azimuth)) <= self.half_width) & (np.asarray(el, float) < self.height)
        return np.where(inside, min(1.0, self.coverage * self.mismatch), 0.0)


Scene = tuple  # of obstructions


def scene_top(scene: Scene, az) -> np.ndarray:
    """Combined silhouette: the highest edge at each azimuth, NaN where there is no obstruction."""
    az = np.asarray(az, float)
    out = np.full(az.shape, np.nan)
    for ob in scene:
        out = np.fmax(out, ob.top(az))
    return out


def scene_beam_loss(scene: Scene, az, el) -> np.ndarray:
    """Combined beam loss of independent obstructions: 1 - prod(1 - loss_i)."""
    keep = np.ones(np.broadcast(np.asarray(az, float), np.asarray(el, float)).shape)
    for ob in scene:
        keep = keep * (1 - ob.beam_loss(az, el))
    return 1 - keep


SCENARIOS: dict[str, Scene] = {
    "none": (),
    "tree": (Tree(),),
    "building": (Building(),),
    "row": (Row(),),
    "row-dense": (Row(gcr=0.5, name="row-dense"),),
    "pole": (Pole(),),
    "building+pole": (Building(), Pole(azimuth=150.0, height=35.0)),
}
