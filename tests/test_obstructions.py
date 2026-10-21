import numpy as np
import pytest

from pv_shade.obstructions import SCENARIOS, Building, Pole, Row, Tree, scene_beam_loss, scene_top


def test_row_critical_angle_matches_geometry():
    row = Row(tilt=30.0, gcr=0.5)
    b = np.radians(30.0)
    expected = np.degrees(np.arctan(0.5 * np.sin(b) / (1 - 0.5 * np.cos(b))))
    assert row.critical_angle == pytest.approx(expected)
    # no shade just above the critical profile angle, shade just below it (straight in front of the row)
    assert row.beam_loss(180.0, row.critical_angle + 0.2) == 0
    assert row.beam_loss(180.0, row.critical_angle - 2.0) > 0


def test_row_silhouette_drops_towards_the_side():
    row = Row()
    top = row.top(np.array([180.0, 135.0, 100.0, 300.0]))
    assert top[0] == pytest.approx(row.critical_angle)
    assert top[0] > top[1] > top[2] > 0
    assert np.isnan(top[3])  # behind the row


def test_building_is_opaque_below_its_edge_only():
    b = Building(azimuth=215.0, half_width=20.0, height=22.0)
    assert b.beam_loss(215.0, 21.0) == 1.0
    assert b.beam_loss(215.0, 23.0) == 0.0
    assert b.beam_loss(250.0, 5.0) == 0.0  # outside the wall
    assert np.isnan(b.top(250.0))


def test_tree_crown_is_partly_transparent_and_trunk_opaque():
    t = Tree(transmittance=0.4, coverage=0.5, mismatch=1.0)
    assert t.beam_loss(t.azimuth, (t.crown_top + t.crown_base) / 2 + 3) == pytest.approx(0.5 * 0.6)
    assert t.beam_loss(t.azimuth, 1.0) == pytest.approx(0.5)  # trunk
    assert t.beam_loss(t.azimuth + t.half_width - 0.5, 1.0) == 0.0  # sky below the edge of the crown
    assert t.top(t.azimuth) == pytest.approx(t.crown_top)


def test_pole_is_narrow():
    p = Pole(azimuth=160.0, half_width=1.5, height=30.0, coverage=0.08, mismatch=3.0)
    assert p.beam_loss(160.0, 10.0) == pytest.approx(0.24)
    assert p.beam_loss(162.0, 10.0) == 0.0


def test_scene_combines_independent_losses():
    scene = SCENARIOS["building+pole"]
    b, p = scene
    az, el = p.azimuth, 10.0
    combined = scene_beam_loss(scene, az, el)
    assert combined == pytest.approx(1 - (1 - b.beam_loss(az, el)) * (1 - p.beam_loss(az, el)))
    assert np.isnan(scene_top((), 180.0))
    assert scene_top(scene, p.azimuth) == pytest.approx(p.height)
