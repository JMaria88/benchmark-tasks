"""The screwdriver task's parts: meshes and USD files (no Isaac Sim). Run: python -m pytest (in this folder)."""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

import partkit as threads
from partkit import Driver, Insert, Screw, Thread

SCREW, INSERT, DRIVER = Screw(), Insert(), Driver()


@pytest.mark.parametrize(
    "mesh",
    [
        threads.shank_mesh(SCREW),
        threads.head_mesh(SCREW),
        threads.insert_mesh(INSERT),
        threads.bit_mesh(DRIVER),
        threads.shaft_mesh(DRIVER),
        threads.handle_mesh(DRIVER),
    ],
    ids=["shank", "head", "insert", "bit", "shaft", "handle"],
)
def test_every_part_is_a_closed_outward_facing_mesh(mesh):
    m = trimesh.Trimesh(*mesh)
    assert m.is_watertight and m.is_winding_consistent and m.volume > 0


def test_the_profile_runs_root_to_crest_over_one_pitch():
    t = Thread()
    u = np.linspace(0, 1, 401)
    f = t.profile(u)
    assert f.min() == 0 and f.max() == 1 and f[0] == f[-1] == 0
    assert t.radius(np.array([0.0]), 0.0)[0] == pytest.approx(t.minor_r)
    with pytest.raises(ValueError, match="too deep"):
        Thread(depth=0.003).profile(u)


@pytest.mark.parametrize("yaw", [0.0, 1.0, -2.5])
def test_a_mated_screw_keeps_the_clearance_from_the_bore_everywhere(yaw):
    """At ``mated_z`` the shank's surface is exactly ``clearance`` inside the bore's at every angle and height."""
    t = INSERT.thread
    verts, _ = threads.shank_mesh(SCREW)
    rim = verts[np.hypot(verts[:, 0], verts[:, 1]) > 1e-9]  # drop the caps' centres
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = rim[:, 0] * c - rim[:, 1] * s, rim[:, 0] * s + rim[:, 1] * c
    z = rim[:, 2] + threads.mated_z(t, yaw, 1)
    inside = (z > 0) & (z < INSERT.height)
    theta, r = np.arctan2(y, x)[inside], np.hypot(x, y)[inside]
    bore = np.array(
        [
            t.radius(np.array([a]), h, t.clearance)[0]
            for a, h in zip(theta, z[inside], strict=True)
        ]
    )
    assert (bore - r).min() == pytest.approx(t.clearance, abs=1e-6)


def test_the_head_is_round_with_a_hex_socket_the_bit_fits():
    verts, _ = threads.head_mesh(SCREW)
    r = np.hypot(verts[:, 0], verts[:, 1])
    assert r.max() == pytest.approx(SCREW.head_d / 2)
    assert verts[:, 2].min() == pytest.approx(SCREW.length) and verts[
        :, 2
    ].max() == pytest.approx(SCREW.height)
    socket = verts[(r > 1e-9) & (r < SCREW.head_d / 2 - 1e-6)]
    below = socket[socket[:, 2] < SCREW.height - SCREW.socket_countersink + 1e-9]
    assert np.abs(below[:, 1]).max() == pytest.approx(
        SCREW.socket_flats / 2
    )  # flats face +-y
    assert socket[:, 2].min() == pytest.approx(SCREW.height - SCREW.socket_depth)
    mouth = socket[np.isclose(socket[:, 2], SCREW.height)]
    assert np.abs(mouth[:, 1]).max() == pytest.approx(
        SCREW.socket_flats / 2 + SCREW.socket_countersink
    )
    bit, _ = threads.bit_mesh(DRIVER)
    assert np.abs(bit[:, 1]).max() == pytest.approx(DRIVER.bit_flats / 2)
    assert DRIVER.bit_flats < SCREW.socket_flats
    assert np.abs(bit[:, 0]).max() == pytest.approx(
        DRIVER.bit_flats / math.sqrt(3)
    )  # corners on +-x, as the socket's


def test_the_handle_is_an_octagon_with_flats_facing_x_and_y():
    verts, _ = threads.handle_mesh(DRIVER)
    assert np.abs(verts[:, 0]).max() == pytest.approx(DRIVER.handle_flats / 2)
    assert np.abs(verts[:, 1]).max() == pytest.approx(DRIVER.handle_flats / 2)
    assert verts[:, 2].min() == pytest.approx(DRIVER.shaft_length) and verts[
        :, 2
    ].max() == pytest.approx(DRIVER.length)


def test_the_bore_widens_into_a_chamfer_at_its_mouth():
    verts, _ = threads.insert_mesh(INSERT)
    r = np.hypot(verts[:, 0], verts[:, 1])
    bore = (r > 1e-9) & (r < INSERT.outer_d / 2 - 1e-6)
    top = bore & np.isclose(verts[:, 2], INSERT.height)
    t = INSERT.thread
    assert r[top].min() == pytest.approx(
        t.major_d / 2 + t.clearance + INSERT.chamfer, abs=1e-6
    )
    with pytest.raises(ValueError, match="chamfer"):
        threads.insert_mesh(Insert(chamfer=0.01))


def test_turning_clockwise_seen_from_above_lowers_a_mated_screw():
    t = Thread()
    assert threads.mated_z(t, -math.pi, 2) == pytest.approx(1.5 * t.pitch)


def test_build_writes_sdf_colliders_once_without_importing_usd_here(tmp_path):
    code = (
        "import sys\n"
        "import partkit as threads\n"
        f"paths = threads.build(threads.Screw(), threads.Insert(), {str(tmp_path)!r}, driver=threads.Driver())\n"
        "print(paths['screw'], paths['insert'], paths['driver'])\n"
        "assert 'pxr' not in sys.modules\n"
    )
    here = Path(__file__).resolve().parent
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=here,
    ).stdout
    out = out.split()
    screw_path, insert_path, driver_path = map(Path, out)
    screw_text, insert_text, driver_text = (
        p.read_text() for p in (screw_path, insert_path, driver_path)
    )
    for text in (screw_text, insert_text):
        assert "PhysxSDFMeshCollisionAPI" in text and 'approximation = "sdf"' in text
    assert (
        'approximation = "convexHull"' in driver_text
        and "PhysxSDFMeshCollisionAPI" not in driver_text
    )
    assert (
        "physics:mass = 0.02" in screw_text
        and "physics:kinematicEnabled = 1" in insert_text
    )
    assert "physics:mass = 0.08" in driver_text
    before = screw_path.stat().st_mtime_ns
    paths = threads.build(Screw(), Insert(), tmp_path, driver=Driver())
    assert paths == {"screw": screw_path, "insert": insert_path, "driver": driver_path}
    assert screw_path.stat().st_mtime_ns == before
    assert (
        threads.build(Screw(), Insert(), tmp_path, friction=(0.2, 0.2))["screw"]
        != screw_path
    )


def test_the_centre_of_mass_of_a_box_is_its_middle():
    m = trimesh.creation.box(extents=(0.02, 0.04, 0.06))
    m.apply_translation((0.01, 0.0, 0.03))
    vol, com = threads.centre_of_mass((m.vertices, m.faces))
    assert vol == pytest.approx(0.02 * 0.04 * 0.06) and com == pytest.approx(
        [0.01, 0.0, 0.03], abs=1e-9
    )
    _, screw = threads.centre_of_mass(
        threads.shank_mesh(SCREW), threads.head_mesh(SCREW)
    )
    assert np.hypot(*threads.shaft_mesh(DRIVER)[0][:, :2].T).max() == pytest.approx(
        DRIVER.shaft_d / 2
    )
    assert 0 < screw[2] < SCREW.height and np.abs(screw[:2]).max() < 1e-5
