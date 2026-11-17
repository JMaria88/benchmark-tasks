"""The adapter patches in yamkit.py, and the task file's layout and demo (no Isaac Sim)."""

from pathlib import Path

import numpy as np
import pytest
from yam_arena import demo, station, task_file

import yamkit

TASK = Path(__file__).resolve().parent / "screwdriver_task.py"


def test_a_tilted_grip_turns_the_fingers_level_along_its_heading():
    yaw = np.radians(30.0)
    level = yamkit.grip_rotation(yaw, -np.pi / 2)
    heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    assert level @ [0, 0, 1] == pytest.approx(heading, abs=1e-9)  # the approach axis
    assert level @ [1, 0, 0] == pytest.approx(
        yamkit.grip_rotation(yaw) @ [1, 0, 0], abs=1e-9
    )  # same closing axis
    stood = (
        level @ yamkit.grip_rotation(yaw).T
    )  # what lay along the heading, far end pointing back, now points down
    assert stood @ -heading == pytest.approx([0, 0, -1], abs=1e-9)
    assert yamkit.grip_rotation(0.4) == pytest.approx(
        demo.grip_rotation(0.4)
    )  # untilted: as the adapter's


def test_the_patched_planner_plays_plain_waypoints_as_before_and_tilted_ones_tilted():
    plain = [demo.Waypoint(1.0, (0.3, 0.2, 0.9), 0.0, "a", 0.5)]
    _, rot, _, _ = demo.ArmPlan(plain, station.BASES["left"]).target(1.5)
    assert rot == pytest.approx(demo.grip_rotation(0.5))
    tilted = [yamkit.TiltedWaypoint(1.0, (0.3, 0.2, 0.9), 0.0, "b", 0.5, -np.pi / 2)]
    _, rot, _, _ = demo.ArmPlan(tilted, station.BASES["left"]).target(1.5)
    assert rot == pytest.approx(yamkit.grip_rotation(0.5, -np.pi / 2))


def test_a_usd_body_passes_the_task_file_checks():
    body = yamkit.usd_body(
        "screw",
        "/x.usda",
        np.array([0.1, 0.2, 0.8]),
        rot_xyzw=(0, 0.70710678, 0, 0.70710678),
        mass=0.02,
    )
    body._check()
    assert (
        body.free
        and body.pos == (0.1, 0.2, 0.8)
        and all(type(v) is float for v in body.pos)
    )
    assert not yamkit.usd_body("insert", "/x.usda", (0, 0, 0.75)).free
    with pytest.raises(ValueError, match="unit quaternion"):
        yamkit.usd_body("bad", "/x.usda", (0, 0, 0), rot_xyzw=(0, 0, 0, 2))


def test_the_task_loads_and_its_demo_is_reachable():
    task = task_file.load(TASK)
    assert task.names == ("insert", "screw", "driver") and task.free_objects == (
        "screw",
        "driver",
    )
    assert all(Path(o.usd_path).is_file() for o in task.objects)
    for arm, waypoints in task.demo().items():
        plan = demo.ArmPlan(waypoints, station.BASES[arm])
        for k in range(round(plan.duration * 10) + 3):
            plan.action7((k + 1) / 10)
        assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1, arm


def test_a_screw_lying_on_the_table_scores_nothing():
    task = task_file.load(TASK)
    m = task.module

    def state(spec, rot):
        r = np.asarray(rot, float)
        return task_file.ObjectState(
            np.asarray(spec.pos, float),
            r,
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
        )

    objects = {o.name: state(o, o.rot_xyzw) for o in task.objects}
    assert task.score(objects) == 0.0 and not task.success(objects)
    assert m.distance(objects) > 0.1


def _upright(task, name, origin_z):
    """An ObjectState for the screw or driver standing upright on the insert's axis, its origin (tip) at origin_z."""
    m = task.module
    spec = {
        "screw": (m.SCREW_UP, m.SCREW.height),
        "driver": (m.DRIVER_UP, m.DRIVER.length),
    }[name]
    rot, length = spec
    quat = m.matrix_to_quat_xyzw(rot)
    centre = np.array(
        [*m.INSERT_XY, origin_z + length / 2]
    )  # the bounding-box centre, on the axis
    return task_file.ObjectState(centre, quat, np.zeros(3), np.zeros(3), centre, centre)


def test_full_score_needs_the_bit_in_the_socket():
    task = task_file.load(TASK)
    m = task.module
    tip = (
        m.INSERT_TOP - m.SCREW.length + 0.0003
    )  # seated: the head's underside 0.3 mm above the insert's top
    screw = _upright(task, "screw", tip)
    socket_floor = tip + m.SCREW.height - m.SCREW.socket_depth
    insert = next(o for o in task.objects if o.name == "insert")
    insert_state = task_file.ObjectState(
        np.asarray(insert.pos), np.array([0, 0, 0, 1.0]), *(np.zeros(3),) * 4
    )
    with_bit = {
        "insert": insert_state,
        "screw": screw,
        "driver": _upright(task, "driver", socket_floor + 0.0005),
    }
    assert (
        m.engaged(with_bit)
        and task.success(with_bit)
        and task.score(with_bit) == pytest.approx(1.0)
    )
    lying = next(o for o in task.objects if o.name == "driver")
    lying_state = task_file.ObjectState(
        np.asarray(lying.pos), np.asarray(lying.rot_xyzw), *(np.zeros(3),) * 4
    )
    by_hand = {
        **with_bit,
        "driver": lying_state,
    }  # seated, but the screwdriver still on the table
    assert (
        not m.engaged(by_hand)
        and not task.success(by_hand)
        and task.score(by_hand) == pytest.approx(0.2)
    )
