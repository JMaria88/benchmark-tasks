"""What the screwdriver task needs beyond the ``yam_arena`` task-file format, added at runtime: the adapter's files are
not changed (tested with yam-isaac-lab-arena 0059d02). Importing this module installs it.

- :class:`UsdBody`: a rigid body from a USD file (here, the SDF-collided screw, insert and screwdriver from
  ``partkit.py``) at a full pose, so a part can start lying on its side. Free with a ``mass``; without one, kinematic
  where it is put. Spawned by wrapping ``yam_arena.scene.task_object`` once that module is imported (after Isaac Sim
  starts; an import hook waits for it).
- :class:`TiltedWaypoint` and :func:`grip_rotation`: a scripted grip tipped about its fingers' closing axis as well as
  turned about the vertical, so a hand can turn level and stand up what it picked up lying down. ``ArmPlan.target``
  is replaced by the same code reading a waypoint's ``tilt`` (0 for plain ``Waypoint``\\ s, which play as before).
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass

import numpy as np
from yam_arena import demo, ik
from yam_arena.task_file import Asset
from yam_arena.urdf import axis_angle_matrix, quat_xyzw_to_matrix

# -- a USD rigid body at a full pose ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class UsdBody(Asset):
    """A rigid body from ``usd_path``, its origin at ``pos`` turned ``rot_xyzw`` (world, m), unscaled, with the
    colliders, physics materials and contact offsets the file authors. ``mass`` (kg) makes it free, overriding any
    mass in the file; None keeps it kinematic. ``asset``/``xy``/``extent`` only satisfy the task-file checks: use
    :func:`usd_body`."""

    usd_path: str = ""
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


def usd_body(
    name: str,
    usd_path,
    pos,
    *,
    rot_xyzw=(0.0, 0.0, 0.0, 1.0),
    mass: float | None = None,
) -> UsdBody:
    pos = tuple(
        float(v) for v in pos
    )  # plain floats: Arena's reset cannot take NumPy scalars
    rot = tuple(float(v) for v in rot_xyzw)
    if not abs(math.hypot(*rot) - 1.0) < 1e-6:
        raise ValueError(
            f"{name}: rot_xyzw must be a unit quaternion, got {rot_xyzw!r}"
        )
    return UsdBody(
        name,
        asset=name,
        xy=pos[:2],
        extent=0.1,
        mass=mass,
        usd_path=str(usd_path),
        pos=pos,
        rot_xyzw=rot,
    )


def _spawn(scene, spec: UsdBody):
    if spec.free:
        addon = {
            "rigid_props": scene.PhysxRigidBodyPropertiesCfg(kinematic_enabled=False),
            "mass_props": {"": [scene.sim_utils.MassCfg(mass=spec.mass)]},
            "mass_props_create_if_missing": True,
        }
    else:
        addon = {
            "func": scene._spawn_fixture,
            "rigid_props": scene.PhysxRigidBodyPropertiesCfg(kinematic_enabled=True),
        }
    return scene.Object(
        name=spec.name,
        usd_path=spec.usd_path,
        object_type=scene.ObjectType.RIGID,
        scale=(1.0, 1.0, 1.0),
        initial_pose=scene.Pose(position_xyz=spec.pos, rotation_xyzw=spec.rot_xyzw),
        spawn_cfg_addon=addon,
    )


def _patch_scene(scene) -> None:
    if getattr(scene, "_screwdriver_task", False):
        return
    stock = scene.task_object

    def task_object(spec):
        return _spawn(scene, spec) if isinstance(spec, UsdBody) else stock(spec)

    scene.task_object, scene._screwdriver_task = task_object, True


class _AfterSceneImport(importlib.abc.MetaPathFinder):
    """Patches ``yam_arena.scene`` as soon as it is imported (it imports Isaac Sim, so not before the app starts)."""

    def find_spec(self, fullname, path, target=None):
        if fullname != "yam_arena.scene":
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch_scene(module)

        spec.loader.exec_module = exec_and_patch
        return spec


# -- a grip tipped about its fingers' closing axis -------------------------------------------------------------------


@dataclass
class TiltedWaypoint(demo.Waypoint):
    tilt: float = 0.0  # first tipped about the fingers' closing axis (rad): -pi/2 points the fingers along the heading


def grip_rotation(yaw: float = 0.0, tilt: float = 0.0) -> np.ndarray:
    """The TCP rotation of a straight-down grip tipped ``tilt`` about the fingers' closing axis (world y), then
    turned ``yaw`` about the vertical. ``tilt=-pi/2`` holds the hand level, its fingers pointing along
    ``(cos yaw, sin yaw, 0)``: an object gripped from above lying along that line, its far end pointing back,
    ends up upright with that end down."""
    rot = quat_xyzw_to_matrix(np.asarray(demo.DOWN_GRIP))
    if tilt != 0.0:
        rot = axis_angle_matrix(np.array([0.0, 1.0, 0.0]), tilt) @ rot
    return (
        rot if yaw == 0.0 else axis_angle_matrix(np.array([0.0, 0.0, 1.0]), yaw) @ rot
    )


def _target(self, t: float):
    """``ArmPlan.target``, reading each waypoint's ``tilt`` (0 if it has none)."""
    prev_pos, prev_rot, prev_finger = self.start_pos, self.start_rot, self.finger_start
    for w in self.waypoints:
        goal, rot = (
            np.asarray(w.pos, dtype=float),
            grip_rotation(w.yaw, getattr(w, "tilt", 0.0)),
        )
        if t < w.duration:
            s = ik.smoothstep(t / w.duration)
            return (
                prev_pos + s * (goal - prev_pos),
                ik.slerp_matrix(prev_rot, rot, s),
                prev_finger + s * (w.finger - prev_finger),
                w.label,
            )
        t -= w.duration
        prev_pos, prev_rot, prev_finger = goal, rot, w.finger
    return prev_pos, prev_rot, prev_finger, "done"


# -- install ---------------------------------------------------------------------------------------------------------


def install() -> None:
    """Patch the adapter (once per process)."""
    demo.ArmPlan.target = _target
    if "yam_arena.scene" in sys.modules:
        _patch_scene(sys.modules["yam_arena.scene"])
    elif not any(isinstance(f, _AfterSceneImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _AfterSceneImport())


install()
