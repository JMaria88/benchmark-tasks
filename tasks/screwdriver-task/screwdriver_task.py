"""On the station table: a brass threaded insert (17 mm across, 25 mm tall) fixed upright on the table, right of
centre; a steel socket cap screw (20 mm round head, 9 mm threaded shank, 10 mm hex socket) lying on its side
in front of the left arm; and a hex-bit screwdriver (red octagonal handle, 10 mm hex bit) lying in front of the
right arm. The screw's thread is right-handed with a 3 mm pitch: turned clockwise seen from above it goes down.
The goal is the screw driven into the insert until its head sits on the insert's top.

Set in the insert's mouth, the screw only stands while held: just its tip is in the thread. The screwdriver is
tall and top-heavy, and falls over if nothing holds it. A wrist turns about 240 deg, so the handle is turned
half a turn at a time, the turning hand letting go and regripping in between while something else keeps the
driver upright. The socket's mouth is countersunk and the bit's tip tapered, so pressing the bit onto the
socket steers it in.

Run it:  inspect-robots run --embodiment yam_arena -E task=screwdriver_task.py -E control_hz=20 \\
           --instruction "screw the screw into the insert with the screwdriver" --scorer task_score --max-steps 1800
Demo:    add --policy yam_scripted -E object_jitter=0 (it reaches success in about 60 s, 1,200 steps at 20 Hz)
"""

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN
from yam_arena.ik import matrix_to_quat_xyzw
from yam_arena.task_file import TABLE_TOP
from yam_arena.urdf import axis_angle_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))  # this folder's own modules

import partkit
from yamkit import TiltedWaypoint, grip_rotation, usd_body

SCENE = "station"
TOP = TABLE_TOP[SCENE]

THREAD = partkit.Thread(major_d=0.012, pitch=0.003, depth=0.001, clearance=0.0003)
SCREW = partkit.Screw(
    thread=THREAD, length=0.009, head_d=0.020, head_height=0.012, mass=0.02
)
INSERT = partkit.Insert(thread=THREAD, height=0.025, outer_d=0.017, chamfer=0.0015)
DRIVER = partkit.Driver(
    shaft_length=0.100,
    shaft_d=0.009,
    handle_flats=0.026,
    handle_length=0.060,
    mass=0.08,
)
THREAD_FRICTION = (0.3, 0.3)  # static, dynamic: dry steel on brass
ASSETS = (
    Path(tempfile.gettempdir()) / "yam_tasks" / "screwdriver_task"
)  # generated, named by their parameters
PATHS = partkit.build(SCREW, INSERT, ASSETS, friction=THREAD_FRICTION, driver=DRIVER)

Z = np.array([0.0, 0.0, 1.0])


def _rz(deg: float) -> np.ndarray:
    return axis_angle_matrix(Z, math.radians(deg))


def _quat(rot: np.ndarray) -> tuple[float, float, float, float]:
    return tuple(float(v) for v in matrix_to_quat_xyzw(rot))


# -- layout ----------------------------------------------------------------------------------
# Each object lies where its arm picks it from above and, tipping its hand level (TiltedWaypoint.tilt = -90 deg),
# stands it upright: it lies along the hand's heading, its far end (the screw's tip, the driver's bit)
# pointing back at the arm. Picked at yaw psi, an object ends up turned by the same rotation as the hand.

INSERT_XY = np.array([0.30, -0.16])
SCREW_YAW = -57.0  # the left hand's heading, picking and placing the screw (deg)
DRIVER_YAW = 45.0  # the right hand's heading, picking and placing the driver (deg)
# The driver stands up turned DRIVER_YAW; the screw is laid so it stands up turned the same, mod 60 deg, so the
# bit nominally meets the socket square.
SIGMA = DRIVER_YAW - 60.0  # the screw's yaw, upright


def _stood_up(psi: float) -> np.ndarray:
    """The rotation an object gripped from above at heading ``psi`` goes through as the hand tips level."""
    return (
        grip_rotation(math.radians(psi), -math.pi / 2)
        @ grip_rotation(math.radians(psi)).T
    )


SCREW_UP = _rz(SIGMA)
SCREW_LYING = _stood_up(SCREW_YAW).T @ SCREW_UP
DRIVER_UP = _rz(DRIVER_YAW)
DRIVER_LYING = _stood_up(DRIVER_YAW).T @ DRIVER_UP

HEAD_C = (
    SCREW.length + SCREW.head_height / 2
)  # the head's centre above the screw's tip, on its axis
HANDLE_C = DRIVER.shaft_length + DRIVER.handle_length / 2
# Each is gripped at its centre of mass (on its axis: in the screw's head, in the driver's handle), so its
# weight cannot turn it in the fingers.
SCREW_GRIP = partkit.centre_of_mass(
    partkit.shank_mesh(SCREW), partkit.head_mesh(SCREW)
)[1][2]
DRIVER_GRIP = partkit.centre_of_mass(
    partkit.bit_mesh(DRIVER), partkit.shaft_mesh(DRIVER), partkit.handle_mesh(DRIVER)
)[1][2]
SCREW_HEAD_XY = np.array([0.22, 0.12])  # where the screw's head lies
DRIVER_HANDLE_XY = np.array([0.24, -0.30])  # where the driver's handle lies
REST = 0.0005  # spawned this far above the table


def _lying_origin(head_xy, rot, axial: float, radius: float) -> np.ndarray:
    """The origin (on the axis) of an object lying on its side, its point ``axial`` along the axis over ``head_xy``."""
    centre = np.array([*head_xy, TOP + radius + REST])
    return centre - rot @ np.array([0.0, 0.0, axial])


SCREW_POS = _lying_origin(SCREW_HEAD_XY, SCREW_LYING, HEAD_C, SCREW.head_d / 2)
DRIVER_POS = _lying_origin(
    DRIVER_HANDLE_XY, DRIVER_LYING, HANDLE_C, DRIVER.handle_flats / 2
)

OBJECTS = [
    usd_body("insert", PATHS["insert"], (*INSERT_XY, TOP)),
    usd_body(
        "screw", PATHS["screw"], SCREW_POS, rot_xyzw=_quat(SCREW_LYING), mass=SCREW.mass
    ),
    usd_body(
        "driver",
        PATHS["driver"],
        DRIVER_POS,
        rot_xyzw=_quat(DRIVER_LYING),
        mass=DRIVER.mass,
    ),
]

# -- scoring ---------------------------------------------------------------------------------

INSERT_TOP = TOP + INSERT.height
SEATED_GAP = 0.0006  # head within ~70 deg of a turn of seated
MAX_TILT_DEG = 3.0
MAX_OFF_AXIS = 0.001
# A screw set in the insert's mouth has its tip ~2 mm inside, on the thread's first turn: its head's underside
# 7 mm up. The drive is scored from there.
GAP_STARTED = SCREW.length - 0.002


def _axis(quat_xyzw) -> np.ndarray:
    """An object's body z in the world."""
    x, y, z, w = quat_xyzw
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def _tilt(axis) -> float:
    return math.degrees(math.acos(float(np.clip(axis[2], -1.0, 1.0))))


def _screw(objects):
    """(head underside above the insert's top, tilt in deg, the underside's centre off the insert's axis, m)."""
    s = objects["screw"]
    axis = _axis(s.quat_xyzw)
    underside = s.pos + axis * (
        SCREW.length - SCREW.height / 2
    )  # pos is the bounding-box centre
    return (
        float(underside[2] - INSERT_TOP),
        _tilt(axis),
        float(np.hypot(*(underside[:2] - INSERT_XY))),
    )


def started(objects) -> bool:
    """The screw is standing in the insert's mouth, its tip inside."""
    gap, tilt, off = _screw(objects)
    return gap < SCREW.length and tilt < 10.0 and off < 0.002


def engaged(objects) -> bool:
    """The driver's bit is in the screw's socket, the two in line."""
    s, d = objects["screw"], objects["driver"]
    s_axis, d_axis = _axis(s.quat_xyzw), _axis(d.quat_xyzw)
    socket_top = s.pos + s_axis * (SCREW.height / 2)
    bit_tip = d.pos - d_axis * (DRIVER.length / 2)
    depth = float(np.dot(socket_top - bit_tip, s_axis))
    off = float(np.linalg.norm(np.cross(bit_tip - socket_top, s_axis)))
    return bool(
        depth > 0.002
        and off < 0.0015
        and float(np.dot(s_axis, d_axis)) > math.cos(math.radians(10))
    )


def success(objects):
    """Seated with the screwdriver: the head's underside within 0.6 mm of the insert's top, upright and on its
    axis, and the bit in the socket; held for SUCCESS_HOLD_S (1 s). Not ``resting()``: a tightened screw's SDF
    contacts leave PhysX reporting velocity noise while its pose stays put."""
    gap, tilt, off = _screw(objects)
    return bool(
        gap < SEATED_GAP
        and tilt < MAX_TILT_DEG
        and off < MAX_OFF_AXIS
        and engaged(objects)
    )


def distance(objects):
    """Metres the screw's head still has to go: to the insert's mouth while it is off it, then down."""
    gap, _, off = _screw(objects)
    return float(np.hypot(max(gap, 0.0), off))


def score(objects):
    """Partial credit (success scores 1.0): 0.2 for the screw standing in the insert; with the bit in its socket,
    0.1 more and 0.7 for the share of the way driven from where a set-down screw sits to seated. The drive counts
    only with the bit in the socket: a screw turned in by its head, not with the screwdriver, scores 0.2."""
    if not started(objects):
        return 0.0
    if not engaged(objects):
        return 0.2
    gap, _, _ = _screw(objects)
    driven = float(np.clip((GAP_STARTED - gap) / (GAP_STARTED - SEATED_GAP), 0.0, 1.0))
    return 0.3 + 0.7 * driven


# -- the scripted demo ---------------------------------------------------------------------
# Left arm: picks the screw by its head, stands it up, sets it in the insert's mouth and holds it there until the
# bit is in its socket, then holds the driver's shaft lightly while the right arm turns it. Right arm: picks the
# driver by its handle, stands it up beside the insert, presses the bit into the socket, lets go once the left
# hand has the shaft, then grips the handle from above and turns it clockwise, half a turn at a time.

SQUEEZE = 0.002  # each pad is driven this far into what it holds: ~10 N at the gripper's 5000 N/m
PICK_Z = 0.015  # TCP height above the table picking a lying object: the pad tips clear the table by ~2 mm
SCREW_LIFT_Z = 0.115  # TCP heights to stand each object up at, where it picked it: the far end swings below
DRIVER_LIFT_Z = 0.20
SET_DEPTH = (
    SCREW.length - GAP_STARTED
)  # the left hand holds the screw's tip this far inside the insert's mouth
SOCKET_TOP = INSERT_TOP - SET_DEPTH + SCREW.height  # where the socket is then
STEADY_Z = (
    TOP + 0.087
)  # the steadying hand's TCP: its fingers clear the screw's head and the handle
LIGHT = 0.0005  # a light grip, each pad driven this far in: ~2.5 N, which holds a part upright but lets it turn
PRESS = 0.002  # the right hand aims the bit's tip this far below the socket's floor: it pushes, never just rests
LINE_UP = (
    SCREW_YAW + 30.0,
    SCREW_YAW - 30.0,
    SCREW_YAW,
)  # headings the left hand turns the screw through
LINE_UP_S = 2.5  # seconds for each
CLEAR = 0.09  # and, letting go, takes its open fingers this far back before turning them down
DRIVER_ASIDE = (
    0.05  # the right hand waits this far back along its heading while the screw goes in
)
BACK_OFF = 0.035  # the right hand, opened and level, backs out this far along its heading: tips clear the handle
TURN_GRIP_YAW = (
    DRIVER_YAW + 45.0
)  # the right hand closes on two of the handle's flats here (they repeat every
# 45 deg)...
TURN_DEG = 180.0  # ...and turns it this far clockwise, each time
TURNS = 5  # the drive takes three or four (the pads give ~10 deg a turn); the episode ends at success


# inspect-robots' default guardrails limit each control step's change to 5% of an action's range: at 10 Hz the
# gripper (0 closed .. 1 open) moves at most 0.5 a second (1.0 at 20 Hz), and the plan gives its fingers that long,
# so it plays the same at either rate.
GRIP_RATE = 0.5


def _up(p, dz: float) -> np.ndarray:
    return np.asarray(p, dtype=float) + np.array([0.0, 0.0, dz])


def _heading(psi: float) -> np.ndarray:
    return np.array([math.cos(math.radians(psi)), math.sin(math.radians(psi)), 0.0])


def _grasp(psi: float, point_world) -> tuple[np.ndarray, np.ndarray]:
    """(TCP picking from above at heading ``psi``, the object's ``point_world`` in that TCP's frame)."""
    tcp = np.array([point_world[0], point_world[1], TOP + PICK_Z])
    return tcp, grip_rotation(math.radians(psi)).T @ (np.asarray(point_world) - tcp)


def _level_tcp(psi: float, grasp: np.ndarray, point_target) -> np.ndarray:
    """The level hand's TCP that puts the grasped point at ``point_target``."""
    return (
        np.asarray(point_target)
        - grip_rotation(math.radians(psi), -math.pi / 2) @ grasp
    )


class _Arm:
    """Waypoints for one arm; ``hold`` keeps the last pose, so both arms' plans stay in step."""

    def __init__(self):
        self.plan: list[TiltedWaypoint] = []

    def go(self, duration, pos, finger, label, yaw_deg, tilt_deg=0.0):
        if self.plan:  # time for the fingers to get there at the rate inspect-robots' default guardrails allow
            change = abs(finger - self.plan[-1].finger) / abs(FINGER_OPEN)
            duration = max(duration, change / GRIP_RATE + 0.3)
        self.plan.append(
            TiltedWaypoint(
                duration,
                tuple(float(v) for v in pos),
                finger,
                label,
                math.radians(yaw_deg),
                math.radians(tilt_deg),
            )
        )

    def hold(self, duration, label="wait", finger=None):
        last = self.plan[-1]
        self.plan.append(
            TiltedWaypoint(
                duration,
                last.pos,
                last.finger if finger is None else finger,
                label,
                last.yaw,
                last.tilt,
            )
        )

    @property
    def t(self) -> float:
        return sum(w.duration for w in self.plan)


def _sync(*arms: _Arm) -> None:
    end = max(a.t for a in arms)
    for a in arms:
        if a.t < end - 1e-9:
            a.hold(end - a.t)


def demo():
    left, right = _Arm(), _Arm()
    open_, closed = FINGER_OPEN, FINGER_CLOSED
    held_s = -(
        SCREW.head_d / 2 - SQUEEZE
    )  # finger targets just inside each object: a firm, not crushing, grip
    held_d = -(DRIVER.handle_flats / 2 - SQUEEZE)
    steady = -(DRIVER.shaft_d / 2 - LIGHT)  # a light grip on the round shaft
    released = -(
        DRIVER.handle_flats / 2 / math.cos(math.pi / 8) + 0.006
    )  # clear of the handle's corners

    # 1. both arms pick their object and stand it up
    ps, sg = _grasp(SCREW_YAW, SCREW_POS + SCREW_LYING @ [0.0, 0.0, SCREW_GRIP])
    left.go(1.5, _up(ps, 0.08), open_, "to the screw", SCREW_YAW)
    left.go(1.2, ps, open_, "descend", SCREW_YAW)
    left.go(0.8, ps, held_s, "grip the head", SCREW_YAW)
    lifted = _up(ps, SCREW_LIFT_Z - PICK_Z)
    left.go(1.5, lifted, held_s, "lift", SCREW_YAW)
    tip_in = np.array([*INSERT_XY, INSERT_TOP - SET_DEPTH])
    above = _level_tcp(
        SCREW_YAW, sg, _up(tip_in, SCREW_GRIP + 0.06)
    )  # the tip 6 cm over the insert's mouth
    # turning level on the way: the hand stays above 0.1 m, so the screw swings clear of the insert
    left.go(3.5, above, held_s, "stand it up over the insert", SCREW_YAW, -90)
    over_mouth = _level_tcp(SCREW_YAW, sg, _up(tip_in, SCREW_GRIP + 0.01))
    left.go(1.0, over_mouth, held_s, "down over the mouth", SCREW_YAW, -90)
    pd, dg = _grasp(DRIVER_YAW, DRIVER_POS + DRIVER_LYING @ [0.0, 0.0, DRIVER_GRIP])
    right.go(1.5, _up(pd, 0.08), open_, "to the driver", DRIVER_YAW)
    right.go(1.2, pd, open_, "descend", DRIVER_YAW)
    right.go(0.8, pd, held_d, "grip the handle", DRIVER_YAW)
    lifted = _up(pd, DRIVER_LIFT_Z - PICK_Z)
    right.go(1.5, lifted, held_d, "lift", DRIVER_YAW)
    # it waits beside the insert, the bit 3 cm above where the socket will be, while the screw goes in
    bit_high = np.array([*INSERT_XY, SOCKET_TOP + 0.03])
    stage = _level_tcp(DRIVER_YAW, dg, _up(bit_high, DRIVER_GRIP))
    aside = stage - _heading(DRIVER_YAW) * DRIVER_ASIDE
    right.go(3.5, aside, held_d, "stand it up beside the insert", DRIVER_YAW, -90)
    _sync(left, right)

    # 2. the left hand sets the screw in the insert's mouth
    place = _level_tcp(SCREW_YAW, sg, _up(tip_in, SCREW_GRIP))
    left.go(0.8, place, held_s, "set it in", SCREW_YAW, -90)
    right.hold(left.t - right.t)

    # 3. holding the screw there (only its lead-in is in the thread: let go, it would topple), the left hand eases
    # to a light grip, which keeps it upright but lets it turn, and the right hand presses the bit down onto the
    # socket. The round head rolls a little as the fingers close on it, so the socket's turn is not quite known;
    # the countersunk mouth and the bit's taper steer the bit in and turn the screw to meet it. Then the left
    # hand lets go of the head, rises round the shaft and holds that lightly, upright but free to turn.
    engaged_tip = np.array([*INSERT_XY, SOCKET_TOP - SCREW.socket_depth])
    right.go(1.2, stage, held_d, "bit over the socket", DRIVER_YAW, -90)
    right.hold(0.8, "let it settle")
    _sync(left, right)
    left.go(0.5, place, -(SCREW.head_d / 2 - LIGHT), "ease the grip", SCREW_YAW, -90)
    right.hold(left.t - right.t)
    pressed = _up(engaged_tip, -PRESS)
    seat = _level_tcp(DRIVER_YAW, dg, _up(pressed, DRIVER_GRIP))
    right.go(1.5, seat, held_d, "bit into the socket", DRIVER_YAW, -90)
    right.hold(1.0, "press it in")
    _sync(left, right)
    # if the hex did not line up, the bit sits pressed in the countersink, which with the insert's mouth holds the
    # screw on its axis: the left hand turns it slowly 30 deg each way (the hex repeats every 60) until the bit
    # drops in; then the pads just slip on the head
    for i, psi in enumerate(LINE_UP):
        light = -(SCREW.head_d / 2 - LIGHT)
        left.go(
            LINE_UP_S,
            _level_tcp(psi, sg, _up(tip_in, SCREW_GRIP)),
            light,
            f"line up the socket {i + 1}",
            psi,
            -90,
        )
    right.hold(left.t - right.t, "press it in")
    left.go(0.5, place, open_, "let go", SCREW_YAW, -90)
    steady_at = _up(
        place, STEADY_Z - place[2]
    )  # straight up: the open pads pass either side of the head
    left.go(1.0, steady_at, open_, "up to the shaft", SCREW_YAW, -90)
    left.go(0.6, steady_at, steady, "steady the shaft", SCREW_YAW, -90)
    right.hold(left.t - right.t)
    right.go(0.6, seat, open_, "let go", DRIVER_YAW, -90)
    # out along its heading (away from the left hand), then round to point down, well clear of the handle
    out = seat - _heading(DRIVER_YAW) * BACK_OFF
    right.go(0.8, out, open_, "back off", DRIVER_YAW, -90)
    right.go(
        1.0,
        _up(seat - _heading(DRIVER_YAW) * CLEAR, 0.03),
        open_,
        "clear of the handle",
        DRIVER_YAW,
        -90,
    )

    # 4. turn the handle from above: grip, turn, let go, turn back. The handle sinks a pitch per turn.
    k = THREAD.pitch / 360.0
    handle_bottom = engaged_tip[2] + DRIVER.shaft_length
    grip_z = handle_bottom + 0.030
    yaw0 = TURN_GRIP_YAW
    slide = np.array(
        [math.cos(math.radians(yaw0)), math.sin(math.radians(yaw0)), 0.0]
    )  # the hand's own -y
    for n in range(1, TURNS + 1):
        at = np.array([*INSERT_XY, grip_z])
        if (
            n == 1
        ):  # come in from the side, along the fingers' width, so the open pads pass either side of it
            right.go(
                3.2, _up(at - slide * 0.08, 0.01), open_, "over beside the handle", yaw0
            )
            right.go(1.0, at - slide * 0.08, open_, "down beside it", yaw0)
            right.go(1.0, at, open_, "around the handle", yaw0)
        mid, end = yaw0 - TURN_DEG / 2, yaw0 - TURN_DEG
        right.go(0.6, at, closed, f"grip {n}", yaw0)
        right.go(1.6, _up(at, -k * TURN_DEG / 2), closed, f"turn {n}a", mid)
        right.go(1.6, _up(at, -k * TURN_DEG), closed, f"turn {n}b", end)
        right.go(0.5, _up(at, -k * TURN_DEG), released, f"let go {n}", end)
        grip_z -= k * TURN_DEG
        if n < TURNS:
            at = np.array([*INSERT_XY, grip_z])
            right.go(1.3, at, released, f"turn back {n}a", mid)
            right.go(1.3, at, released, f"turn back {n}b", yaw0)
    right.go(
        1.0,
        np.array([*INSERT_XY, grip_z + 0.05]),
        open_,
        "up",
        math.degrees(right.plan[-1].yaw),
    )
    _sync(left, right)
    left.go(0.6, steady_at, open_, "let go", SCREW_YAW, -90)
    right.hold(left.t - right.t)
    return {"left": left.plan, "right": right.plan}
