"""Check that the screw threads into the insert in PhysX, with no robot: does turning it drive it at the pitch?

The task's own insert stands fixed on a ground plane; the task's screw starts mated in it, ``--start-turns``
pitches up (default: its tip about 3 mm inside the insert's mouth, the thread caught). Phases:

1. settle: no torque. The thread should hold the screw: it must not slide down, tip over or creep round.
2. drive: a yaw-rate controller turns the screw clockwise (seen from above) at ``--rate`` turns/s, with
   ``--press`` N pushing it down, until the head seats or ``--drive-s`` runs out. It should descend one pitch
   per turn and stop on the insert's top face.
3. back out: the same, counter-clockwise for one turn. It should rise one pitch.

    python check_physics.py [--friction 0.3] [--out DIR]     (in this folder)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # the task's modules, next to this file

from yam_arena.checks import Checks

from screwdriver_task import INSERT, SCREW  # the task's parts

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--friction", type=float, default=0.3, help="Thread friction (shank and bore)."
)
parser.add_argument("--start-turns", type=int, default=None)
parser.add_argument("--rate", type=float, default=0.5, help="Drive speed (turns/s).")
parser.add_argument(
    "--press", type=float, default=1.0, help="Downward push while driving (N)."
)
parser.add_argument("--drive-s", type=float, default=6.0)
parser.add_argument(
    "--settle-s",
    type=float,
    default=3.0,
    help="Idle time the screw must not creep over.",
)
parser.add_argument("--assets", type=Path, default=Path("/tmp/yam_threads"))
parser.add_argument("--out", type=Path, default=None)
args = parser.parse_args()
checks = Checks("check_physics", args.out)

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import numpy as np
import torch


def main():
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg

    import partkit as threads

    screw, insert = SCREW, INSERT
    paths = threads.build(
        screw, insert, args.assets, friction=(args.friction, args.friction)
    )
    screw_usd, insert_usd = paths["screw"], paths["insert"]
    dt = 1 / 240
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device="cuda:0"))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.UsdFileCfg(usd_path=str(insert_usd)).func(
        "/World/insert", sim_utils.UsdFileCfg(usd_path=str(insert_usd))
    )
    turns = (
        args.start_turns
        if args.start_turns is not None
        else int((insert.height - 0.003) // screw.thread.pitch)
    )
    z0 = threads.mated_z(screw.thread, 0.0, turns)
    obj = RigidObject(
        RigidObjectCfg(
            prim_path="/World/screw",
            spawn=sim_utils.UsdFileCfg(usd_path=str(screw_usd)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z0)),
        )
    )
    sim.reset()
    seat_z = (
        insert.height - screw.length
    )  # tip height once the head rests on the insert

    yaw_unwrapped, last_yaw = 0.0, 0.0
    log = []

    def read():
        nonlocal yaw_unwrapped, last_yaw
        p = obj.data.root_link_pose_w.torch[0].cpu().numpy()
        x, y, z, w = p[3:]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        d = (yaw - last_yaw + math.pi) % (2 * math.pi) - math.pi
        yaw_unwrapped += d
        last_yaw = yaw
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))
        return p[:3], yaw_unwrapped, tilt

    def step(torque_z: float, force_z: float, phase: str, t: float):
        obj.set_external_force_and_torque(
            torch.tensor([[[0.0, 0.0, force_z]]], device=sim.device),
            torch.tensor([[[0.0, 0.0, torque_z]]], device=sim.device),
            is_global=True,
        )
        obj.write_data_to_sim()
        sim.step()
        obj.update(dt)
        pos, yaw, tilt = read()
        log.append((phase, t, pos[2], yaw, tilt))
        return pos, yaw, tilt

    def drive(
        turns_per_s: float, seconds: float, press: float, phase: str, stop_at_seat: bool
    ):
        # PI on the yaw rate: the screw's inertia is ~1e-6 kg m^2, so the proportional gain stays small for
        # stability at 240 Hz and the integral supplies the torque the thread's friction takes.
        kp, ki, cap = 2e-4, 5e-3, 0.3  # N m per rad/s, N m per rad, N m
        target = 2 * math.pi * turns_per_s
        start = read()
        peak, integral = 0.0, 0.0
        for i in range(int(seconds / dt)):
            w = float(obj.data.root_link_vel_w.torch[0, 5])
            integral = max(-cap / ki, min(cap / ki, integral + (target - w) * dt))
            tau = max(-cap, min(cap, kp * (target - w) + ki * integral))
            peak = max(peak, abs(tau))
            pos, _, _ = step(tau, -press, phase, i * dt)
            if stop_at_seat and pos[2] - seat_z < 2e-4 and i * dt > 0.5:
                break
        end = read()
        return start, end, peak

    # 1. settle: 0.5 s to take up the thread's play, then the screw must not creep round on its own (a screw
    # that turns itself in would score for a policy that does nothing).
    for i in range(int(0.5 / dt)):
        pos, yaw, tilt = step(0.0, 0.0, "settle", i * dt)
    z_settled, yaw_settled = pos[2], yaw
    for i in range(int(args.settle_s / dt)):
        pos, yaw, tilt = step(0.0, 0.0, "settle", 0.5 + i * dt)
    drop = f"z {z0 * 1e3:.2f} -> {pos[2] * 1e3:.2f} mm"
    checks.check(
        "settle: screw holds in the thread (drop < 0.5 mm)", z0 - pos[2] < 5e-4, drop
    )
    # PhysX's friction lets a loaded screw slip round slowly (~0.25 deg/s at friction 0.3); under 0.5 deg/s it
    # would take over 15 minutes to turn itself the 480 deg to its seat.
    creep = math.degrees(yaw - yaw_settled)
    checks.check(
        "settle: creeps under 0.5 deg/s",
        abs(creep) / args.settle_s < 0.5,
        f"turned {creep:+.2f} deg in {args.settle_s:g} s, z {(pos[2] - z_settled) * 1e3:+.3f} mm",
    )
    checks.check(
        # With only its first turn of thread caught, an unheld screw leans in the insert's mouth (~5 deg) but does not
        # fall; driven in, it straightens (checked below).
        "settle: screw leans but stands (tilt < 10 deg)",
        tilt < 10.0,
        f"tilt {tilt:.2f} deg",
    )

    # 2. drive clockwise until seated
    (ps, ys, _), (pe, ye, te), peak = drive(
        -args.rate, args.drive_s, args.press, "drive", stop_at_seat=True
    )
    turns = (ye - ys) / (2 * math.pi)
    dz = pe[2] - ps[2]
    lead = dz / turns if abs(turns) > 1e-6 else float("nan")
    pitch = screw.thread.pitch
    checks.check(
        "drive: descends one pitch per clockwise turn (+-15%)",
        abs(lead - pitch) < 0.15 * pitch,
        f"{turns:+.2f} turns, dz {dz * 1e3:+.2f} mm, lead {lead * 1e3:.2f} mm/turn (pitch {pitch * 1e3:.1f})",
    )
    seat = f"tip {pe[2] * 1e3:.2f} mm, seated at {seat_z * 1e3:.2f}"
    checks.check("drive: head seats on the insert", pe[2] - seat_z < 5e-4, seat)
    checks.check(
        "drive: stays upright",
        te < 3.0,
        f"tilt {te:.2f} deg, peak torque {peak:.3f} N m",
    )

    # 3. back out one turn
    (ps, ys, _), (pe, ye, te), peak = drive(
        args.rate, 1.0 / args.rate, 0.0, "back_out", stop_at_seat=False
    )
    turns = (ye - ys) / (2 * math.pi)
    dz = pe[2] - ps[2]
    checks.check(
        "back out: rises about one pitch per turn",
        abs(dz / turns - screw.thread.pitch) < 0.2 * screw.thread.pitch
        if turns > 0.1
        else False,
        f"{turns:+.2f} turns, dz {dz * 1e3:+.2f} mm, peak torque {peak:.3f} N m",
    )
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            args.out / "trace.csv",
            np.array([r[1:] for r in log]),
            delimiter=",",
            header="t,z,yaw,tilt",
        )
    return checks.finish(exit_process=False)


if __name__ == "__main__":
    code = main()
    app.close()
    sys.exit(code)
