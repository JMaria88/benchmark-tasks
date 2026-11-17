# Drive a cap screw into a threaded insert with a screwdriver

A brass threaded insert stands fixed on the table. A steel socket cap screw lies on its side in front of the left
arm, and a screwdriver with a red octagonal handle lies in front of the right arm. One hand picks up the screw, stands
it up and sets its tip in the insert's mouth, and keeps hold of it: with only its first turn of thread caught, a screw
let go there leans over. The other hand picks up the screwdriver, stands it up and presses its hex bit into the screw's
socket. Then the first hand lets go of the screw and holds the screwdriver's shaft upright, lightly enough that it can
turn, while the other grips the handle from above and turns it clockwise, half a turn at a time (a wrist turns only
about 240 degrees), letting go and turning back between strokes, until the screw's head sits on the insert.

The task is one task file, [`screwdriver_task.py`](screwdriver_task.py), for the `yam_arena` embodiment (scene,
scoring and the scripted demo), plus two modules it imports from the same folder: [`partkit.py`](partkit.py), which
generates the screw, the insert and the screwdriver, and [`yamkit.py`](yamkit.py), which adds what the task needs to
the task-file format at runtime without changing `yam_arena`'s files.

What it tests: standing up parts that lie on their side, lining a tool up with a small socket and pressing it in, turning
in repeated strokes (let go, turn back, take it again), and handing the job of holding something still from one hand to
the other at the right moments.

How it differs from the fastening tasks already in this repository: in
[`screw-into-a-nut-held-behind-a-plate`](../screw-into-a-nut-held-behind-a-plate) the screw is turned by hand, not with
a tool; in [`driver-bit-handle-repair`](../driver-bit-handle-repair) a powered screwdriver does the turning; in
[`confined-ratchet-fastening`](../confined-ratchet-fastening) the bolt is already started and stands on its own. Here
the tool is a hand screwdriver turned in strokes, both parts start lying loose and have to be stood up, the screw
stands in the insert only while held until the bit is in its socket, and the screwdriver stays up only while one hand
holds it, so the job of holding passes from one hand to the other twice. And the thread is a real one: the screw and
the insert's bore are helical surfaces collided as signed distance fields, so the screw goes in only by turning, at
its pitch, rather than through a joint that switches on when the thread is judged to have caught.

Why two hands: the screw set in the insert's mouth stands only while held, and a screwdriver standing on it falls over
when nothing holds it. Every time the turning hand lets go to turn back, the other hand has to be holding the
screwdriver up.

## How it is modelled

`partkit.py` builds each part as a mesh and writes it to a USD file with PhysX colliders (generated on first use into
the system's temporary directory, named by their parameters, so nothing generated is committed):

- **The thread** is a real helix on both parts: the screw's surface and the insert's bore follow the same trapezoidal
  thread profile (12 mm across, 3 mm pitch, 30 degree flanks), the bore 0.3 mm further out. Both are collided as
  signed distance fields (SDF), so the screw can only go in by turning, at the thread's pitch. The insert's mouth has a
  1.5 mm 45 degree chamfer and the screw's tip a 1.5 mm lead-in.
- **The screw's head** is round, with a 10 mm hex socket whose mouth is countersunk 2.5 mm; it is collided as an SDF
  too, so the bit can enter the socket.
- **The screwdriver** is three convex parts: a hex bit (its tip tapered 2 mm), a round shaft, and an octagonal handle
  (it lies still on one flat, and a gripper closes on two).

The masses are 20 g (screw) and 80 g (screwdriver); the insert is kinematic. Friction: 0.3 on the thread and between
bit and socket, 1.2 on the knurled head and the rubber handle.

`yamkit.py` adds, at runtime: a `UsdBody` object type (a USD rigid body at a full pose, so a part can start lying on
its side), spawned by wrapping `yam_arena.scene.task_object`; and a grip tipped about its fingers' closing axis
(`TiltedWaypoint`), for the scripted demo to stand parts up, by replacing `ArmPlan.target` with the same code reading
a waypoint's tilt. Plain waypoints play as before. Tested with yam-isaac-lab-arena at commit `0059d02`.

The score is computed from the screw's and the screwdriver's poses after every control step (see `task.json`): the gap
between the screw head's underside and the insert's top, the screw's tilt and how far it is off the insert's axis, and
whether the bit is in the socket (its tip at least 2 mm below the socket's top, within 1.5 mm of the
screw's axis, the two axes within 10 degrees).

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` (the screw seated: a gap under 0.6 mm, upright within 3 degrees and on
the insert's axis within 1 mm, with the bit in its socket) has held for 1 s, which also ends the trial; otherwise
`score()` at the end of the episode (0, 0.2, or 0.3 to 1.0; see `task.json`). Driving counts only with the bit in the
socket: a screw turned in by its head, with the screwdriver elsewhere, scores 0.2. `-E object_jitter=0` keeps every object exactly where the task file puts
it, so every trial starts from the same layout. The cameras render at 1280 x 704 for sharp videos; Dream-1's
client scales every frame to 320 x 176 before sending it. Run these from this folder.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=screwdriver_task.py -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 \
  --instruction "Screw the steel cap screw into the brass insert with the red screwdriver. Pick up the screw and the screwdriver from where they lie, stand the screw up in the insert's mouth and hold it there, put the screwdriver's hex bit into the screw's socket, then turn the screwdriver clockwise, seen from above, until the screw's head sits on top of the insert. Neither the screw nor the screwdriver stays standing on its own until the screw is driven in, so keep one of them held while the other hand works." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/screwdriver-dream
```

The scripted demo, which reaches full score. Both arms pick up their part from above at once and stand it up, turning
the hand level on the way. The left hand lowers the screw into the insert's mouth and holds it; the right hand, which
has waited beside the insert, brings the bit over the socket and presses it in, the countersink and the bit's taper
steering it. The left hand eases its grip and turns the screw slowly 30 degrees each way, so the socket meets the bit
whatever way the screw ended up turned. It lets go, rises round the shaft and holds it lightly. The right hand lets go,
draws back, comes down beside the handle from the side and grips it from above, then turns it clockwise in strokes of
180 degrees, letting go and turning back between them; the screw seats after about three:

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=screwdriver_task.py -E object_jitter=0 -E control_hz=20 -E cam_width=1280 -E cam_height=704 \
  --instruction "Screw the steel cap screw into the brass insert with the red screwdriver. Pick up the screw and the screwdriver from where they lie, stand the screw up in the insert's mouth and hold it there, put the screwdriver's hex bit into the screw's socket, then turn the screwdriver clockwise, seen from above, until the screw's head sits on top of the insert. Neither the screw nor the screwdriver stays standing on its own until the screw is driven in, so keep one of them held while the other hand works." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/screwdriver-demo
```

The demo is paced for `inspect-robots`' default guardrails (each control step may change an action by at most 5% of
its range), so it plays the same with them on.

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/screwdriver-demo/*.json; do inspect-robots video "$log"; done
```

## Tests and physics check

`python -m pytest` in this folder (no simulator) checks that every part's mesh is closed, that a mated screw keeps the
thread's clearance everywhere, the socket, bit and handle shapes, that the patches play plain waypoints unchanged,
that the demo's every waypoint is reachable, that a screw lying on the table scores 0, and that a seated screw scores
1.0 only with the bit in its socket (0.2 without).

`python check_physics.py` runs the task's screw and insert in PhysX with no robot: left alone in the insert's mouth,
the screw holds in the thread and leans but stands (about 5 degrees); turned clockwise with a small torque it goes down
about one pitch per turn and seats on the insert; turned back, it rises one pitch per turn.

## Runtime per trial

On one NVIDIA L40S: about 5 minutes for a Dream-1 trial of 1,800 steps (310 s on the clock, with Dream answering in
0.72 s on average), and about 4 minutes for the scripted demo (1,200 steps, 60 s of simulated time, 224 s on the
clock, at 1280 x 704). Isaac Sim's first start on a new machine adds a few minutes while it compiles its shaders, and the first trial
generates the parts' USD files (a few seconds).

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed, and the scripted demo follows fixed waypoints.
