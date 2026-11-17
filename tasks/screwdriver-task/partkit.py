"""The screwdriver task's parts: threaded screw and insert meshes, a hex-bit screwdriver, and USD files with SDF
colliders for them. NumPy and ``pxr`` only, and nothing from ``yam_arena``: it runs without Isaac Sim, so the task
file can build its assets before the simulator starts.

Both parts are radial height fields: rings of vertices at increasing heights, each ring's radius a function of
the angle about the part's axis (+z), closed by caps. A thread is the radius following a trapezoidal profile
of the phase ``u = frac(z / pitch - theta / 2 pi)``, which is constant along a right-hand helix. The screw's
surface and the insert's bore follow the same profile, the bore ``clearance`` further out, so the two mesh
when their frames differ by a turn ``psi`` and a rise ``pitch * (k + psi / 2 pi)`` (:func:`mated_z`): turning
the screw clockwise seen from above (negative yaw) drives it down.

The screw is a socket cap screw: a round head with a hex socket, which the screwdriver's hex bit turns. Socket
and bit share their hexagon's orientation (corners at 0, 60, ... deg), so the bit drops in when the screw's and
the driver's yaws differ by a multiple of 60 deg.

Frames: the screw's origin is its tip on the axis, the head at the top; the insert's origin is the centre of
its bottom face; the driver's is the bit's tip, the handle at the top. Metres.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Thread:
    major_d: float = 0.012  # crest diameter of the screw (m)
    pitch: float = 0.003  # rise per turn (m)
    depth: float = 0.001  # crest to root, radially (m)
    clearance: float = 0.0003  # radial gap between screw and bore (m)
    flank_deg: float = 30.0  # flank angle from the radial direction (ISO: 30)

    @property
    def minor_r(self) -> float:
        return self.major_d / 2 - self.depth

    def profile(self, u: np.ndarray) -> np.ndarray:
        """0 at the root, 1 at the crest, over one pitch (``u`` in 0..1): root flat, rising flank, crest flat,
        falling flank, the two flats equally wide."""
        flank = (
            self.depth * math.tan(math.radians(self.flank_deg)) / self.pitch
        )  # fraction of a pitch
        if not 0 < 2 * flank < 1:
            raise ValueError("thread too deep for its pitch and flank angle")
        flat = (1 - 2 * flank) / 2
        u = np.mod(u, 1.0)
        return np.interp(
            u,
            [0.0, flat, flat + flank, 2 * flat + flank, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
        )

    def radius(self, theta: np.ndarray, z: float, offset: float = 0.0) -> np.ndarray:
        return (
            self.minor_r
            + offset
            + self.depth * self.profile(z / self.pitch - theta / (2 * math.pi))
        )


@dataclass(frozen=True)
class Screw:
    thread: Thread = field(default_factory=Thread)
    length: float = 0.010  # threaded shank, tip to the underside of the head (m)
    lead_in: float = (
        0.0015  # the thread fades to the minor radius over this much of the tip (m)
    )
    head_d: float = 0.020  # round head (m)
    head_height: float = 0.012
    socket_flats: float = 0.010  # hex socket in the head's top, across the flats (m)
    socket_depth: float = 0.006
    socket_countersink: float = (
        0.0025  # the socket's mouth widens 45 deg by this much: it guides a bit in (m)
    )
    mass: float = 0.02  # kg

    @property
    def height(self) -> float:
        return self.length + self.head_height


@dataclass(frozen=True)
class Insert:
    thread: Thread = field(default_factory=Thread)
    height: float = 0.012  # also the length of the threaded bore (m)
    outer_d: float = 0.032
    chamfer: float = 0.0015  # 45 deg countersink at the bore's mouth, this deep (m): it centres a screw set on it


@dataclass(frozen=True)
class Driver:
    """A hex-bit screwdriver: an octagonal handle (it lies on a flat, and a gripper closes on two flats) on a
    round shaft whose last ``bit_length`` is the hex bit."""

    bit_flats: float = (
        0.0096  # across the flats: the socket's less 0.2 mm each side (m)
    )
    bit_chamfer: float = 0.002  # the bit's tip narrows by this much over as long (45 deg), to find the socket
    bit_length: float = 0.012
    shaft_d: float = 0.009  # the round shaft above the bit
    shaft_length: float = 0.080  # tip to handle
    handle_flats: float = 0.026
    handle_length: float = 0.060
    mass: float = 0.08  # kg

    @property
    def length(self) -> float:
        return self.shaft_length + self.handle_length


def mated_z(thread: Thread, yaw: float, turns: int) -> float:
    """Height of the screw's tip above the insert's bottom at which its thread meets the bore's, screw turned
    ``yaw`` (rad) about the shared axis from the insert's frame, ``turns`` whole pitches up."""
    return thread.pitch * (turns + yaw / (2 * math.pi))


def _polygon_radius(
    theta: np.ndarray, flats: float, sides: int, corner0: float = 0.0
) -> np.ndarray:
    """A regular polygon's radius at each angle, a corner at ``corner0`` (rad)."""
    a = np.mod(theta - corner0, 2 * math.pi / sides) - math.pi / sides
    return flats / 2 / np.cos(a)


def _hex_radius(theta: np.ndarray, flats: float) -> np.ndarray:
    """A hexagon's radius at each angle: corners at 0, 60, ... deg, flats facing 30, 90, 150, ..."""
    return _polygon_radius(theta, flats, 6)


OCTAGON_CORNER0 = (
    math.pi / 8
)  # the handle's corners at 22.5, 67.5, ... deg: flats face 0, 45, 90, ...


def _rings_mesh(
    rings: list[tuple[float, np.ndarray]], theta: np.ndarray, inward: bool = False
):
    """Vertices and triangles of a tube through ``rings`` (z, radius per theta), outward normals (or inward,
    for a bore). Open at both ends."""
    n = len(theta)
    c, s = np.cos(theta), np.sin(theta)
    verts = np.concatenate(
        [np.stack([r * c, r * s, np.full(n, z)], axis=1) for z, r in rings]
    )
    faces = []
    j = np.arange(n)
    k = (j + 1) % n
    for i in range(len(rings) - 1):
        a, b = i * n + j, i * n + k
        a2, b2 = a + n, b + n
        tri1 = np.stack([a, b, b2], axis=1)
        tri2 = np.stack([a, b2, a2], axis=1)
        faces += [tri1, tri2]
    faces = np.concatenate(faces)
    if inward:
        faces = faces[:, ::-1]
    return verts, faces


def _cap(verts, faces, ring_start: int, n: int, z: float, up: bool):
    """Fan a ring of ``n`` vertices starting at ``ring_start`` to a new centre vertex at height ``z``."""
    centre = len(verts)
    verts = np.vstack([verts, [0.0, 0.0, z]])
    j = np.arange(n)
    k = (j + 1) % n
    tri = np.stack([np.full(n, centre), ring_start + j, ring_start + k], axis=1)
    if not up:
        tri = tri[:, ::-1]
    return verts, np.vstack([faces, tri])


def _annulus(faces, outer_start: int, inner_start: int, n: int, up: bool):
    """Join two rings of ``n`` vertices at one height (an annular face, normal up or down)."""
    j = np.arange(n)
    k = (j + 1) % n
    o, o2, i, i2 = outer_start + j, outer_start + k, inner_start + j, inner_start + k
    tri = np.concatenate([np.stack([o, i2, o2], axis=1), np.stack([o, i, i2], axis=1)])
    if up:
        tri = tri[:, ::-1]
    return np.vstack([faces, tri])


def _z_samples(z0: float, z1: float, step: float) -> np.ndarray:
    return np.linspace(z0, z1, max(2, math.ceil((z1 - z0) / step) + 1))


def shank_mesh(screw: Screw, n_theta: int = 144, z_step: float | None = None):
    """(vertices, triangles) of the screw's threaded shank, a lead-in at the tip, capped flat under the head."""
    t = screw.thread
    z_step = z_step or t.pitch / 30
    theta = np.linspace(0.0, 2 * math.pi, n_theta, endpoint=False)
    rings = []
    for z in _z_samples(0.0, screw.length, z_step):
        fade = min(1.0, z / screw.lead_in) if screw.lead_in > 0 else 1.0
        rings.append((z, t.minor_r + fade * (t.radius(theta, z) - t.minor_r)))
    verts, faces = _rings_mesh(rings, theta)
    verts, faces = _cap(verts, faces, 0, n_theta, 0.0, up=False)
    return _cap(
        verts, faces, (len(rings) - 1) * n_theta, n_theta, screw.length, up=True
    )


def head_mesh(screw: Screw, n_theta: int = 144):
    """(vertices, triangles) of the round head on top of the shank, a hex socket sunk in its top."""
    if n_theta % 6:
        raise ValueError(
            "n_theta must be a multiple of 6, so the socket's corners are vertices"
        )
    corner = screw.socket_flats / 2 / math.cos(math.pi / 6) + screw.socket_countersink
    if not (
        corner < screw.head_d / 2 - 0.0005
        and screw.socket_countersink < screw.socket_depth < screw.head_height
    ):
        raise ValueError("the socket must fit inside the head")
    theta = np.linspace(0.0, 2 * math.pi, n_theta, endpoint=False)
    outer = np.full(n_theta, screw.head_d / 2)
    hex_r = _hex_radius(theta, screw.socket_flats)
    floor = screw.height - screw.socket_depth
    c = screw.socket_countersink
    verts, faces = _rings_mesh([(screw.length, outer), (screw.height, outer)], theta)
    socket = [(screw.height, hex_r + c), (screw.height - c, hex_r), (floor, hex_r)]
    sv, sf = _rings_mesh(socket, theta)  # descending: its normals face the axis
    verts, faces = np.vstack([verts, sv]), np.vstack([faces, sf + len(verts)])
    verts, faces = _cap(verts, faces, 0, n_theta, screw.length, up=False)
    faces = _annulus(faces, n_theta, 2 * n_theta, n_theta, up=True)
    return _cap(verts, faces, (2 + len(socket) - 1) * n_theta, n_theta, floor, up=True)


def screw_mesh(screw: Screw):
    """The whole screw as one mesh (shank and head; they share the face under the head)."""
    (sv, sf), (hv, hf) = shank_mesh(screw), head_mesh(screw)
    return np.vstack([sv, hv]), np.vstack([sf, hf + len(sv)])


def insert_mesh(insert: Insert, n_theta: int = 144, z_step: float | None = None):
    """(vertices, triangles) of the insert: a cylinder with a threaded through-bore."""
    t = insert.thread
    z_step = z_step or t.pitch / 30
    theta = np.linspace(0.0, 2 * math.pi, n_theta, endpoint=False)
    outer_r = np.full(n_theta, insert.outer_d / 2)
    zs = _z_samples(0.0, insert.height, z_step)
    mouth = (
        insert.height - insert.chamfer
    )  # the countersink widens the bore 1:1 above here
    if (
        not 0 <= insert.chamfer < insert.height
        or t.major_d / 2 + insert.chamfer >= insert.outer_d / 2
    ):
        raise ValueError("the insert's chamfer must fit inside it")
    cone = (
        t.major_d / 2 + t.clearance - mouth
    )  # the countersink's radius at z is cone + z
    bore = [
        (z, np.maximum(t.radius(theta, z, offset=t.clearance), cone + z)) for z in zs
    ]
    ov, of = _rings_mesh([(0.0, outer_r), (insert.height, outer_r)], theta)
    iv, if_ = _rings_mesh(bore, theta, inward=True)
    verts = np.vstack([ov, iv])
    faces = np.vstack([of, if_ + len(ov)])
    inner0, inner_top = len(ov), len(ov) + (len(bore) - 1) * n_theta
    faces = _annulus(faces, 0, inner0, n_theta, up=False)
    faces = _annulus(faces, n_theta, inner_top, n_theta, up=True)
    return verts, faces


def _prism_rings(theta, rings):
    """Rings (z, radius per theta) closed by flat caps: a solid of revolution-like prism stack."""
    n = len(theta)
    verts, faces = _rings_mesh(rings, theta)
    verts, faces = _cap(verts, faces, 0, n, rings[0][0], up=False)
    return _cap(verts, faces, (len(rings) - 1) * n, n, rings[-1][0], up=True)


def bit_mesh(driver: Driver):
    """(vertices, triangles) of the hex bit, its tip chamfered: one convex solid."""
    theta = np.radians(np.arange(0.0, 360.0, 60.0))
    full = _hex_radius(theta, driver.bit_flats)
    tip = _hex_radius(theta, driver.bit_flats - 2 * driver.bit_chamfer)
    return _prism_rings(
        theta, [(0.0, tip), (driver.bit_chamfer, full), (driver.bit_length, full)]
    )


def shaft_mesh(driver: Driver, n_theta: int = 32):
    """(vertices, triangles) of the round shaft from the bit to the handle: one convex solid."""
    if not driver.shaft_d < driver.bit_flats:
        raise ValueError(
            "the shaft must be thinner than the bit, to pass the socket's mouth"
        )
    theta = np.linspace(0.0, 2 * math.pi, n_theta, endpoint=False)
    r = np.full(n_theta, driver.shaft_d / 2)
    return _prism_rings(theta, [(driver.bit_length, r), (driver.shaft_length, r)])


def handle_mesh(driver: Driver):
    """(vertices, triangles) of the octagonal handle, flats facing 0, 45, 90, ... deg: one convex solid."""
    theta = OCTAGON_CORNER0 + np.radians(np.arange(0.0, 360.0, 45.0))
    r = _polygon_radius(theta, driver.handle_flats, 8, OCTAGON_CORNER0)
    return _prism_rings(theta, [(driver.shaft_length, r), (driver.length, r)])


def centre_of_mass(*meshes) -> tuple[float, np.ndarray]:
    """(volume, centre) of closed outward-facing meshes of one uniform density: how PhysX spreads a body's
    mass over its colliders."""
    vol, moment = 0.0, np.zeros(3)
    for verts, faces in meshes:
        a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        v = (
            np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
        )  # signed tetrahedra with the origin
        vol += v.sum()
        moment += (v[:, None] * (a + b + c) / 4.0).sum(axis=0)
    return float(vol), moment / vol


# -- USD ---------------------------------------------------------------------------------


def _key(*parts) -> str:
    blob = json.dumps(
        [asdict(p) if hasattr(p, "__dataclass_fields__") else p for p in parts],
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Part:
    """One collider of a rigid body: a mesh, how PhysX collides it, and its look and friction."""

    name: str
    verts: np.ndarray
    faces: np.ndarray
    approximation: str  # "sdf" or "convexHull"
    friction: tuple[float, float]  # static, dynamic
    color: tuple[float, float, float]
    metallic: float = 0.6
    torsional_patch: float = (
        0.0  # m: PhysX's torsional friction radius for contacts on it (0: none)
    )


def write_usd(
    path: Path, parts: list[Part], *, mass: float | None, sdf_resolution: int = 256
) -> Path:
    """One rigid body made of ``parts``. ``mass`` None: kinematic (it stays where it is put)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/part")
    stage.SetDefaultPrim(root.GetPrim())
    rb = root.GetPrim()
    body = UsdPhysics.RigidBodyAPI.Apply(rb)
    if mass is None:
        body.CreateKinematicEnabledAttr(True)
    else:
        UsdPhysics.MassAPI.Apply(rb).CreateMassAttr(mass)
    # As PhysX's nuts-and-bolts demo: more position iterations for the thread contacts, and no sleeping, so a
    # screw at rest still answers a gripper's twist at once.
    rb.AddAppliedSchema("PhysxRigidBodyAPI")
    rb.CreateAttribute(
        "physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int
    ).Set(16)
    rb.CreateAttribute(
        "physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int
    ).Set(1)
    rb.CreateAttribute("physxRigidBody:sleepThreshold", Sdf.ValueTypeNames.Float).Set(
        0.0
    )

    for part in parts:
        base = f"/part/{part.name}"
        mesh = UsdGeom.Mesh.Define(stage, base)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(part.verts.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(part.faces)))
        mesh.CreateFaceVertexIndicesAttr(
            Vt.IntArray.FromNumpy(part.faces.astype(np.int32).ravel())
        )
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        lo, hi = part.verts.min(axis=0), part.verts.max(axis=0)
        mesh.CreateExtentAttr([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())])
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*part.color)])
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
            part.approximation
        )
        if part.approximation == "sdf":
            prim.AddAppliedSchema("PhysxSDFMeshCollisionAPI")
            prim.CreateAttribute(
                "physxSDFMeshCollision:sdfResolution", Sdf.ValueTypeNames.Int
            ).Set(sdf_resolution)
        # Contacts within 1 mm: the thread's clearance is 0.3 mm, so a wider offset would make the flanks
        # touch before they meet.
        prim.AddAppliedSchema("PhysxCollisionAPI")
        prim.CreateAttribute(
            "physxCollision:contactOffset", Sdf.ValueTypeNames.Float
        ).Set(0.001)
        prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(
            0.0
        )
        if part.torsional_patch:
            # A finger pad pressed flat on it touches over a patch, which resists twisting about the pad's normal;
            # PhysX's few contact points do not, unless given a patch radius.
            prim.CreateAttribute(
                "physxCollision:torsionalPatchRadius", Sdf.ValueTypeNames.Float
            ).Set(part.torsional_patch)
            prim.CreateAttribute(
                "physxCollision:minTorsionalPatchRadius", Sdf.ValueTypeNames.Float
            ).Set(part.torsional_patch)

        look = UsdShade.Material.Define(stage, f"{base}_look")
        shader = UsdShade.Shader.Define(stage, f"{base}_look/shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*part.color)
        )
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(part.metallic)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
        look.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(look)
        phys = UsdShade.Material.Define(stage, f"{base}_physics")
        pm = UsdPhysics.MaterialAPI.Apply(phys.GetPrim())
        pm.CreateStaticFrictionAttr(part.friction[0])
        pm.CreateDynamicFrictionAttr(part.friction[1])
        pm.CreateRestitutionAttr(0.0)
        binding.Bind(phys, UsdShade.Tokens.weakerThanDescendants, "physics")
    stage.GetRootLayer().Save()
    return path


STEEL = (0.72, 0.72, 0.75)
BLACK_OXIDE = (0.12, 0.12, 0.13)
BRASS = (0.78, 0.62, 0.25)
HANDLE_RED = (0.75, 0.12, 0.10)
GRIPPY = (
    1.2,
    1.0,
)  # constants.GRIPPY_FRICTION: knurled head and rubber handle, where the fingers hold them
BIT_FRICTION = (0.3, 0.3)  # steel bit in a steel socket
GRIP_PATCH = (
    0.005  # m: where fingers hold the parts, they touch over a patch about this big
)


def _parts(kind: str, part, friction) -> tuple[list[Part], float | None]:
    if kind == "screw":
        return [
            Part("shank", *shank_mesh(part), "sdf", tuple(friction), STEEL),
            Part("head", *head_mesh(part), "sdf", GRIPPY, BLACK_OXIDE, 0.3, GRIP_PATCH),
        ], part.mass
    if kind == "insert":
        return [
            Part("body", *insert_mesh(part), "sdf", tuple(friction), BRASS, 0.8)
        ], None
    return [
        Part("bit", *bit_mesh(part), "convexHull", BIT_FRICTION, STEEL),
        Part("shaft", *shaft_mesh(part), "convexHull", BIT_FRICTION, STEEL),
        Part(
            "handle",
            *handle_mesh(part),
            "convexHull",
            GRIPPY,
            HANDLE_RED,
            0.0,
            GRIP_PATCH,
        ),
    ], part.mass


def _write_parts(jobs: dict[str, tuple[object, Path]], friction) -> None:
    """Write each file to a temporary name renamed into place, so a concurrent run never reads half a file."""
    for kind, (part, path) in jobs.items():
        if path.exists():
            continue
        parts, mass = _parts(kind, part, friction)
        tmp = path.with_name(f".{path.stem}.{os.getpid()}.usda")
        write_usd(tmp, parts, mass=mass)
        tmp.replace(path)


def _spec(part) -> dict:
    d = asdict(part)
    if "thread" in d:
        d["thread"] = asdict(part.thread)
    return d


KINDS = {"screw": Screw, "insert": Insert, "driver": Driver}


def build(
    screw: Screw,
    insert: Insert,
    out_dir: Path,
    *,
    friction=(0.3, 0.3),
    driver: Driver | None = None,
) -> dict[str, Path]:
    """Write (or reuse) the USD files of the screw, the insert and (given) the driver under ``out_dir``, named by
    their parameters: ``{"screw": path, "insert": path, "driver": path}``. ``friction`` is the thread's (shank
    and bore); the screw's head and the driver's handle are grippy.

    The files are written by a child Python: a task file calls this as it loads, before Isaac Sim starts, and
    importing ``pxr`` into that process first would stop Kit (which brings its own USD) from starting."""
    out_dir = Path(out_dir)
    parts = {
        "screw": screw,
        "insert": insert,
        **({"driver": driver} if driver is not None else {}),
    }
    paths = {
        kind: out_dir / f"{kind}_{_key(part, list(friction), 7)}.usda"
        for kind, part in parts.items()
    }
    if all(p.exists() for p in paths.values()):
        return paths
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "friction": list(friction),
        "jobs": {k: [_spec(parts[k]), str(paths[k])] for k in parts},
    }
    done = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        input=json.dumps(spec),
        text=True,
        capture_output=True,
        check=False,
    )
    if done.returncode != 0:
        raise RuntimeError(
            f"writing the screwdriver task's assets failed:\n{done.stderr[-2000:]}"
        )
    return paths


def _load(kind: str, d: dict):
    if "thread" in d:
        d = {**d, "thread": Thread(**d["thread"])}
    return KINDS[kind](**d)


def _main() -> None:
    spec = json.loads(sys.stdin.read())
    jobs = {
        kind: (_load(kind, d), Path(path)) for kind, (d, path) in spec["jobs"].items()
    }
    _write_parts(jobs, spec["friction"])


if __name__ == "__main__":
    _main()
