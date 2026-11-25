# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Teaching sketch of half-edge walks and the Quad-Edge Rot cycle.

This is not a mesh kernel. There is no splice, no boundary twin, and
no mutation. The pointers exist so a notebook can watch three
different operators on the tetrahedron boundary:

- `next` circulates one oriented triangle
- `twin` reverses that primal arrow onto the adjacent triangle
- `Rot` cycles primal forward, dual, primal reverse, dual reverse

`Sym = Rot²` reverses a primal arrow and therefore sits next to
`twin` in the comparison. `Rot` itself leaves the primal edge. A
half-edge record has no slot for that dual edge.

Classical twins need a coherent orientation, so `build_halfedge`
refuses RP² and reports the conflict from `orientation_attempt`.
Face-count Euler characteristic does not need this sketch; the gallery
in `euler_surface_demo.py` never builds it.

Guibas and Stolfi, "Primitives for the manipulation of general
subdivisions and the computation of Voronoi diagrams", ACM TOG 4
(1985), is the Quad-Edge source. The identity used here is

    Lnext(e) = Rot⁻¹(Onext(Rot(e)))

with `Onext(Rot(e))` set equal to `Rot(Lnext(e))` so the identity
holds by construction on an oriented closed triangulation.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import ipywidgets as W
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_SERIES_DIR = Path(__file__).resolve().parent
if str(_SERIES_DIR) not in sys.path:
    sys.path.insert(0, str(_SERIES_DIR))

from euler_surface_demo import (  # noqa: E402
    orientation_attempt,
    tetrahedron_boundary,
    torus_grid,
)
from lib import SimplicialComplex  # noqa: E402

_FACE = "#d7e3f4"
_EDGE = "#94a3b8"
_PRIMAL = "#c2410c"
_DUAL = "#1d4e89"


class TeachingMeshError(ValueError):
    """The sketch refuses a complex it would have to fake."""


@dataclass(frozen=True)
class HalfEdge:
    """One directed side of a triangle.

    `nxt` is the DCEL next pointer: the following directed edge of the
    same face. `twin` is the opposite direction on the same undirected
    edge, carried by the adjacent face. `face` is an index into the
    oriented face list, not a vertex id.
    """

    origin: int
    dest: int
    face: int
    twin: int
    nxt: int


@dataclass(frozen=True)
class HalfEdgeSketch:
    """Half-edges of one coherently oriented closed triangulation."""

    edges: tuple[HalfEdge, ...]
    faces: tuple[tuple[int, int, int], ...]

    def face_ring(self, index: int) -> tuple[int, ...]:
        """`next` until the walk returns. Length 3 on a triangle."""
        return _walk(index, lambda cursor: self.edges[cursor].nxt, len(self.edges))

    def vertex_ring(self, index: int) -> tuple[int, ...]:
        """Outgoing edges about `origin`, in the order `next(twin(·))`."""
        return _walk(
            index,
            lambda cursor: self.edges[self.edges[cursor].twin].nxt,
            len(self.edges),
        )


@dataclass(frozen=True)
class QuadRecord:
    """One of the four directed records in a Quad-Edge bundle.

    `primal` records have vertex endpoints. Dual records have face-index
    endpoints: `Rot` of a primal arrow runs from its left face to its
    right face. `onext` is the next record about that origin.
    """

    origin: int
    dest: int
    primal: bool
    rot: int
    onext: int


@dataclass(frozen=True)
class QuadEdgeSketch:
    """Quad-Edge records built from a half-edge sketch. No splice."""

    records: tuple[QuadRecord, ...]
    half: HalfEdgeSketch
    record_of_half: tuple[int, ...]

    def rot_cycle(self, index: int) -> tuple[int, ...]:
        """`e, Rot(e), Rot²(e), Rot³(e)`. The fourth Rot returns to `e`."""
        return _walk(index, lambda cursor: self.records[cursor].rot, 4)

    def onext_ring(self, index: int) -> tuple[int, ...]:
        """Records about one origin, primal or dual according to `index`."""
        return _walk(
            index,
            lambda cursor: self.records[cursor].onext,
            len(self.records),
        )

    def invrot(self, index: int) -> int:
        """`Rot⁻¹`, three steps of a 4-cycle."""
        cursor = index
        for _ in range(3):
            cursor = self.records[cursor].rot
        return cursor

    def lnext(self, index: int) -> int:
        """Face successor `Rot⁻¹(Onext(Rot(e)))`."""
        rotated = self.records[index].rot
        followed = self.records[rotated].onext
        return self.invrot(followed)


def _walk(start: int, step, limit: int) -> tuple[int, ...]:
    ring = [start]
    cursor = step(start)
    while cursor != start:
        ring.append(cursor)
        if len(ring) > limit:
            raise TeachingMeshError("operator walk did not close")
        cursor = step(cursor)
    return tuple(ring)


def build_halfedge(K: SimplicialComplex) -> HalfEdgeSketch:
    """Half-edges for a coherently oriented closed triangulation.

    Every edge must lie in two triangles, and the orientation
    propagation must succeed. RP² fails the second test; a single
    triangle fails the first (three boundary edges, no twin). The
    sketch does not invent a missing twin and does not store a
    boundary sentinel.
    """
    attempt = orientation_attempt(K)
    if not attempt.manifold_edges or not attempt.closed:
        raise TeachingMeshError(
            "half-edge twins in this sketch need every edge in exactly two triangles"
        )
    if not attempt.coherent or attempt.faces is None:
        edge = attempt.conflict_edge
        where = "an unknown edge" if edge is None else f"{{{edge[0]},{edge[1]}}}"
        raise TeachingMeshError(
            "classical half-edge orientation conflicts on edge "
            f"{where}; this sketch does not invent twins"
        )
    faces = attempt.faces
    directed: dict[tuple[int, int], int] = {}
    for face_index, face in enumerate(faces):
        for slot in range(3):
            origin = face[slot]
            dest = face[(slot + 1) % 3]
            index = 3 * face_index + slot
            if (origin, dest) in directed:
                raise TeachingMeshError("duplicate directed edge under this orientation")
            directed[(origin, dest)] = index
    built: list[HalfEdge] = []
    for face_index, face in enumerate(faces):
        for slot in range(3):
            origin = face[slot]
            dest = face[(slot + 1) % 3]
            twin_key = (dest, origin)
            if twin_key not in directed:
                raise TeachingMeshError(
                    f"directed edge {(origin, dest)} has no opposite"
                )
            built.append(
                HalfEdge(
                    origin=origin,
                    dest=dest,
                    face=face_index,
                    twin=directed[twin_key],
                    nxt=3 * face_index + ((slot + 1) % 3),
                )
            )
    return HalfEdgeSketch(tuple(built), faces)


def build_quadedge(K: SimplicialComplex) -> QuadEdgeSketch:
    """Four records per undirected edge, Rot-linked, Onext from the half-edges.

    Primal `Onext` is the half-edge vertex circulator `next(twin(·))`.
    Dual `Onext` is set by `Onext(Rot(e)) = Rot(next(e))`, which is the
    Guibas–Stolfi face successor rewritten so `Lnext` reproduces `next`.
    The construction is refused wherever `build_halfedge` is refused.
    """
    half = build_halfedge(K)
    seeds: list[int] = []
    seen: set[frozenset[int]] = set()
    for index, edge in enumerate(half.edges):
        key = frozenset((edge.origin, edge.dest))
        if key in seen:
            continue
        seen.add(key)
        seeds.append(index if edge.origin < edge.dest else edge.twin)
    groups = len(seeds)
    rot = [0] * (4 * groups)
    onext = [-1] * (4 * groups)
    origin = [0] * (4 * groups)
    dest = [0] * (4 * groups)
    primal = [False] * (4 * groups)
    record_of_half = [-1] * len(half.edges)
    for group, seed in enumerate(seeds):
        base = 4 * group
        forward = half.edges[seed]
        back = half.edges[forward.twin]
        for step in range(4):
            rot[base + step] = base + ((step + 1) % 4)
        for record, edge, edge_index in (
            (base, forward, seed),
            (base + 2, back, forward.twin),
        ):
            primal[record] = True
            origin[record] = edge.origin
            dest[record] = edge.dest
            record_of_half[edge_index] = record
        origin[base + 1] = forward.face
        dest[base + 1] = back.face
        origin[base + 3] = back.face
        dest[base + 3] = forward.face
    for index, edge in enumerate(half.edges):
        record = record_of_half[index]
        circulator = half.edges[edge.twin].nxt
        onext[record] = record_of_half[circulator]
        onext[rot[record]] = rot[record_of_half[edge.nxt]]
    if any(value < 0 for value in onext) or any(value < 0 for value in record_of_half):
        raise TeachingMeshError("Quad-Edge pointers were not all set")
    records = tuple(
        QuadRecord(
            origin=origin[index],
            dest=dest[index],
            primal=primal[index],
            rot=rot[index],
            onext=onext[index],
        )
        for index in range(4 * groups)
    )
    return QuadEdgeSketch(records, half, tuple(record_of_half))


def flip_sentence(K: SimplicialComplex, sketch: HalfEdgeSketch) -> str:
    """Describe one diagonal a flip would rewrite. Does not rewire it.

    On the tetrahedron boundary every fourth vertex pair is already an
    edge, so the sentence says the candidate diagonal is already
    present. On the grid torus the sentence names a missing diagonal.
    Either way the sketch stops at the report: an industrial kernel
    would edit `next` and `twin`, and a Quad-Edge kernel would splice
    `Onext` so the dual of that edge moves with it.
    """
    present = {
        frozenset((edge.origin, edge.dest)) for edge in sketch.edges
    }
    fallback = ""
    for edge_index, edge in enumerate(sketch.edges):
        if edge.origin > edge.dest:
            continue
        twin = sketch.edges[edge.twin]
        apex_a = sketch.edges[edge.nxt].dest
        apex_b = sketch.edges[twin.nxt].dest
        diagonal = frozenset((apex_a, apex_b))
        left = min(apex_a, apex_b)
        right = max(apex_a, apex_b)
        shared = f"{{{edge.origin},{edge.dest}}}"
        other = f"{{{left},{right}}}"
        sentence = (
            f"edge {shared} is shared by faces {edge.face} and {twin.face}; "
            f"the opposite vertices are {other}"
        )
        if diagonal not in present and left != right:
            return (
                f"{sentence}. That pair is not an edge. A flip would delete "
                f"{shared} and add {other}, rewriting next on those two faces "
                f"and the twins of the new diagonal. A Quad-Edge implementation "
                f"splices Onext instead, and Rot of the edited edge is the dual "
                f"edge that moves with it. This sketch does not rewire."
            )
        if not fallback:
            fallback = (
                f"{sentence}. That pair is already an edge (half-edge "
                f"{edge_index} on this complex). A flip would not produce a "
                f"new simplex. This sketch does not rewire."
            )
    if not fallback:
        raise TeachingMeshError("no undirected edge to describe")
    return fallback


def _arrow(edge: HalfEdge) -> str:
    return f"{edge.origin}->{edge.dest}"


def _endpoint(primal: bool, vertex: int) -> str:
    return f"V{vertex}" if primal else f"F{vertex}"


def _record_name(sketch: QuadEdgeSketch, index: int, start: int) -> str:
    steps = sketch.rot_cycle(start).index(index)
    record = sketch.records[index]
    kind = "primal" if record.primal else "dual"
    if steps == 0:
        role = "e"
    elif steps == 1:
        role = "Rot(e)"
    elif steps == 2:
        role = "Sym(e)=Rot²(e)"
    else:
        role = "Rot³(e)"
    ends = (
        f"{_endpoint(record.primal, record.origin)}->"
        f"{_endpoint(record.primal, record.dest)}"
    )
    return f"{role} [{kind}] {ends}"


def trace_report() -> str:
    """Operator trace for the notebook cell: sphere, torus, and the RP² refusal."""
    sphere = tetrahedron_boundary()
    torus = torus_grid()
    sphere_q = build_quadedge(sphere)
    torus_q = build_quadedge(torus)
    half = sphere_q.half
    seed = _seed_halfedge(half)
    face_ring = half.face_ring(seed)
    vertex_ring = half.vertex_ring(seed)
    primal = sphere_q.record_of_half[seed]
    rot_cycle = sphere_q.rot_cycle(primal)
    dual_start = sphere_q.records[primal].rot
    dual_ring = sphere_q.onext_ring(dual_start)
    lines = [
        "sphere, one face ring via next:",
        "  " + "  ".join(_arrow(half.edges[index]) for index in face_ring),
        "sphere, outgoing edges via next(twin(·)):",
        "  "
        + "  ".join(_arrow(half.edges[index]) for index in vertex_ring)
        + f"  about vertex {half.edges[seed].origin}",
        "sphere, Rot cycle of that primal arrow:",
    ]
    lines.extend(
        "  " + _record_name(sphere_q, index, primal) for index in rot_cycle
    )
    lines.append(
        "sphere, Onext about the dual origin (a primal face): "
        + "  ".join(
            f"F{sphere_q.records[index].origin}->F{sphere_q.records[index].dest}"
            for index in dual_ring
        )
    )
    lines.append(
        "Lnext reproduces next: "
        + str(
            all(
                sphere_q.lnext(sphere_q.record_of_half[index])
                == sphere_q.record_of_half[half.edges[index].nxt]
                for index in range(len(half.edges))
            )
        )
    )
    primal_lengths = sorted(
        {
            len(torus_q.onext_ring(record))
            for record, data in enumerate(torus_q.records)
            if data.primal
        }
    )
    dual_lengths = sorted(
        {
            len(torus_q.onext_ring(record))
            for record, data in enumerate(torus_q.records)
            if not data.primal
        }
    )
    lines.append(
        f"grid torus, Onext cycle lengths: primal {primal_lengths}, dual {dual_lengths}"
    )
    lines.append("flip, tetrahedron boundary: " + flip_sentence(sphere, half))
    lines.append("flip, grid torus: " + flip_sentence(torus, torus_q.half))
    from euler_surface_demo import real_projective_plane

    try:
        build_halfedge(real_projective_plane())
    except TeachingMeshError as err:
        lines.append("RP² half-edge sketch: " + str(err))
    else:
        lines.append("RP² half-edge sketch built; that is a bug")
    return "\n".join(lines)


def _seed_halfedge(sketch: HalfEdgeSketch) -> int:
    for index, edge in enumerate(sketch.edges):
        if (edge.origin, edge.dest) == (0, 1):
            return index
    return 0


_TETRA = {
    0: np.array([1.0, 1.0, 1.0]),
    1: np.array([1.0, -1.0, -1.0]),
    2: np.array([-1.0, 1.0, -1.0]),
    3: np.array([-1.0, -1.0, 1.0]),
}


def _face_centroid(sketch: HalfEdgeSketch, face: int) -> np.ndarray:
    return sum(_TETRA[vertex] for vertex in sketch.faces[face]) / 3.0


def _offset_arrow(origin: int, dest: int, toward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    start = _TETRA[origin]
    end = _TETRA[dest]
    chord = end - start
    mid = 0.5 * (start + end)
    lateral = toward - mid
    norm = float(np.linalg.norm(lateral))
    if norm > 1e-9:
        lateral = lateral / norm * 0.16
    return start + 0.22 * chord + lateral, end - 0.22 * chord + lateral


def _active_indices(mode: str, step: int, quad: QuadEdgeSketch) -> dict[str, int]:
    half = quad.half
    seed = _seed_halfedge(half)
    if mode == "face":
        ring = half.face_ring(seed)
        chosen = ring[step % len(ring)]
    elif mode == "vertex":
        ring = half.vertex_ring(seed)
        chosen = ring[step % len(ring)]
    else:
        chosen = seed
    primal = quad.record_of_half[chosen if mode != "rot" else seed]
    if mode == "rot":
        cycle = quad.rot_cycle(primal)
        highlighted = cycle[step % len(cycle)]
    else:
        highlighted = primal
    return {"half": chosen if mode != "rot" else seed, "record": highlighted, "seed": seed}


def draw_tds(mode: str, step: int):
    """Draw the tetrahedron and the Rot 4-cycle for one operator step.

    `mode` is `face` (`next`), `vertex` (`next` after `twin`), or `rot`.
    Returns the figure and a one-line reading.
    """
    if mode not in {"face", "vertex", "rot"}:
        raise ValueError("mode must be face, vertex, or rot")
    quad = build_quadedge(tetrahedron_boundary())
    active = _active_indices(mode, step, quad)
    half_index = active["half"]
    record_index = active["record"]
    edge = quad.half.edges[half_index]
    record = quad.records[record_index]
    fig = plt.figure(figsize=(10.2, 4.8))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(121, projection="3d", computed_zorder=False)
    side = fig.add_subplot(122)
    _draw_mesh(ax, quad, edge, record)
    _draw_rot_panel(side, quad, record_index)
    reading = _step_reading(mode, quad, half_index, record_index, active["seed"])
    fig.text(0.5, 0.02, reading, ha="center", va="bottom", fontsize=9, color="#333333")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.10, top=0.88, wspace=0.05)
    return fig, reading


def _draw_mesh(ax, quad: QuadEdgeSketch, edge: HalfEdge, record: QuadRecord) -> None:
    complex_ = tetrahedron_boundary()
    tris = [np.vstack([_TETRA[v] for v in sorted(face)]) for face in complex_.facets]
    rgba = (0.84, 0.89, 0.96, 0.28)
    ax.add_collection3d(
        Poly3DCollection(tris, facecolors=[rgba] * len(tris), edgecolors="none")
    )
    for face_edge in complex_.faces_by_dim().get(1, ()):
        a, b = (int(v) for v in sorted(face_edge))
        p, q = _TETRA[a], _TETRA[b]
        ax.plot(
            [p[0], q[0]],
            [p[1], q[1]],
            [p[2], q[2]],
            color=_EDGE,
            lw=1.15,
        )
    toward = _face_centroid(quad.half, edge.face)
    start, end = _offset_arrow(edge.origin, edge.dest, toward)
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        color=_PRIMAL,
        lw=2.4,
    )
    ax.scatter([end[0]], [end[1]], [end[2]], s=28, c=_PRIMAL, depthshade=False)
    if not record.primal:
        left = _face_centroid(quad.half, record.origin)
        right = _face_centroid(quad.half, record.dest)
        ax.plot(
            [left[0], right[0]],
            [left[1], right[1]],
            [left[2], right[2]],
            color=_DUAL,
            lw=2.4,
        )
        ax.scatter([right[0]], [right[1]], [right[2]], s=28, c=_DUAL, depthshade=False)
    center = sum(_TETRA.values()) / 4.0
    for vertex, point in _TETRA.items():
        direction = point - center
        outward = point + 0.28 * direction / np.linalg.norm(direction)
        ax.text(
            outward[0],
            outward[1],
            outward[2],
            str(vertex),
            fontsize=11,
            ha="center",
            va="center",
        )
    ax.view_init(elev=22, azim=-72)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_axis_off()
    ax.set_title("orange: half-edge    blue: Rot onto the dual", fontsize=10)


def _bundle_cycle(quad: QuadEdgeSketch, record_index: int) -> tuple[int, ...]:
    """Rot cycle of one undirected edge, starting at a stable primal arrow.

    The start is the primal record whose ordered endpoints are smaller,
    so the four corners stay put while a slider walks Rot.
    """
    start = record_index
    for _ in range(4):
        if quad.records[start].primal:
            break
        start = quad.records[start].rot
    reverse = quad.records[quad.records[start].rot].rot
    forward_ends = (quad.records[start].origin, quad.records[start].dest)
    reverse_ends = (quad.records[reverse].origin, quad.records[reverse].dest)
    if reverse_ends < forward_ends:
        start = reverse
    return quad.rot_cycle(start)


def _draw_rot_panel(ax, quad: QuadEdgeSketch, highlighted: int) -> None:
    cycle = _bundle_cycle(quad, highlighted)
    labels = ("e\nprimal", "Rot(e)\ndual", "Sym(e)\nprimal reverse", "Rot³(e)\ndual reverse")
    radius = 1.2
    points = []
    for step in range(4):
        angle = math.pi / 2.0 - step * math.pi / 2.0
        points.append(np.array([radius * math.cos(angle), radius * math.sin(angle)]))
    for step, point in enumerate(points):
        nxt = points[(step + 1) % 4]
        gap = 0.34
        direction = nxt - point
        length = float(np.linalg.norm(direction))
        direction = direction / length
        ax.annotate(
            "",
            xy=tuple(nxt - gap * direction),
            xytext=tuple(point + gap * direction),
            arrowprops={"arrowstyle": "->", "color": "#64748b", "lw": 1.15},
        )
    for step, point in enumerate(points):
        index = cycle[step]
        color = _PRIMAL if index == highlighted else "#e2e8f0"
        ax.scatter(
            [point[0]],
            [point[1]],
            s=90,
            c=color,
            zorder=3,
            edgecolors="#334155",
            linewidths=0.6,
        )
        # Labels sit just outside the cycle so they do not cover the arrowheads.
        outward = point / radius * (radius + 0.42)
        ax.text(
            outward[0],
            outward[1],
            labels[step],
            ha="center",
            va="center",
            fontsize=8,
            color="#111827",
            zorder=4,
        )
    ax.text(
        0.0,
        0.0,
        "one\nundirected\nedge",
        ha="center",
        va="center",
        fontsize=8,
        color="#334155",
    )
    ax.set_xlim(-2.15, 2.15)
    ax.set_ylim(-2.05, 2.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title("Rot⁴ = id.  Sym = Rot².  Rot is not twin.", fontsize=10)


def _step_reading(
    mode: str,
    quad: QuadEdgeSketch,
    half_index: int,
    record_index: int,
    seed: int,
) -> str:
    edge = quad.half.edges[half_index]
    record = quad.records[record_index]
    if mode == "face":
        return (
            f"next walks {edge.origin}->{edge.dest} on face {edge.face}. "
            "Three steps close the triangle. twin is not this arrow."
        )
    if mode == "vertex":
        return (
            f"next(twin(·)) walks {edge.origin}->{edge.dest} about vertex {edge.origin}. "
            "The ring is the primal star. It does not visit a dual vertex."
        )
    kind = "primal vertices" if record.primal else "faces"
    ends = (
        f"{_endpoint(record.primal, record.origin)}→"
        f"{_endpoint(record.primal, record.dest)}"
    )
    return (
        f"Rot step {quad.rot_cycle(quad.record_of_half[seed]).index(record_index)} "
        f"of the arrow {quad.half.edges[seed].origin}→{quad.half.edges[seed].dest}: "
        f"{ends} on {kind}."
    )


def launch_surface_tds_demo() -> None:
    """Step `next`, the vertex circulator, or `Rot` on the tetrahedron boundary.

    Call this from a hidden notebook cell. The algebra itself is
    `build_halfedge` and `build_quadedge`; the figure only highlights
    the current arrow.
    """
    busy = {"on": False}
    mode = W.Dropdown(
        options=[
            ("half-edge next (one face)", "face"),
            ("half-edge next(twin) (one vertex)", "vertex"),
            ("Quad-Edge Rot (primal and dual)", "rot"),
        ],
        value="face",
        description="operator",
        style={"description_width": "70px"},
        layout=W.Layout(width="460px"),
    )
    step = W.IntSlider(
        value=0,
        min=0,
        max=2,
        step=1,
        description="step",
        continuous_update=False,
        readout=True,
        style={"description_width": "70px"},
        layout=W.Layout(width="460px"),
    )
    status = W.HTML(value="")
    out = W.Output()

    def _sync_span(name: str) -> None:
        step.max = 3 if name == "rot" else 2
        if step.value > step.max:
            step.value = 0

    def _redraw(_=None) -> None:
        if busy["on"]:
            return
        busy["on"] = True
        try:
            with out:
                clear_output(wait=True)
                fig, reading = draw_tds(mode.value, int(step.value))
                display(fig)
                plt.close(fig)
                print(reading)
            status.value = f"<b>{mode.value}</b> step {int(step.value)}"
        except Exception as err:
            status.value = (
                f"<span style='color:#a00'><b>draw error:</b> {err}</span>"
            )
            with out:
                clear_output(wait=True)
                print("draw error:", err)
        finally:
            busy["on"] = False

    def _on_mode(change) -> None:
        if busy["on"]:
            return
        busy["on"] = True
        try:
            _sync_span(change["new"])
        finally:
            busy["on"] = False
        _redraw()

    hint = W.HTML(
        "Orange is a half-edge, offset into its face so the twin can sit "
        "beside it. Blue appears on a Rot step that has left the primal: "
        "it joins the centroids of the two faces that share the edge. "
        "The right panel is the same undirected edge as a 4-cycle. "
        "Sym reverses the primal arrow; Rot is the dual. They are not "
        "two names for twin."
    )
    ui = W.VBox(
        [
            W.HTML(
                "<b>half-edge and Quad-Edge on the tetrahedron boundary</b> — "
                "edit <code>surface_tds_demo.py</code> to change the pointers."
            ),
            mode,
            step,
            hint,
            status,
            out,
        ]
    )
    _sync_span(mode.value)
    mode.observe(_on_mode, names="value")
    step.observe(_redraw, names="value")
    display(ui)
    _redraw()
