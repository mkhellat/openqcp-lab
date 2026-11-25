# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Manifold mesh against two dirty triangle complexes.

The notebook cell only calls `launch_mesh_manifold_demo`. Coordinates
and the ipywidgets layout live here. `mesh_report` prints the same
three complexes without a figure, so a code cell can record the
verdicts.

The three complexes are a facet list each:

- tetrahedron boundary — closed combinatorial surface
- two triangles meeting at a vertex — pinched
- three triangles meeting at one edge — branched

`closed_surface_link_condition` is the manifold test from notebook 02.
Edge incidence is a second count, read off the facets: a closed
surface uses every edge twice. Vertex degree, from the adjacency
list, does not store that count. On the branched example the outer
vertices have degree 2 while the shared edge sits in three triangles.
A half-edge `twin` and a Quad-Edge `Rot` both want a single opposite
face, so both are undefined on that edge. This module does not build
either operator. It records the incidence that would refuse them.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import ipywidgets as W
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_SERIES_DIR = Path(__file__).resolve().parent
if str(_SERIES_DIR) not in sys.path:
    sys.path.insert(0, str(_SERIES_DIR))

from lib import (  # noqa: E402
    SimplicialComplex,
    closed_surface_link_condition,
    euler_characteristic,
    is_disjoint_union_of_cycles,
    vertex_degrees,
)

_FACE_OK = "#d7e3f4"
_FACE_DIRTY = "#fde8d0"
_EDGE = "#4a5568"
_EDGE_BAD = "#c2410c"
_VERTEX = "#1d4e89"
_VERTEX_BAD = "#c2410c"


@dataclass(frozen=True)
class _Preset:
    """One triangle complex and a placement of its vertices."""

    key: str
    menu: str
    title: str
    reading: str
    coords: tuple[tuple[int, tuple[float, ...]], ...]
    complex: SimplicialComplex
    ambient: int
    mark_vertices: frozenset[int]
    mark_edges: frozenset[frozenset[int]]

    def point(self, vertex: int) -> np.ndarray:
        return np.asarray(dict(self.coords)[vertex], dtype=float)


@dataclass(frozen=True)
class MeshFact:
    """Link verdict and incidence for one preset, in stable order."""

    key: str
    title: str
    manifold: bool
    chi: int
    degrees: tuple[tuple[int, int], ...]
    edge_incidences: tuple[tuple[tuple[int, int], int], ...]
    max_edge_incidence: int
    branched_edges: tuple[tuple[int, int], ...]
    failing_vertices: tuple[int, ...]


def _from_facets(facets: tuple[tuple[int, ...], ...]) -> SimplicialComplex:
    frozen = frozenset(frozenset(facet) for facet in facets)
    vertices = frozenset().union(*frozen)
    return SimplicialComplex(vertices, frozen)


def _tetrahedron_boundary() -> SimplicialComplex:
    vertices = (0, 1, 2, 3)
    facets = tuple(tuple(v for v in vertices if v != omit) for omit in vertices)
    return _from_facets(facets)


def _presets() -> tuple[_Preset, ...]:
    tetra = {
        0: (1.0, 1.0, 1.0),
        1: (1.0, -1.0, -1.0),
        2: (-1.0, 1.0, -1.0),
        3: (-1.0, -1.0, 1.0),
    }
    pinch = {
        0: (0.0, 0.0),
        1: (-1.15, 0.72),
        2: (-0.2, 1.25),
        3: (1.15, -0.72),
        4: (0.2, -1.25),
    }
    branch = {
        0: (-1.05, 0.0),
        1: (1.05, 0.0),
        2: (0.0, 1.2),
        3: (-0.72, -1.05),
        4: (0.72, -1.05),
    }
    return (
        _Preset(
            key="manifold",
            menu="manifold mesh (tetrahedron boundary)",
            title="manifold mesh — tetrahedron boundary",
            reading=(
                "Four triangles, each edge in two of them. "
                "Every vertex link is one cycle."
            ),
            coords=tuple(tetra.items()),
            complex=_tetrahedron_boundary(),
            ambient=3,
            mark_vertices=frozenset(),
            mark_edges=frozenset(),
        ),
        _Preset(
            key="pinch",
            menu="pinched vertex (two triangles)",
            title="pinched vertex — two triangles",
            reading=(
                "The triangles meet only at vertex 0. "
                "That link is two disjoint edges."
            ),
            coords=tuple(pinch.items()),
            complex=_from_facets(((0, 1, 2), (0, 3, 4))),
            ambient=2,
            mark_vertices=frozenset({0}),
            mark_edges=frozenset(),
        ),
        _Preset(
            key="branched",
            menu="branched edge (three triangles)",
            title="branched edge — three triangles",
            reading=(
                "Edge {0,1} lies in three triangles. "
                "Outer vertices still have degree 2."
            ),
            coords=tuple(branch.items()),
            complex=_from_facets(((0, 1, 2), (0, 1, 3), (0, 1, 4))),
            ambient=2,
            mark_vertices=frozenset(),
            mark_edges=frozenset({frozenset({0, 1})}),
        ),
    )


_PRESETS = {item.key: item for item in _presets()}
_ORDER = tuple(item.key for item in _presets())


def _edge_incidences(K: SimplicialComplex) -> dict[frozenset[int], int]:
    """How many triangles contain each edge, from the facet list."""
    counts: dict[frozenset[int], int] = {}
    for facet in K.facets:
        verts = sorted(facet)
        if len(verts) != 3:
            raise ValueError("mesh presets are triangle complexes")
        for i, j in ((0, 1), (0, 2), (1, 2)):
            edge = frozenset((verts[i], verts[j]))
            counts[edge] = counts.get(edge, 0) + 1
    return counts


def _single_cycle(link: SimplicialComplex) -> bool:
    """One cycle, not a disjoint union of several."""
    if not is_disjoint_union_of_cycles(link):
        return False
    adj = link.adjacency_list()
    if not adj:
        return False
    seen: set[int] = set()
    stack = [next(iter(adj))]
    while stack:
        vertex = stack.pop()
        if vertex in seen:
            continue
        seen.add(vertex)
        stack.extend(adj[vertex] - seen)
    return seen == set(adj)


def _failing_vertices(K: SimplicialComplex) -> tuple[int, ...]:
    failed = []
    for vertex in sorted(K.vertices):
        if not _single_cycle(K.link(frozenset((vertex,)))):
            failed.append(vertex)
    return tuple(failed)


def _fact(preset: _Preset) -> MeshFact:
    incidences = _edge_incidences(preset.complex)
    ranked = tuple(
        (tuple(sorted(edge)), count)
        for edge, count in sorted(
            incidences.items(),
            key=lambda item: (sorted(item[0]), item[1]),
        )
    )
    branched = tuple(edge for edge, count in ranked if count >= 3)
    degrees = tuple(sorted(vertex_degrees(preset.complex).items()))
    return MeshFact(
        key=preset.key,
        title=preset.title,
        manifold=closed_surface_link_condition(preset.complex),
        chi=euler_characteristic(preset.complex),
        degrees=degrees,
        edge_incidences=ranked,
        max_edge_incidence=max(count for _, count in ranked),
        branched_edges=branched,
        failing_vertices=_failing_vertices(preset.complex),
    )


def example_facts() -> tuple[MeshFact, ...]:
    """Facts for the three presets, checked against the teaching claims.

    The manifold row passes the link condition with every edge in two
    triangles. The pinch fails the link condition with every edge in
    one triangle. The branched row has max incidence 3 on edge (0, 1)
    while vertex 2 still has adjacency-list degree 2.
    """
    facts = tuple(_fact(_PRESETS[key]) for key in _ORDER)
    manifold, pinch, branched = facts
    if facts[0].key != "manifold" or not manifold.manifold:
        raise AssertionError(manifold)
    if manifold.chi != 2 or manifold.max_edge_incidence != 2:
        raise AssertionError(manifold)
    if manifold.failing_vertices or manifold.branched_edges:
        raise AssertionError(manifold)
    if any(degree != 3 for _, degree in manifold.degrees):
        raise AssertionError(manifold.degrees)
    if pinch.manifold or pinch.chi != 1 or pinch.max_edge_incidence != 1:
        raise AssertionError(pinch)
    if pinch.failing_vertices != (0, 1, 2, 3, 4) or pinch.branched_edges:
        raise AssertionError(pinch)
    if dict(pinch.degrees)[0] != 4:
        raise AssertionError(pinch.degrees)
    if branched.manifold or branched.chi != 1:
        raise AssertionError(branched)
    if branched.max_edge_incidence != 3 or branched.branched_edges != ((0, 1),):
        raise AssertionError(branched)
    if dict(branched.degrees)[2] != 2:
        raise AssertionError(branched.degrees)
    if branched.failing_vertices != (0, 1, 2, 3, 4):
        raise AssertionError(branched.failing_vertices)
    return facts


def mesh_report() -> str:
    """Plain-text verdicts for all three presets."""
    lines: list[str] = []
    for fact in example_facts():
        degree_text = ", ".join(f"{vertex}:{degree}" for vertex, degree in fact.degrees)
        incidence_text = ", ".join(
            f"{{{edge[0]},{edge[1]}}}:{count}" for edge, count in fact.edge_incidences
        )
        if fact.failing_vertices:
            failed = ", ".join(str(vertex) for vertex in fact.failing_vertices)
        else:
            failed = "none"
        lines.append(fact.title)
        lines.append(f"  closed_surface_link_condition: {fact.manifold}")
        lines.append(f"  chi: {fact.chi}")
        lines.append(f"  vertex degrees (adjacency list): {degree_text}")
        lines.append(f"  edge incidences (facet list): {incidence_text}")
        lines.append(f"  max edge incidence: {fact.max_edge_incidence}")
        lines.append(f"  vertices whose link is not one cycle: {failed}")
        lines.append("")
    lines.append(
        "Degree 2 on a branched-mesh vertex does not record the edge "
        "that sits in three triangles. twin and Rot want one opposite face."
    )
    return "\n".join(lines).rstrip() + "\n"


def _hex_rgb(color: str) -> tuple[float, float, float]:
    value = color.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _frame(ax, points: list[np.ndarray], ambient: int, *, pad: float) -> None:
    pts = np.vstack(points)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = 0.5 * (lo + hi)
    span = np.maximum(hi - lo, 0.85)
    lo = center - 0.5 * span - pad
    hi = center + 0.5 * span + pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    if ambient == 2:
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        return
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_axis_off()


def _draw_edges(ax, preset: _Preset) -> None:
    ambient = preset.ambient
    for edge in preset.complex.faces_by_dim().get(1, ()):
        a, b = (int(v) for v in sorted(edge))
        marked = edge in preset.mark_edges
        color = _EDGE_BAD if marked else _EDGE
        width = 3.4 if marked else 1.35
        p, q = preset.point(a), preset.point(b)
        coords = [[p[0], q[0]], [p[1], q[1]]]
        if ambient == 3:
            coords.append([p[2], q[2]])
        ax.plot(*coords, color=color, lw=width, solid_capstyle="round", zorder=3)


def _draw_vertices(ax, preset: _Preset) -> None:
    for vertex in sorted(preset.complex.vertices):
        point = preset.point(vertex)
        marked = vertex in preset.mark_vertices
        color = _VERTEX_BAD if marked else _VERTEX
        size = 70 if marked else 32
        coords = [[point[0]], [point[1]]]
        kwargs = {"s": size, "c": color, "zorder": 4}
        if preset.ambient == 3:
            coords.append([point[2]])
            kwargs["depthshade"] = False
        ax.scatter(*coords, **kwargs)
        label_at = point
        if preset.ambient == 3:
            center = sum(preset.point(v) for v in preset.complex.vertices) / len(
                preset.complex.vertices
            )
            direction = point - center
            label_at = point + 0.22 * direction / np.linalg.norm(direction)
        else:
            label_at = point + np.array([0.0, 0.12])
        ax.text(
            *label_at,
            str(vertex),
            fontsize=9,
            color="#222222",
            ha="center",
            va="bottom",
            zorder=5,
        )


def draw_mesh(key: str, *, elev: float = 18.0, azim: float = -52.0):
    """Draw one preset. Returns the figure and that preset's report block."""
    preset = _PRESETS[key]
    fact = _fact(preset)
    fig = plt.figure(figsize=(6.4, 5.2))
    fig.patch.set_facecolor("white")
    if preset.ambient == 3:
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
        ax.view_init(elev=elev, azim=azim)
    else:
        ax = fig.add_subplot(111)
    triangles = [
        np.vstack([preset.point(v) for v in sorted(facet)]) for facet in preset.complex.facets
    ]
    face = _FACE_OK if fact.manifold else _FACE_DIRTY
    rgba = (*_hex_rgb(face), 0.55 if preset.ambient == 3 else 0.95)
    collection_kwargs = {
        "facecolors": [rgba] * len(triangles),
        "edgecolors": "none",
    }
    if preset.ambient == 3:
        ax.add_collection3d(Poly3DCollection(triangles, **collection_kwargs))
    else:
        ax.add_collection(PolyCollection(triangles, **collection_kwargs, zorder=1))
    _draw_edges(ax, preset)
    _draw_vertices(ax, preset)
    _frame(ax, [preset.point(v) for v in preset.complex.vertices], preset.ambient, pad=0.35)
    verdict = "passes" if fact.manifold else "fails"
    ax.set_title(
        f"{preset.title}\nlink condition {verdict}    χ = {fact.chi}",
        fontsize=12,
        color="#222222",
        pad=10,
    )
    fig.text(
        0.5,
        0.02,
        preset.reading,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#333333",
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.86)
    block = mesh_report().split("\n\n")[_ORDER.index(key)]
    return fig, block + "\n"


def launch_mesh_manifold_demo() -> None:
    """Draw the manifold mesh, with a dropdown for the two dirty ones.

    Orange marks the pinched vertex or the branched edge. The status
    line is the link condition. Call this from a hidden notebook cell.
    """
    busy = {"on": False}
    preset_w = W.Dropdown(
        options=[(_PRESETS[key].menu, key) for key in _ORDER],
        value="manifold",
        description="mesh",
        style={"description_width": "52px"},
        layout=W.Layout(width="460px"),
    )
    elev_s = W.FloatSlider(
        value=18.0,
        min=-90.0,
        max=90.0,
        step=1.0,
        description="elev",
        continuous_update=False,
        style={"description_width": "40px"},
        layout=W.Layout(width="300px"),
    )
    azim_s = W.FloatSlider(
        value=-52.0,
        min=-180.0,
        max=180.0,
        step=1.0,
        description="azim",
        continuous_update=False,
        style={"description_width": "40px"},
        layout=W.Layout(width="300px"),
    )
    status = W.HTML(value="")
    out = W.Output()
    cam_box = W.VBox(
        [
            W.HTML("<b>camera</b> — tetrahedron boundary only"),
            W.HBox([elev_s, azim_s]),
        ]
    )

    def _sync_camera(key: str) -> None:
        if _PRESETS[key].ambient == 3:
            cam_box.layout.display = None
        else:
            cam_box.layout.display = "none"

    def _redraw(_=None) -> None:
        if busy["on"]:
            return
        busy["on"] = True
        try:
            key = preset_w.value
            with out:
                clear_output(wait=True)
                fig, report = draw_mesh(
                    key,
                    elev=float(elev_s.value),
                    azim=float(azim_s.value),
                )
                display(fig)
                plt.close(fig)
                print(report)
            fact = _fact(_PRESETS[key])
            verdict = "passes" if fact.manifold else "fails"
            status.value = (
                f"<b>{_PRESETS[key].title}</b> — "
                f"link condition {verdict}, "
                f"max edge incidence {fact.max_edge_incidence}"
            )
        except Exception as err:
            status.value = f"<span style='color:#a00'><b>draw error:</b> {err}</span>"
            with out:
                clear_output(wait=True)
                print("draw error:", err)
        finally:
            busy["on"] = False

    hint = W.HTML(
        "The link condition is <code>closed_surface_link_condition</code>. "
        "Edge incidence is counted from the facet list. Vertex degree is "
        "the adjacency list and does not store how many triangles share "
        "an edge. Orange is the pinch or the branched edge."
    )
    ui = W.VBox(
        [
            W.HTML(
                "<b>Manifold mesh and two dirty complexes</b> — "
                "edit <code>mesh_manifold_demo.py</code> to change the drawings."
            ),
            preset_w,
            hint,
            cam_box,
            status,
            out,
        ]
    )
    _sync_camera(preset_w.value)
    preset_w.observe(_redraw, names="value")
    elev_s.observe(_redraw, names="value")
    azim_s.observe(_redraw, names="value")
    display(ui)
    _redraw()


if __name__ == "__main__":
    print(mesh_report(), end="")
