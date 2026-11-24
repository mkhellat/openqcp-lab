# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Scrub vertices and edges, and draw the link of the chosen simplex.

The notebook cell only calls `launch_link_demo`. Coordinates, the link
colouring, and the ipywidgets layout live here.

Two panels: the complex, with the chosen simplex in blue and its link
in orange, and the link drawn on its own. An edge that contains the
chosen vertex is a coface of that vertex. It belongs to the star from
notebook 01, and it is not a simplex of the link, so it stays gray
unless that edge is itself the simplex being scrubbed.

Presets, in dropdown order:

- the boundary of a tetrahedron, a combinatorial 2-sphere
- two triangles that meet only at a vertex (pinched)
- a 5-cycle, a closed combinatorial 1-manifold
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
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_SERIES_DIR = Path(__file__).resolve().parent
if str(_SERIES_DIR) not in sys.path:
    sys.path.insert(0, str(_SERIES_DIR))

from lib import (  # noqa: E402
    SimplicialComplex,
    closed_surface_link_condition,
    is_disjoint_union_of_cycles,
    links_are_circles,
    vertex_degrees,
)

_STYLE = {
    "chosen": {"edge": "#1d4e89", "vertex": "#1d4e89", "lw": 3.6, "size": 78},
    "link": {"edge": "#c2410c", "vertex": "#c2410c", "lw": 2.8, "size": 64},
    "rest": {"edge": "#8a8a8a", "vertex": "#8a8a8a", "lw": 1.15, "size": 28},
}
_FACE = "#e6e6e6"
_FACE_ALPHA_2D = 0.95
_FACE_ALPHA_3D = 0.28
_KIND = {0: "vertex", 1: "edge"}


@dataclass(frozen=True)
class _Preset:
    """One abstract complex plus a geometric placement of its vertices."""

    key: str
    menu: str
    title: str
    coords: tuple[tuple[int, tuple[float, ...]], ...]
    complex: SimplicialComplex
    default_sigma: frozenset[int]
    ambient: int

    def point(self, vertex: int) -> np.ndarray:
        return np.asarray(dict(self.coords)[vertex], dtype=float)


def _complex_from_facets(
    facets: tuple[tuple[int, ...], ...],
) -> SimplicialComplex:
    frozen = frozenset(frozenset(facet) for facet in facets)
    vertices = frozenset().union(*frozen)
    return SimplicialComplex(vertices, frozen)


def _preset(
    key: str,
    menu: str,
    title: str,
    coords: dict[int, tuple[float, ...]],
    facets: tuple[tuple[int, ...], ...],
    default: tuple[int, ...],
) -> _Preset:
    complex_ = _complex_from_facets(facets)
    if set(coords) != set(complex_.vertices):
        raise ValueError(f"{key}: coordinates must match the vertex set")
    ambient = {len(xyz) for xyz in coords.values()}
    if ambient not in ({2}, {3}):
        raise ValueError(f"{key}: coordinates must lie in R^2 or R^3")
    sigma = frozenset(default)
    if sigma not in complex_.all_simplices() or len(sigma) > 2:
        raise ValueError(f"{key}: default must be a vertex or an edge")
    ordered = tuple((v, coords[v]) for v in sorted(coords))
    return _Preset(key, menu, title, ordered, complex_, sigma, ambient.pop())


def _pentagon() -> dict[int, tuple[float, float]]:
    coords = {}
    for i in range(5):
        theta = math.pi / 2 + 2.0 * math.pi * i / 5.0
        coords[i] = (math.cos(theta), math.sin(theta))
    return coords


def _build_presets() -> dict[str, _Preset]:
    """Good surface, pinched surface, closed 1-manifold."""
    specs = (
        _preset(
            "sphere",
            "tetrahedron boundary (links are spheres)",
            "tetrahedron boundary",
            {
                0: (1.0, 1.0, 1.0),
                1: (1.0, -1.0, -1.0),
                2: (-1.0, 1.0, -1.0),
                3: (-1.0, -1.0, 1.0),
            },
            ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
            (0,),
        ),
        _preset(
            "pinched",
            "pinched triangles (shared vertex)",
            "pinched triangles",
            {
                0: (0.0, 0.0),
                1: (-1.15, 0.9),
                2: (-1.15, -0.9),
                3: (1.15, 0.9),
                4: (1.15, -0.9),
            },
            ((0, 1, 2), (0, 3, 4)),
            (0,),
        ),
        _preset(
            "cycle",
            "cycle C5 (closed 1-manifold)",
            "cycle C5",
            _pentagon(),
            tuple((i, (i + 1) % 5) for i in range(5)),
            (0,),
        ),
    )
    return {preset.key: preset for preset in specs}


PRESETS: dict[str, _Preset] = _build_presets()


def format_sigma(sigma: frozenset[int]) -> str:
    """Brace notation with vertices in increasing order."""
    return "{" + ",".join(str(v) for v in sorted(sigma)) + "}"


def simplex_label(sigma: frozenset[int]) -> str:
    """Kind, then the vertex set. The scrubber only offers vertices and edges."""
    kind = _KIND.get(len(sigma) - 1, f"{len(sigma) - 1}-simplex")
    return f"{kind} {format_sigma(sigma)}"


def _scrub_simplices(complex_: SimplicialComplex) -> list[frozenset[int]]:
    """Vertices, then edges, each block in lexicographic order."""
    by_dim = complex_.faces_by_dim()
    ordered: list[frozenset[int]] = []
    for dim in (0, 1):
        block = sorted(by_dim.get(dim, ()), key=lambda face: sorted(face))
        ordered.extend(block)
    return ordered


def _component_count(complex_: SimplicialComplex) -> int:
    """Components of the 1-skeleton, walked on its adjacency list."""
    adj = complex_.adjacency_list()
    unseen = set(complex_.vertices)
    count = 0
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            vertex = stack.pop()
            for neighbour in adj[vertex]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
    return count


def _is_s0(link: SimplicialComplex) -> bool:
    """Two points and no edge: the PL 0-sphere."""
    return (
        link.dimension() == 0
        and len(link.vertices) == 2
        and all(len(facet) == 1 for facet in link.facets)
    )


def _is_one_cycle(link: SimplicialComplex) -> bool:
    """A single cycle. Disjoint unions fail the 'one sphere' reading."""
    if not is_disjoint_union_of_cycles(link):
        return False
    return _component_count(link) == 1


def _is_path(link: SimplicialComplex) -> bool:
    """One path of edges: a PL 1-ball, including a single edge."""
    if link.dimension() != 1 or not link.is_pure(1):
        return False
    if _component_count(link) != 1:
        return False
    degrees = vertex_degrees(link)
    endpoints = [degree for degree in degrees.values() if degree == 1]
    if len(endpoints) != 2:
        return False
    return all(degree in (1, 2) for degree in degrees.values())


def _sphere_line(link: SimplicialComplex, expected: int) -> str:
    """Tutorial reading of one link. Spheres of dimension > 1 are deferred."""
    actual = link.dimension()
    if expected == -1 and actual == -1 and not link.vertices:
        return "reading: empty complex, the PL sphere of dimension -1"
    if expected == -1:
        return "reading: link is occupied, so it is not the (-1)-sphere"
    if expected == 0 and _is_s0(link):
        return "reading: two points and no edge, a PL 0-sphere"
    if expected == 0 and actual == 0 and len(link.vertices) == 1:
        return "reading: one point, a PL 0-ball, not a 0-sphere"
    if expected == 0:
        return (
            f"reading: {len(link.vertices)} vertices in a {actual}-complex, "
            "not a PL 0-sphere"
        )
    if expected == 1 and _is_one_cycle(link):
        return "reading: one cycle, a combinatorial 1-sphere"
    if expected == 1 and _is_path(link):
        return "reading: one path, a PL 1-ball, not a 1-sphere"
    if expected == 1:
        count = _component_count(link)
        noun = "component" if count == 1 else "components"
        return f"reading: {count} {noun}, not a PL 1-sphere"
    if actual != expected:
        return (
            f"reading: link dimension {actual}, "
            f"and a PL {expected}-sphere has dimension {expected}"
        )
    return (
        f"reading: a PL {expected}-sphere is not decided in this demo "
        "(notebooks 04 and 05)"
    )


def link_report(complex_: SimplicialComplex, sigma: frozenset[int]) -> str:
    """Plain-text link, adjacency row, and a low-dimensional sphere reading."""
    link = complex_.link(sigma)
    dim_sigma = len(sigma) - 1
    parent_dim = complex_.dimension()
    expected = parent_dim - dim_sigma - 1
    facets = sorted(tuple(sorted(facet)) for facet in link.facets)
    lines = [
        f"sigma = {format_sigma(sigma)}    dim {dim_sigma}",
        (
            f"parent dim {parent_dim}    "
            f"link dim {link.dimension()}    "
            f"expected {parent_dim} - {dim_sigma} - 1 = {expected}"
        ),
    ]
    if len(sigma) == 1:
        vertex = next(iter(sigma))
        neighbours = sorted(complex_.adjacency_list()[vertex])
        lines.append(
            f"Adj({vertex}) = {neighbours}    "
            f"degree {len(neighbours)}    "
            f"link vertices {sorted(link.vertices)}"
        )
    else:
        lines.append(f"link vertices {sorted(link.vertices)}")
    lines.append(f"link facets {facets}")
    lines.append(_sphere_line(link, expected))
    if complex_.dimension() == 1 and complex_.is_pure(1):
        lines.append(f"links_are_circles {links_are_circles(complex_)}")
    elif complex_.dimension() == 2 and complex_.is_pure(2):
        lines.append(
            "closed_surface_link_condition "
            f"{closed_surface_link_condition(complex_)}"
        )
    return "\n".join(lines)


def _hex_rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _vertex_role(
    vertex: int,
    sigma: frozenset[int],
    link_vertices: frozenset[int],
) -> str:
    if vertex in sigma:
        return "chosen"
    if vertex in link_vertices:
        return "link"
    return "rest"


def _edge_role(
    edge: frozenset[int],
    sigma: frozenset[int],
    link_edges: frozenset[frozenset[int]],
) -> str:
    if edge == sigma:
        return "chosen"
    if edge in link_edges:
        return "link"
    return "rest"


def _ordered_by_role(items, role_of) -> list:
    buckets = {"rest": [], "link": [], "chosen": []}
    for item in items:
        buckets[role_of(item)].append(item)
    return buckets["rest"] + buckets["link"] + buckets["chosen"]


def _fill_triangles_2d(ax, preset: _Preset, faces) -> None:
    if not faces:
        return
    polys = [np.vstack([preset.point(v) for v in sorted(face)]) for face in faces]
    rgba = (*_hex_rgb(_FACE), _FACE_ALPHA_2D)
    ax.add_collection(
        PolyCollection(
            polys,
            facecolors=[rgba] * len(polys),
            edgecolors="none",
            zorder=1,
        )
    )


def _fill_triangles_3d(ax, preset: _Preset, faces) -> None:
    if not faces:
        return
    tris = [np.vstack([preset.point(v) for v in sorted(face)]) for face in faces]
    rgba = (*_hex_rgb(_FACE), _FACE_ALPHA_3D)
    ax.add_collection3d(
        Poly3DCollection(
            tris,
            facecolors=[rgba] * len(tris),
            edgecolors="none",
        )
    )


def _plot_segment(ax, p, q, *, color: str, lw: float, ambient: int, zorder: int) -> None:
    if ambient == 2:
        ax.plot(
            [p[0], q[0]],
            [p[1], q[1]],
            color=color,
            lw=lw,
            solid_capstyle="round",
            zorder=zorder,
        )
        return
    ax.plot(
        [p[0], q[0]],
        [p[1], q[1]],
        [p[2], q[2]],
        color=color,
        lw=lw,
        solid_capstyle="round",
        zorder=zorder,
    )


def _scatter_vertex(ax, point, *, color: str, size: float, ambient: int) -> None:
    if ambient == 2:
        ax.scatter([point[0]], [point[1]], s=size, c=color, zorder=4)
        return
    ax.scatter(
        [point[0]],
        [point[1]],
        [point[2]],
        s=size,
        c=color,
        depthshade=False,
        zorder=5,
    )


def _label_vertex(ax, preset: _Preset, vertex: int, *, ambient: int, center: np.ndarray) -> None:
    point = preset.point(vertex)
    if ambient == 2:
        ax.annotate(
            str(vertex),
            xy=(point[0], point[1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=11,
            color="#222222",
            zorder=6,
        )
        return
    direction = point - center
    norm = float(np.linalg.norm(direction))
    outward = point if norm == 0.0 else point + 0.28 * direction / norm
    ax.text(
        outward[0],
        outward[1],
        outward[2],
        str(vertex),
        fontsize=11,
        color="#222222",
        ha="center",
        va="center",
        zorder=6,
    )


def _frame(ax, points: list[np.ndarray], ambient: int, *, pad: float) -> None:
    pts = np.vstack(points)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = 0.5 * (lo + hi)
    span = np.maximum(hi - lo, 0.85)
    lo = center - 0.5 * span - pad
    hi = center + 0.5 * span + pad
    if ambient == 2:
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        return
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_axis_off()


def _draw_edges(ax, preset: _Preset, edges, *, ambient: int, role_of) -> None:
    ordered = _ordered_by_role(edges, role_of)
    for edge in ordered:
        role = role_of(edge)
        style = _STYLE[role]
        a, b = (preset.point(v) for v in sorted(edge))
        _plot_segment(
            ax,
            a,
            b,
            color=style["edge"],
            lw=style["lw"],
            ambient=ambient,
            zorder=3 if role == "rest" else 4,
        )


def _draw_vertices(ax, preset: _Preset, vertices, *, ambient: int, role_of, center: np.ndarray) -> None:
    ordered = _ordered_by_role(vertices, role_of)
    for vertex in ordered:
        style = _STYLE[role_of(vertex)]
        _scatter_vertex(
            ax,
            preset.point(vertex),
            color=style["vertex"],
            size=style["size"],
            ambient=ambient,
        )
        _label_vertex(ax, preset, vertex, ambient=ambient, center=center)


def _centroid(preset: _Preset, vertices) -> np.ndarray:
    if not vertices:
        return np.zeros(preset.ambient)
    return np.mean([preset.point(v) for v in vertices], axis=0)


def _draw_parent(ax, preset: _Preset, sigma, link, elev: float, azim: float) -> None:
    ambient = preset.ambient
    by_dim = preset.complex.faces_by_dim()
    link_edges = link.faces_by_dim().get(1, frozenset())
    if ambient == 3:
        _fill_triangles_3d(ax, preset, by_dim.get(2, ()))
    else:
        _fill_triangles_2d(ax, preset, by_dim.get(2, ()))
    _draw_edges(
        ax,
        preset,
        by_dim.get(1, ()),
        ambient=ambient,
        role_of=lambda edge: _edge_role(edge, sigma, link_edges),
    )
    center = _centroid(preset, preset.complex.vertices)
    _draw_vertices(
        ax,
        preset,
        sorted(preset.complex.vertices),
        ambient=ambient,
        role_of=lambda vertex: _vertex_role(vertex, sigma, link.vertices),
        center=center,
    )
    points = [preset.point(v) for v in preset.complex.vertices]
    _frame(ax, points, ambient, pad=0.35)
    if ambient == 3:
        ax.view_init(elev=elev, azim=azim)
    ax.set_title(preset.title, fontsize=12, pad=8)


class _LinkPlane:
    """Planar coordinates for the link panel.

    A 2-complex keeps its picture. A link drawn from an R^3 complex is
    projected onto the best-fit plane of its vertices, so a 3-cycle
    stays a triangle instead of an edge-on segment.
    """

    def __init__(self, coords: dict[int, np.ndarray]) -> None:
        self.coords = coords
        self.ambient = 2

    def point(self, vertex: int) -> np.ndarray:
        return self.coords[vertex]


def _link_plane(preset: _Preset, vertices) -> _LinkPlane:
    ids = sorted(vertices)
    if preset.ambient == 2:
        return _LinkPlane({v: preset.point(v) for v in ids})
    if len(ids) == 1:
        return _LinkPlane({ids[0]: np.zeros(2)})
    pts = np.vstack([preset.point(v) for v in ids])
    centered = pts - pts.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    rank = max(1, int(np.sum(singular > 1e-8)))
    flat = centered @ vt[:rank].T
    if flat.shape[1] == 1:
        flat = np.column_stack([flat[:, 0], np.zeros(len(ids))])
    return _LinkPlane({v: flat[i] for i, v in enumerate(ids)})


def _draw_link_panel(ax, preset: _Preset, sigma, link, expected: int) -> None:
    ax.set_title(f"link of {simplex_label(sigma)}", fontsize=12, pad=8)
    if not link.vertices:
        message = "empty link"
        if expected == -1:
            message = "empty link\nPL sphere of dimension -1"
        ax.text(
            0.5,
            0.5,
            message,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="#333333",
        )
        ax.set_axis_off()
        return
    plane = _link_plane(preset, link.vertices)
    link_edges = link.faces_by_dim().get(1, ())
    _draw_edges(
        ax,
        plane,
        link_edges,
        ambient=2,
        role_of=lambda _edge: "link",
    )
    center = _centroid(plane, link.vertices)
    _draw_vertices(
        ax,
        plane,
        sorted(link.vertices),
        ambient=2,
        role_of=lambda _vertex: "link",
        center=center,
    )
    _frame(ax, [plane.point(v) for v in link.vertices], 2, pad=0.45)


def draw_link(
    name: str,
    sigma: frozenset[int],
    *,
    elev: float = 18.0,
    azim: float = -52.0,
):
    """Draw preset `name` and the link of `sigma`.

    Returns the matplotlib figure and the plain-text reading. The
    caller displays and closes the figure. `sigma` is a vertex or an
    edge of the preset.
    """
    preset = PRESETS[name]
    if sigma not in preset.complex.all_simplices():
        raise ValueError(f"{format_sigma(sigma)} is not a simplex of {name}")
    link = preset.complex.link(sigma)
    expected = preset.complex.dimension() - (len(sigma) - 1) - 1
    fig = plt.figure(figsize=(10.4, 4.8))
    fig.patch.set_facecolor("white")
    if preset.ambient == 3:
        ax_l = fig.add_subplot(121, projection="3d", computed_zorder=False)
    else:
        ax_l = fig.add_subplot(121)
    # The link is at most a 1-complex here. Draw it in its own plane
    # so the tetrahedron's opposite 3-cycle is not seen edge-on.
    ax_r = fig.add_subplot(122)
    _draw_parent(ax_l, preset, sigma, link, elev, azim)
    _draw_link_panel(ax_r, preset, sigma, link, expected)
    fig.text(
        0.5,
        0.02,
        "blue: chosen simplex    orange: its link    "
        "gray: the rest, including cofaces that properly contain the choice",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#333333",
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.10, top=0.88, wspace=0.12)
    return fig, link_report(preset.complex, sigma)


def launch_link_demo() -> None:
    """Build the ipywidgets scrubber and draw the tetrahedron's vertex link.

    The slider walks vertices, then edges. The figure is one geometric
    realization; orange marks the link, which `SimplicialComplex.link`
    computes from the facet list. Call this from a hidden notebook cell.
    """
    busy = {"on": False}
    simplices: list[frozenset[int]] = []

    preset_w = W.Dropdown(
        options=[(p.menu, p.key) for p in PRESETS.values()],
        value="sphere",
        description="complex",
        style={"description_width": "70px"},
        layout=W.Layout(width="520px"),
    )
    scrub = W.IntSlider(
        value=0,
        min=0,
        max=1,
        step=1,
        description="scrub",
        continuous_update=False,
        readout=False,
        style={"description_width": "70px"},
        layout=W.Layout(width="520px"),
    )
    readout = W.HTML(value="")
    elev_s = W.FloatSlider(
        value=18.0,
        min=-90.0,
        max=90.0,
        step=1.0,
        description="elev",
        continuous_update=False,
        style={"description_width": "40px"},
        layout=W.Layout(width="320px"),
    )
    azim_s = W.FloatSlider(
        value=-52.0,
        min=-180.0,
        max=180.0,
        step=1.0,
        description="azim",
        continuous_update=False,
        style={"description_width": "40px"},
        layout=W.Layout(width="320px"),
    )
    status = W.HTML(value="<i>scrub a vertex or an edge to draw its link</i>")
    out = W.Output()
    cam_box = W.VBox(
        [
            W.HTML("<b>camera</b> — tetrahedron boundary only"),
            W.HBox([elev_s, azim_s]),
        ]
    )

    def _sync_readout() -> None:
        if not simplices:
            readout.value = ""
            return
        index = int(scrub.value)
        sigma = simplices[index]
        readout.value = (
            f"<b>{simplex_label(sigma)}</b> "
            f"({index + 1}/{len(simplices)})"
        )

    def _load(key: str) -> None:
        preset = PRESETS[key]
        simplices.clear()
        simplices.extend(_scrub_simplices(preset.complex))
        scrub.value = 0
        scrub.max = max(len(simplices) - 1, 0)
        scrub.value = simplices.index(preset.default_sigma)
        _sync_readout()

    def _sync_camera(key: str) -> None:
        if PRESETS[key].ambient == 3:
            cam_box.layout.display = None
        else:
            cam_box.layout.display = "none"

    def _redraw(_=None) -> None:
        if busy["on"]:
            return
        busy["on"] = True
        try:
            if not simplices:
                return
            _sync_readout()
            sigma = simplices[int(scrub.value)]
            with out:
                clear_output(wait=True)
                fig, report = draw_link(
                    preset_w.value,
                    sigma,
                    elev=float(elev_s.value),
                    azim=float(azim_s.value),
                )
                display(fig)
                plt.close(fig)
                print(report)
            status.value = (
                f"<b>ok</b> — {PRESETS[preset_w.value].title}, "
                f"link of {simplex_label(sigma)}"
            )
        except Exception as err:
            status.value = (
                f"<span style='color:#a00'><b>draw error:</b> {err}</span>"
            )
            with out:
                clear_output(wait=True)
                print("draw error:", err)
        finally:
            busy["on"] = False

    def _on_preset(change) -> None:
        if busy["on"]:
            return
        busy["on"] = True
        try:
            _load(change["new"])
            _sync_camera(change["new"])
        finally:
            busy["on"] = False
        _redraw()

    hint = W.HTML(
        "Orange is the link: delete the chosen simplex from each "
        "simplex that properly contains it. The right panel draws that "
        "link in its own plane. Edges that merely touch the chosen "
        "vertex stay gray; those edges are cofaces, the star from "
        "notebook 01. The text under the figure is "
        "<code>SimplicialComplex.link</code> and, for a vertex, the "
        "adjacency-list row."
    )
    ui = W.VBox(
        [
            W.HTML(
                "<b>link of a simplex</b> — scrub vertices, then edges. "
                "Edit <code>link_demo.py</code> to change the drawings."
            ),
            preset_w,
            scrub,
            readout,
            hint,
            cam_box,
            status,
            out,
        ]
    )
    _load(preset_w.value)
    _sync_camera(preset_w.value)
    preset_w.observe(_on_preset, names="value")
    scrub.observe(_redraw, names="value")
    elev_s.observe(_redraw, names="value")
    azim_s.observe(_redraw, names="value")
    display(ui)
    _redraw()
