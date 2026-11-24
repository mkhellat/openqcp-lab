# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Draw a small simplicial complex and highlight the star of one simplex.

The notebook cell only calls `launch_complex_draw_demo`. Coordinates,
the star colouring, and the ipywidgets layout live here.

The star of $\\sigma$ is the set of cofaces
$\\{\\tau \\mid \\sigma \\subseteq \\tau\\}$, including $\\sigma$. A face of
$\\sigma$ is not a coface, so a chosen edge is drawn in blue while the
0-simplices at its endpoints stay gray.
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

from lib import SimplicialComplex  # noqa: E402

# Roles follow membership in St(sigma), not geometric incidence.
_STYLE = {
    "chosen": {
        "face": "#1d4e89",
        "edge": "#1d4e89",
        "vertex": "#1d4e89",
        "face_alpha": 0.42,
        "lw": 3.4,
        "size": 70,
    },
    "star": {
        "face": "#ef8a3a",
        "edge": "#c2410c",
        "vertex": "#c2410c",
        "face_alpha": 0.72,
        "lw": 2.6,
        "size": 48,
    },
    "rest": {
        "face": "#e4e4e4",
        "edge": "#8a8a8a",
        "vertex": "#7a7a7a",
        "face_alpha": 0.95,
        "lw": 1.15,
        "size": 32,
    },
}

_KIND = {0: "vertex", 1: "edge", 2: "triangle", 3: "tetrahedron"}


@dataclass(frozen=True)
class _Preset:
    """One abstract complex plus a geometric placement of its vertices."""

    key: str
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
    if sigma not in complex_.all_simplices():
        raise ValueError(f"{key}: default simplex is not a face")
    ordered = tuple((v, coords[v]) for v in sorted(coords))
    return _Preset(key, title, ordered, complex_, sigma, ambient.pop())


def _build_presets() -> dict[str, _Preset]:
    """Four complexes used by the widget, in reading order.

    The book (two triangles on one edge) is the default: the star of
    the shared edge is that edge plus both triangles, which is the
    picture notebook 01 computes in a code cell.
    """
    equilateral = {
        0: (0.0, 0.0),
        1: (2.0, 0.0),
        2: (1.0, 3.0**0.5),
    }
    specs = (
        _preset(
            "triangle",
            "filled triangle",
            equilateral,
            ((0, 1, 2),),
            (0, 1),
        ),
        _preset(
            "book",
            "two triangles on an edge",
            {
                0: (0.0, 0.0),
                1: (2.0, 0.0),
                2: (1.0, 3.0**0.5),
                3: (1.0, -(3.0**0.5)),
            },
            ((0, 1, 2), (0, 1, 3)),
            (0, 1),
        ),
        _preset(
            "pendant",
            "triangle with a pendant edge",
            {
                0: (0.0, 0.0),
                1: (2.0, 0.0),
                2: (1.0, 3.0**0.5),
                3: (1.0, 3.0**0.5 + 1.35),
            },
            ((0, 1, 2), (2, 3)),
            (2,),
        ),
        _preset(
            "tetrahedron",
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
    )
    return {preset.key: preset for preset in specs}


PRESETS: dict[str, _Preset] = _build_presets()


def format_sigma(sigma: frozenset[int]) -> str:
    """Brace notation with vertices in increasing order."""
    return "{" + ",".join(str(v) for v in sorted(sigma)) + "}"


def simplex_label(sigma: frozenset[int]) -> str:
    """Dropdown label: kind, then the vertex set."""
    kind = _KIND.get(len(sigma) - 1, f"{len(sigma) - 1}-simplex")
    return f"{kind} {format_sigma(sigma)}"


def _sorted_simplices(complex_: SimplicialComplex) -> list[frozenset[int]]:
    return sorted(complex_.all_simplices(), key=lambda s: (len(s), sorted(s)))


def _role(
    simplex: frozenset[int],
    sigma: frozenset[int],
    star: frozenset[frozenset[int]],
) -> str:
    if simplex == sigma:
        return "chosen"
    if simplex in star:
        return "star"
    return "rest"


def star_report(complex_: SimplicialComplex, sigma: frozenset[int]) -> str:
    """Plain-text star and 1-skeleton size, for the figure output."""
    star = complex_.star(sigma)
    ordered = sorted(star, key=lambda s: (len(s), sorted(s)))
    adj = complex_.adjacency_list()
    slots = sum(len(nbrs) for nbrs in adj.values())
    dense = len(complex_.vertices) ** 2
    adj_txt = ", ".join(
        f"{v}:{sorted(nbrs)}" for v, nbrs in adj.items()
    )
    lines = [
        f"sigma = {format_sigma(sigma)}    dim {len(sigma) - 1}",
        "St(sigma) = {"
        + ", ".join(format_sigma(s) for s in ordered)
        + "}",
        f"Adj = {{{adj_txt}}}",
        f"neighbour slots {slots}; dense V*V entries {dense}",
    ]
    return "\n".join(lines)


def _ordered_faces(
    faces: frozenset[frozenset[int]] | set[frozenset[int]],
    sigma: frozenset[int],
    star: frozenset[frozenset[int]],
) -> list[frozenset[int]]:
    """Draw the rest, then the star, then the chosen simplex on top."""
    buckets: dict[str, list[frozenset[int]]] = {
        "rest": [],
        "star": [],
        "chosen": [],
    }
    for face in faces:
        buckets[_role(face, sigma, star)].append(face)
    return buckets["rest"] + buckets["star"] + buckets["chosen"]


def _style_for(face: frozenset[int], sigma: frozenset[int], star) -> dict:
    return _STYLE[_role(face, sigma, star)]


def _draw_2d(ax, preset: _Preset, sigma: frozenset[int], star) -> None:
    by_dim = preset.complex.faces_by_dim()
    triangles = _ordered_faces(by_dim.get(2, ()), sigma, star)
    if triangles:
        polys = []
        colors = []
        for face in triangles:
            style = _style_for(face, sigma, star)
            verts = [preset.point(v) for v in sorted(face)]
            polys.append(np.vstack(verts))
            colors.append((*_hex_rgb(style["face"]), style["face_alpha"]))
        ax.add_collection(
            PolyCollection(
                polys,
                facecolors=colors,
                edgecolors="none",
                zorder=1,
            )
        )
    for face in _ordered_faces(by_dim.get(1, ()), sigma, star):
        style = _style_for(face, sigma, star)
        a, b = (preset.point(v) for v in sorted(face))
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color=style["edge"],
            lw=style["lw"],
            solid_capstyle="round",
            zorder=2,
        )
    _scatter_vertices(ax, preset, sigma, star, ambient=2)
    xy = np.vstack([preset.point(v) for v in preset.complex.vertices])
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    pad = 0.22 * np.maximum(hi - lo, 0.4)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def _draw_3d(ax, preset: _Preset, sigma: frozenset[int], star, elev, azim) -> None:
    by_dim = preset.complex.faces_by_dim()
    for face in _ordered_faces(by_dim.get(2, ()), sigma, star):
        style = _style_for(face, sigma, star)
        # Lower alpha than the 2d fills: three faces overlap in projection
        # and would otherwise hide the edges that carry the star colour.
        alpha = 0.50 if face == sigma or face in star else 0.30
        tri = np.vstack([preset.point(v) for v in sorted(face)])
        # Poly3DCollection, not a flat PolyCollection: each triangle
        # has its own z coordinates.
        poly = Poly3DCollection(
            [tri],
            facecolors=[(*_hex_rgb(style["face"]), alpha)],
            edgecolors="none",
            zorder=1,
        )
        ax.add_collection3d(poly)
    for face in _ordered_faces(by_dim.get(1, ()), sigma, star):
        style = _style_for(face, sigma, star)
        a, b = (preset.point(v) for v in sorted(face))
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            [a[2], b[2]],
            color=style["edge"],
            lw=style["lw"] + 0.6,
            solid_capstyle="round",
            zorder=3,
        )
    _scatter_vertices(ax, preset, sigma, star, ambient=3)
    pts = np.vstack([preset.point(v) for v in preset.complex.vertices])
    pad = 0.55
    lo = pts.min(axis=0) - pad
    hi = pts.max(axis=0) + pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def _scatter_vertices(ax, preset, sigma, star, *, ambient: int) -> None:
    by_dim = preset.complex.faces_by_dim()
    faces = _ordered_faces(by_dim.get(0, ()), sigma, star)
    if ambient == 2:
        for face in faces:
            style = _style_for(face, sigma, star)
            v = next(iter(face))
            p = preset.point(v)
            ax.scatter(
                [p[0]],
                [p[1]],
                s=style["size"],
                c=style["vertex"],
                zorder=3,
            )
            ax.annotate(
                str(v),
                xy=(p[0], p[1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=11,
                color="#222222",
                zorder=4,
            )
        return
    center = np.mean(
        [preset.point(u) for u in preset.complex.vertices],
        axis=0,
    )
    for face in faces:
        style = _style_for(face, sigma, star)
        v = next(iter(face))
        p = preset.point(v)
        ax.scatter(
            [p[0]],
            [p[1]],
            [p[2]],
            s=style["size"],
            c=style["vertex"],
            depthshade=False,
            zorder=4,
        )
        direction = p - center
        norm = float(np.linalg.norm(direction))
        outward = p if norm == 0.0 else p + 0.42 * direction / norm
        ax.text(
            outward[0],
            outward[1],
            outward[2],
            str(v),
            fontsize=11,
            color="#222222",
            ha="center",
            va="center",
            zorder=5,
            bbox={
                "boxstyle": "round,pad=0.15",
                "fc": "white",
                "ec": "none",
                "alpha": 0.88,
            },
        )


def _hex_rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def draw_complex_star(
    name: str,
    sigma: frozenset[int],
    *,
    elev: float = 12.0,
    azim: float = -40.0,
):
    """Draw preset `name` and colour the star of `sigma`.

    Returns the matplotlib figure and a plain-text report of the star
    and of the adjacency-list size. The caller displays and closes the
    figure.
    """
    preset = PRESETS[name]
    star = preset.complex.star(sigma)
    fig = plt.figure(figsize=(6.6, 5.4))
    fig.patch.set_facecolor("white")
    if preset.ambient == 3:
        # computed_zorder=False so edges and labels stay above the faces.
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
        _draw_3d(ax, preset, sigma, star, elev, azim)
    else:
        ax = fig.add_subplot(111)
        _draw_2d(ax, preset, sigma, star)
    ax.set_title(
        f"{preset.title}\nstar of {simplex_label(sigma)}",
        fontsize=12,
        pad=8,
    )
    fig.text(
        0.5,
        0.02,
        "blue: chosen simplex    "
        "orange: other simplices in the star    "
        "gray: the rest of the complex",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#333333",
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    return fig, star_report(preset.complex, sigma)


def launch_complex_draw_demo() -> None:
    """Build the ipywidgets UI and paint the book's shared-edge star.

    Choose a complex and a simplex. The figure is one geometric
    realization; orange and blue mark $\\mathrm{St}(\\sigma)$, which
    `SimplicialComplex.star` computes from the facet list.
    """
    busy = {"on": False}
    labels: dict[str, frozenset[int]] = {}

    preset_w = W.Dropdown(
        options=[(p.title, p.key) for p in PRESETS.values()],
        value="book",
        description="complex",
        style={"description_width": "70px"},
        layout=W.Layout(width="460px"),
    )
    simplex_w = W.Dropdown(
        options=[],
        description="simplex",
        style={"description_width": "70px"},
        layout=W.Layout(width="460px"),
    )
    elev_s = W.FloatSlider(
        value=12.0,
        min=-90.0,
        max=90.0,
        step=1.0,
        description="elev",
        continuous_update=False,
        style={"description_width": "40px"},
        layout=W.Layout(width="320px"),
    )
    azim_s = W.FloatSlider(
        value=-40.0,
        min=-180.0,
        max=180.0,
        step=1.0,
        description="azim",
        continuous_update=False,
        style={"description_width": "40px"},
        layout=W.Layout(width="320px"),
    )
    status = W.HTML(value="<i>choose a simplex to paint its star</i>")
    out = W.Output()
    cam_box = W.VBox(
        [
            W.HTML("<b>camera</b> — tetrahedron boundary only"),
            W.HBox([elev_s, azim_s]),
        ]
    )

    def _load_simplices(key: str) -> None:
        preset = PRESETS[key]
        labels.clear()
        options: list[str] = []
        for sigma in _sorted_simplices(preset.complex):
            label = simplex_label(sigma)
            labels[label] = sigma
            options.append(label)
        simplex_w.options = options
        simplex_w.value = simplex_label(preset.default_sigma)

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
            sigma = labels.get(simplex_w.value)
            if sigma is None:
                return
            with out:
                clear_output(wait=True)
                fig, report = draw_complex_star(
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
                f"star of {simplex_w.value}"
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
            _load_simplices(change["new"])
            _sync_camera(change["new"])
        finally:
            busy["on"] = False
        _redraw()

    hint = W.HTML(
        "The star is the set of cofaces of the chosen simplex, "
        "including that simplex. A vertex of a chosen edge is a face "
        "of the edge, so that 0-simplex stays gray while the edge is blue. "
        "The text under the figure is "
        "<code>SimplicialComplex.star</code> and the adjacency list."
    )
    ui = W.VBox(
        [
            W.HTML(
                "<b>star of a simplex</b> — geometric realization of a "
                "small complex. Edit <code>complex_draw_demo.py</code> "
                "to change the drawings."
            ),
            preset_w,
            simplex_w,
            hint,
            cam_box,
            status,
            out,
        ]
    )
    _load_simplices(preset_w.value)
    _sync_camera(preset_w.value)
    preset_w.observe(_on_preset, names="value")
    simplex_w.observe(_redraw, names="value")
    elev_s.observe(_redraw, names="value")
    azim_s.observe(_redraw, names="value")
    display(ui)
    _redraw()
