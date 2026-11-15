"""Interactive cylinder-group addition demo for the manifolds notebook.

Kept beside the notebook so the tutorial cell stays a short launch stub.
"""
from __future__ import annotations

import ast
import operator as op

import ipywidgets as W
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display
from numpy.linalg import norm

_BINOPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow,
}
_UNARY = {ast.UAdd: op.pos, ast.USub: op.neg}
_FUNCS = {
    "sqrt": np.sqrt, "sin": np.sin, "cos": np.cos,
    "tan": np.tan, "abs": abs,
}
_NAMES = {"pi": float(np.pi), "e": float(np.e)}


def _eval_ast(node):
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("only numeric constants allowed")
    if isinstance(node, ast.BinOp):
        fn = _BINOPS.get(type(node.op))
        if fn is None:
            raise ValueError("unsupported operator")
        return float(fn(_eval_ast(node.left), _eval_ast(node.right)))
    if isinstance(node, ast.UnaryOp):
        fn = _UNARY.get(type(node.op))
        if fn is None:
            raise ValueError("unsupported unary op")
        return float(fn(_eval_ast(node.operand)))
    if isinstance(node, ast.Name):
        if node.id in _NAMES:
            return _NAMES[node.id]
        raise ValueError(f"unknown name {node.id!r}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ValueError("only sqrt/sin/cos/tan/abs allowed")
        if node.keywords:
            raise ValueError("keyword args not allowed")
        return float(_FUNCS[node.func.id](*[_eval_ast(a) for a in node.args]))
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


def parse_expr(text):
    s = str(text).strip().replace("^", "**")
    if not s:
        raise ValueError("empty expression")
    return float(_eval_ast(ast.parse(s, mode="eval")))


def wrap_pi(phi):
    return float((phi + np.pi) % (2 * np.pi) - np.pi)


def p_cylinder(u, phi):
    u, phi = np.broadcast_arrays(
        np.asarray(u, dtype=float),
        np.asarray(phi, dtype=float),
    )
    r = 1.0 / np.sqrt(1.0 + u * u)
    return np.stack((r * np.cos(phi), r * np.sin(phi), u * r), axis=-1)


def lift(pts, scale=1.03):
    return np.asarray(pts, dtype=float) * scale


def azim_facing(points):
    # mplot3d azim so the camera looks toward the xy-projection of centroid.
    c = np.sum(np.stack(points, axis=0), axis=0)
    return float(np.degrees(np.arctan2(c[0], -c[1])))


def draw_cylinder_sum(u1, phi1, u2, phi2, elev=22.0, azim=None):
    # Returns a Figure. Caller displays it (do not plt.show / tight_layout).
    u1, u2 = float(u1), float(u2)
    phi1, phi2 = wrap_pi(float(phi1)), wrap_pi(float(phi2))
    u_sum = u1 + u2
    phi_sum = wrap_pi(phi1 + phi2)

    u_max = float(1.0 / np.tan(0.48))
    for name, u in (("u1", u1), ("u2", u2), ("u1+u2", u_sum)):
        if abs(u) > u_max:
            raise ValueError(
                f"{name}={u:.4f} hits a polar cap; keep |u| <= {u_max:.3f}"
            )

    p1 = p_cylinder(u1, phi1)
    p2 = p_cylinder(u2, phi2)
    p_cyl = p_cylinder(u_sum, phi_sum)
    p_R3 = p1 + p2
    p_id = p_cylinder(0.0, 0.0)
    if azim is None:
        azim = azim_facing([p1, p2, p_cyl])

    theta_lo, theta_hi = 0.48, np.pi - 0.48
    phi_g = np.linspace(0.0, 2 * np.pi, 72)
    theta_g = np.linspace(theta_lo, theta_hi, 36)
    PHI, TH = np.meshgrid(phi_g, theta_g)
    X = np.sin(TH) * np.cos(PHI)
    Y = np.sin(TH) * np.sin(PHI)
    Z = np.cos(TH)

    fig = plt.figure(figsize=(7.5, 6.2))
    ax = fig.add_subplot(111, projection="3d")

    # Light translucent belt — coarse mesh; avoid tight_layout (breaks 3d).
    ax.plot_surface(
        X, Y, Z,
        cmap=plt.cm.coolwarm,
        vmin=-1.0,
        vmax=1.0,
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=True,
        alpha=0.15,
        shade=True,
    )

    ph_rim = np.linspace(0.0, 2 * np.pi, 200)
    for th in (theta_lo, theta_hi):
        ax.plot(
            np.sin(th) * np.cos(ph_rim),
            np.sin(th) * np.sin(ph_rim),
            np.cos(th) * np.ones_like(ph_rim),
            color="0.1", lw=1.8, alpha=0.95,
        )

    eq = p_cylinder(0.0, np.linspace(0.0, 2 * np.pi, 180))
    ax.plot(eq[:, 0], eq[:, 1], eq[:, 2], color="0.25", lw=1.1, alpha=0.85)

    for u, phi, c in (
        (u1, phi1, "#1b9e77"),
        (u2, phi2, "#d95f02"),
        (u_sum, phi_sum, "#7570b3"),
    ):
        loc = lift(p_cylinder(u, np.linspace(phi - 0.5, phi + 0.5, 50)))
        ax.plot(loc[:, 0], loc[:, 1], loc[:, 2], color=c, lw=1.2, alpha=0.75, ls="--")
        mer = lift(p_cylinder(np.linspace(0.0, u, 60), phi))
        ax.plot(mer[:, 0], mer[:, 1], mer[:, 2], color=c, lw=1.5, alpha=0.9)
        arc = lift(p_cylinder(u, np.linspace(0.0, phi, 60)))
        ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=c, lw=2.0, alpha=0.95)

    sc = dict(depthshade=False, linewidths=0.7, edgecolors="k")
    ax.scatter(*lift(p_id, 1.025), c="0.15", s=55, **sc)
    ax.scatter(*lift(p1), c="#1b9e77", s=140, **sc, label=r"$p_1$")
    ax.scatter(*lift(p2), c="#d95f02", s=140, **sc, label=r"$p_2$")
    ax.scatter(*lift(p_cyl), c="#7570b3", s=160, marker="D", **sc,
               label=r"cyl sum")
    ax.scatter(*p_R3, c="0.4", s=70, marker="x", depthshade=False,
               label=r"$p_1+p_2$ in $\mathbb{R}^3$")

    ax.text2D(
        0.02, 0.02,
        rf"$p_1=({u1:.2f},{phi1:.2f})$"
        "\n"
        rf"$p_2=({u2:.2f},{phi2:.2f})$"
        "\n"
        rf"cyl $=({u_sum:.2f},{phi_sum:.2f})$"
        "\n"
        rf"elev={elev:.0f}, azim={azim:.0f}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.92),
    )

    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(
        r"$S^2\setminus\{N,S\}\cong\mathbb{R}\times S^1$: "
        r"$(u_1,\phi_1)+(u_2,\phi_2)=(u_1+u_2,\phi_1+\phi_2)$",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.view_init(elev=float(elev), azim=float(azim))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)

    info = {
        "u1": u1, "phi1": phi1, "u2": u2, "phi2": phi2,
        "u_sum": u_sum, "phi_sum": phi_sum,
        "p1": p1, "p2": p2, "p_cyl": p_cyl, "p_R3": p_R3,
        "elev": float(elev), "azim": float(azim),
    }
    return fig, info




def launch_cylinder_group_demo():
    """Build the ipywidgets UI and show an initial snapped figure."""
    _busy = {"on": False}

    mode = W.ToggleButtons(
        options=[("expressions", "expr"), ("sliders", "slide")],
        value="expr",
        description="input",
    )
    u1_t = W.Text(value="0.65", description="u1", layout=W.Layout(width="210px"))
    u2_t = W.Text(value="-0.30", description="u2", layout=W.Layout(width="210px"))
    phi1_t = W.Text(value="-0.55/pi", description="phi1/pi", layout=W.Layout(width="210px"))
    phi2_t = W.Text(value="0.85/pi", description="phi2/pi", layout=W.Layout(width="210px"))
    u_lim = float(1.0 / np.tan(0.48))
    u1_s = W.FloatSlider(value=0.65, min=-u_lim, max=u_lim, step=0.05,
                         description="u1", continuous_update=False)
    u2_s = W.FloatSlider(value=-0.30, min=-u_lim, max=u_lim, step=0.05,
                         description="u2", continuous_update=False)
    phi1_s = W.FloatSlider(value=-0.55, min=-np.pi, max=np.pi, step=0.05,
                           description="phi1", continuous_update=False)
    phi2_s = W.FloatSlider(value=0.85, min=-np.pi, max=np.pi, step=0.05,
                           description="phi2", continuous_update=False)

    # Absolute camera controls (degrees)
    elev_s = W.FloatSlider(
        value=22.0, min=-90.0, max=90.0, step=1.0,
        description="elev", continuous_update=False,
        style={"description_width": "40px"},
        layout=W.Layout(width="320px"),
    )
    azim_s = W.FloatSlider(
        value=-40.0, min=-180.0, max=180.0, step=1.0,
        description="azim", continuous_update=False,
        style={"description_width": "40px"},
        layout=W.Layout(width="320px"),
    )
    follow = W.Checkbox(value=False, description="cam follows cluster")
    snap_btn = W.Button(description="Snap cam", tooltip="Aim at p1,p2,cyl-sum")
    plot_btn = W.Button(description="Plot", button_style="primary")
    status = W.HTML(
        value="<i>phi/pi examples: 2/3, -1/2, sqrt(2)/4, 2^2/3. "
        "Drag elev/azim to orbit.</i>"
    )
    out = W.Output()

    expr_box = W.VBox([W.HBox([u1_t, phi1_t]), W.HBox([u2_t, phi2_t])])
    slide_box = W.VBox([W.HBox([u1_s, phi1_s]), W.HBox([u2_s, phi2_s])])
    slide_box.layout.display = "none"
    cam_box = W.VBox([
        W.HTML("<b>camera</b>"),
        W.HBox([elev_s, azim_s]),
        W.HBox([follow, snap_btn]),
    ])


    def _sync_mode(_=None):
        if mode.value == "expr":
            expr_box.layout.display = None
            slide_box.layout.display = "none"
        else:
            expr_box.layout.display = "none"
            slide_box.layout.display = None


    mode.observe(_sync_mode, names="value")


    def _read_points():
        if mode.value == "expr":
            u1 = parse_expr(u1_t.value)
            u2 = parse_expr(u2_t.value)
            phi1 = parse_expr(phi1_t.value) * np.pi
            phi2 = parse_expr(phi2_t.value) * np.pi
        else:
            u1, u2 = float(u1_s.value), float(u2_s.value)
            phi1, phi2 = float(phi1_s.value), float(phi2_s.value)
        return u1, phi1, u2, phi2


    def _set_sliders_quiet(u1, phi1, u2, phi2):
        with u1_s.hold_trait_notifications(), u2_s.hold_trait_notifications(), \
             phi1_s.hold_trait_notifications(), phi2_s.hold_trait_notifications():
            u1_s.value, u2_s.value = u1, u2
            phi1_s.value, phi2_s.value = wrap_pi(phi1), wrap_pi(phi2)


    def _set_cam_quiet(elev, azim):
        with elev_s.hold_trait_notifications(), azim_s.hold_trait_notifications():
            elev_s.value = float(elev)
            # wrap azim display into [-180, 180]
            a = (float(azim) + 180.0) % 360.0 - 180.0
            azim_s.value = a


    def _snap_cam(_=None):
        try:
            u1, phi1, u2, phi2 = _read_points()
            p1 = p_cylinder(u1, wrap_pi(phi1))
            p2 = p_cylinder(u2, wrap_pi(phi2))
            p_cyl = p_cylinder(u1 + u2, wrap_pi(phi1 + phi2))
            _set_cam_quiet(elev_s.value, azim_facing([p1, p2, p_cyl]))
            _run_plot()
        except Exception as err:
            status.value = f"<span style='color:#a00'><b>snap error:</b> {err}</span>"


    def _run_plot(_=None):
        if _busy["on"]:
            return
        _busy["on"] = True
        try:
            with out:
                clear_output(wait=True)
                try:
                    u1, phi1, u2, phi2 = _read_points()
                    if mode.value == "expr":
                        _set_sliders_quiet(u1, phi1, u2, phi2)
                    else:
                        u1_t.value, u2_t.value = f"{u1:.4g}", f"{u2:.4g}"
                        phi1_t.value = f"({phi1:.4g})/pi"
                        phi2_t.value = f"({phi2:.4g})/pi"

                    if follow.value:
                        p1 = p_cylinder(u1, wrap_pi(phi1))
                        p2 = p_cylinder(u2, wrap_pi(phi2))
                        p_cyl = p_cylinder(u1 + u2, wrap_pi(phi1 + phi2))
                        _set_cam_quiet(elev_s.value, azim_facing([p1, p2, p_cyl]))

                    fig, info = draw_cylinder_sum(
                        u1, phi1, u2, phi2,
                        elev=elev_s.value,
                        azim=azim_s.value,
                    )
                    display(fig)
                    plt.close(fig)
                    print(
                        f"(u1,phi1)=({info['u1']:.3f},{info['phi1']:.3f}) -> "
                        f"{np.round(info['p1'], 4)}"
                    )
                    print(
                        f"(u2,phi2)=({info['u2']:.3f},{info['phi2']:.3f}) -> "
                        f"{np.round(info['p2'], 4)}"
                    )
                    print(
                        f"cyl sum=({info['u_sum']:.3f},{info['phi_sum']:.3f}) -> "
                        f"{np.round(info['p_cyl'], 4)}"
                    )
                    print(
                        f"norms={norm(info['p1']):.4f}, {norm(info['p2']):.4f}, "
                        f"{norm(info['p_cyl']):.4f};  "
                        f"|p1+p2|_R3={norm(info['p_R3']):.4f} (off sphere)"
                    )
                    status.value = (
                        f"<b>ok</b> — cyl sum "
                        f"({info['u_sum']:.3f}, {info['phi_sum']:.3f}); "
                        f"cam elev={info['elev']:.0f}, azim={info['azim']:.0f}"
                    )
                except Exception as err:
                    status.value = (
                        f"<span style='color:#a00'><b>input error:</b> {err}</span>"
                    )
                    print("input error:", err)
        finally:
            _busy["on"] = False


    plot_btn.on_click(_run_plot)
    snap_btn.on_click(_snap_cam)
    for w in (u1_s, u2_s, phi1_s, phi2_s, elev_s, azim_s, follow):
        w.observe(lambda ch: _run_plot(), names="value")

    ui = W.VBox([
        mode, expr_box, slide_box, cam_box,
        W.HBox([plot_btn, status]), out,
    ])
    display(ui)
    _snap_cam()  # initial: snap cam + plot once