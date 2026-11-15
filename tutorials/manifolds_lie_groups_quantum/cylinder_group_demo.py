"""Interactive cylinder-group addition demo for the manifolds notebook.

Kept beside the notebook so the tutorial cell stays a short launch stub.

Supports a static sum and an animate mode: set ranges in θ (polar) and/or
φ, then scrub frames with a Play control like a video player.
"""
from __future__ import annotations

import ast
import operator as op

import ipywidgets as W
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display
from numpy.linalg import norm

# Open-belt polar caps used by the surface mesh (must match draw_*).
_THETA_LO = 0.48
_THETA_HI = float(np.pi - 0.48)
_U_MAX = float(1.0 / np.tan(_THETA_LO))

_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
}
_UNARY = {ast.UAdd: op.pos, ast.USub: op.neg}
_FUNCS = {
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "abs": abs,
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


def u_from_theta(theta):
    """u = cot θ for θ ∈ (0, π)."""
    th = np.asarray(theta, dtype=float)
    return 1.0 / np.tan(th)


def theta_from_u(u):
    """θ = arccot(u) ∈ (0, π)."""
    return float(np.arctan2(1.0, float(u)))


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
    c = np.sum(np.stack(points, axis=0), axis=0)
    return float(np.degrees(np.arctan2(c[0], -c[1])))


def _check_u(name, u):
    if abs(u) > _U_MAX:
        raise ValueError(
            f"{name}={u:.4f} hits a polar cap; keep |u| <= {_U_MAX:.3f} "
            f"(θ in ({_THETA_LO:.2f}, {_THETA_HI:.2f}))"
        )


def _check_theta(name, theta):
    if not (_THETA_LO < theta < _THETA_HI):
        raise ValueError(
            f"{name}={theta:.4f} outside open belt "
            f"({_THETA_LO:.2f}, {_THETA_HI:.2f})"
        )


def lerp(a, b, t):
    return (1.0 - t) * a + t * b


def draw_cylinder_sum(
    u1,
    phi1,
    u2,
    phi2,
    elev=22.0,
    azim=None,
    trails=None,
    frame_label=None,
):
    """Return (fig, info). Caller displays the figure (no tight_layout).

    trails: optional dict with keys p1, p2, p_cyl — each an (N, 3) array.
    """
    u1, u2 = float(u1), float(u2)
    phi1, phi2 = wrap_pi(float(phi1)), wrap_pi(float(phi2))
    u_sum = u1 + u2
    phi_sum = wrap_pi(phi1 + phi2)
    for name, u in (("u1", u1), ("u2", u2), ("u1+u2", u_sum)):
        _check_u(name, u)

    th1, th2, th_sum = theta_from_u(u1), theta_from_u(u2), theta_from_u(u_sum)
    p1 = p_cylinder(u1, phi1)
    p2 = p_cylinder(u2, phi2)
    p_cyl = p_cylinder(u_sum, phi_sum)
    p_R3 = p1 + p2
    p_id = p_cylinder(0.0, 0.0)
    if azim is None:
        azim = azim_facing([p1, p2, p_cyl])

    phi_g = np.linspace(0.0, 2 * np.pi, 72)
    theta_g = np.linspace(_THETA_LO, _THETA_HI, 36)
    PHI, TH = np.meshgrid(phi_g, theta_g)
    X = np.sin(TH) * np.cos(PHI)
    Y = np.sin(TH) * np.sin(PHI)
    Z = np.cos(TH)

    fig = plt.figure(figsize=(7.5, 6.2))
    ax = fig.add_subplot(111, projection="3d")
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
    for th in (_THETA_LO, _THETA_HI):
        ax.plot(
            np.sin(th) * np.cos(ph_rim),
            np.sin(th) * np.sin(ph_rim),
            np.cos(th) * np.ones_like(ph_rim),
            color="0.1",
            lw=1.8,
            alpha=0.95,
        )

    eq = p_cylinder(0.0, np.linspace(0.0, 2 * np.pi, 180))
    ax.plot(eq[:, 0], eq[:, 1], eq[:, 2], color="0.25", lw=1.1, alpha=0.85)

    if trails:
        for key, color in (
            ("p1", "#1b9e77"),
            ("p2", "#d95f02"),
            ("p_cyl", "#7570b3"),
        ):
            tr = np.asarray(trails.get(key, []), dtype=float)
            if tr.ndim == 2 and len(tr) >= 2:
                ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=color, lw=1.0, alpha=0.45)

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
    ax.scatter(
        *lift(p_cyl),
        c="#7570b3",
        s=160,
        marker="D",
        **sc,
        label=r"cyl sum",
    )
    ax.scatter(
        *p_R3,
        c="0.4",
        s=70,
        marker="x",
        depthshade=False,
        label=r"$p_1+p_2$ in $\mathbb{R}^3$",
    )

    frame_line = f"\n{frame_label}" if frame_label else ""
    ax.text2D(
        0.02,
        0.02,
        rf"$p_1$: $(u,\phi)=({u1:.2f},{phi1:.2f})$, $\theta={th1:.2f}$"
        "\n"
        rf"$p_2$: $(u,\phi)=({u2:.2f},{phi2:.2f})$, $\theta={th2:.2f}$"
        "\n"
        rf"cyl: $({u_sum:.2f},{phi_sum:.2f})$, $\theta={th_sum:.2f}$"
        "\n"
        rf"elev={elev:.0f}, azim={azim:.0f}"
        f"{frame_line}",
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
        r"$(u_1,\phi_1)+(u_2,\phi_2)=(u_1+u_2,\phi_1+\phi_2)$"
        "\n"
        r"$u=\cot\theta$",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.view_init(elev=float(elev), azim=float(azim))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.90)

    info = {
        "u1": u1,
        "phi1": phi1,
        "u2": u2,
        "phi2": phi2,
        "u_sum": u_sum,
        "phi_sum": phi_sum,
        "theta1": th1,
        "theta2": th2,
        "theta_sum": th_sum,
        "p1": p1,
        "p2": p2,
        "p_cyl": p_cyl,
        "p_R3": p_R3,
        "elev": float(elev),
        "azim": float(azim),
    }
    return fig, info


def build_animation_frames(
    n_frames,
    *,
    theta1_0,
    theta1_1,
    phi1_0,
    phi1_1,
    theta2_0,
    theta2_1,
    phi2_0,
    phi2_1,
    anim_theta1,
    anim_phi1,
    anim_theta2,
    anim_phi2,
):
    """Return list of (u1, phi1, u2, phi2) and trail dict of xyz paths."""
    if n_frames < 2:
        raise ValueError("need at least 2 frames")
    for name, th in (
        ("θ1 start", theta1_0),
        ("θ1 end", theta1_1),
        ("θ2 start", theta2_0),
        ("θ2 end", theta2_1),
    ):
        _check_theta(name, th)

    frames = []
    trails = {"p1": [], "p2": [], "p_cyl": []}
    for k in range(n_frames):
        t = k / (n_frames - 1)
        th1 = lerp(theta1_0, theta1_1, t) if anim_theta1 else theta1_0
        ph1 = lerp(phi1_0, phi1_1, t) if anim_phi1 else phi1_0
        th2 = lerp(theta2_0, theta2_1, t) if anim_theta2 else theta2_0
        ph2 = lerp(phi2_0, phi2_1, t) if anim_phi2 else phi2_0
        u1 = float(u_from_theta(th1))
        u2 = float(u_from_theta(th2))
        ph1, ph2 = wrap_pi(ph1), wrap_pi(ph2)
        _check_u("u1", u1)
        _check_u("u2", u2)
        _check_u("u1+u2", u1 + u2)
        frames.append((u1, ph1, u2, ph2))
        trails["p1"].append(p_cylinder(u1, ph1))
        trails["p2"].append(p_cylinder(u2, ph2))
        trails["p_cyl"].append(p_cylinder(u1 + u2, wrap_pi(ph1 + ph2)))

    trails = {k: np.stack(v, axis=0) for k, v in trails.items()}
    return frames, trails


def launch_cylinder_group_demo():
    """Build the ipywidgets UI (static + animate) and show an initial figure."""
    busy = {"on": False}
    anim_cache = {"frames": None, "trails": None}

    view = W.ToggleButtons(
        options=[("static", "static"), ("animate", "animate")],
        value="static",
        description="view",
    )
    mode = W.ToggleButtons(
        options=[("expressions", "expr"), ("sliders", "slide")],
        value="expr",
        description="input",
    )

    # Static inputs (also supply fixed endpoints when a coord is not animated)
    u1_t = W.Text(value="0.65", description="u1", layout=W.Layout(width="210px"))
    u2_t = W.Text(value="-0.30", description="u2", layout=W.Layout(width="210px"))
    phi1_t = W.Text(
        value="-0.55/pi", description="phi1/pi", layout=W.Layout(width="210px")
    )
    phi2_t = W.Text(
        value="0.85/pi", description="phi2/pi", layout=W.Layout(width="210px")
    )
    u1_s = W.FloatSlider(
        value=0.65, min=-_U_MAX, max=_U_MAX, step=0.05,
        description="u1", continuous_update=False,
    )
    u2_s = W.FloatSlider(
        value=-0.30, min=-_U_MAX, max=_U_MAX, step=0.05,
        description="u2", continuous_update=False,
    )
    phi1_s = W.FloatSlider(
        value=-0.55, min=-np.pi, max=np.pi, step=0.05,
        description="phi1", continuous_update=False,
    )
    phi2_s = W.FloatSlider(
        value=0.85, min=-np.pi, max=np.pi, step=0.05,
        description="phi2", continuous_update=False,
    )

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
        value="<i>static: u and phi/π. animate: θ ranges (×π) and/or φ (×π); "
        "use Play / frame scrubber.</i>"
    )
    out = W.Output()

    # Animate controls — θ and φ as multiples of π
    anim_th1 = W.Checkbox(value=True, description="anim θ1")
    anim_ph1 = W.Checkbox(value=False, description="anim φ1")
    anim_th2 = W.Checkbox(value=False, description="anim θ2")
    anim_ph2 = W.Checkbox(value=True, description="anim φ2")
    th1_a = W.Text(value="1/3", description="θ1/π a", layout=W.Layout(width="180px"))
    th1_b = W.Text(value="2/3", description="θ1/π b", layout=W.Layout(width="180px"))
    ph1_a = W.Text(value="-0.55/pi", description="φ1/π a", layout=W.Layout(width="180px"))
    ph1_b = W.Text(value="0.4", description="φ1/π b", layout=W.Layout(width="180px"))
    th2_a = W.Text(value="0.6", description="θ2/π a", layout=W.Layout(width="180px"))
    th2_b = W.Text(value="0.6", description="θ2/π b", layout=W.Layout(width="180px"))
    ph2_a = W.Text(value="0.1", description="φ2/π a", layout=W.Layout(width="180px"))
    ph2_b = W.Text(value="0.7", description="φ2/π b", layout=W.Layout(width="180px"))
    n_frames = W.IntSlider(
        value=48, min=8, max=120, step=1,
        description="frames", continuous_update=False,
    )
    interval = W.IntSlider(
        value=80, min=20, max=400, step=10,
        description="ms/frame", continuous_update=False,
    )
    show_trail = W.Checkbox(value=True, description="show trails")
    build_btn = W.Button(description="Build frames", button_style="info")

    play = W.Play(
        value=0, min=0, max=47, step=1, interval=80,
        description="Play", disabled=True,
    )
    frame_s = W.IntSlider(
        value=0, min=0, max=47, step=1,
        description="frame", continuous_update=True, disabled=True,
        layout=W.Layout(width="420px"),
    )
    W.jslink((play, "value"), (frame_s, "value"))
    W.jslink((play, "interval"), (interval, "value"))

    expr_box = W.VBox([W.HBox([u1_t, phi1_t]), W.HBox([u2_t, phi2_t])])
    slide_box = W.VBox([W.HBox([u1_s, phi1_s]), W.HBox([u2_s, phi2_s])])
    slide_box.layout.display = "none"
    static_box = W.VBox([mode, expr_box, slide_box])

    anim_box = W.VBox([
        W.HTML(
            "<b>animate</b> — ranges are multiples of π "
            "(θ∈(0,1)×π on the open belt; φ free). "
            "Unchecked coords stay at endpoint <i>a</i>."
        ),
        W.HBox([anim_th1, th1_a, th1_b]),
        W.HBox([anim_ph1, ph1_a, ph1_b]),
        W.HBox([anim_th2, th2_a, th2_b]),
        W.HBox([anim_ph2, ph2_a, ph2_b]),
        W.HBox([n_frames, interval, show_trail]),
        W.HBox([build_btn, play, frame_s]),
    ])
    anim_box.layout.display = "none"

    cam_box = W.VBox([
        W.HTML("<b>camera</b>"),
        W.HBox([elev_s, azim_s]),
        W.HBox([follow, snap_btn]),
    ])

    def _sync_input_mode(_=None):
        if mode.value == "expr":
            expr_box.layout.display = None
            slide_box.layout.display = "none"
        else:
            expr_box.layout.display = "none"
            slide_box.layout.display = None

    def _sync_view(_=None):
        if view.value == "static":
            static_box.layout.display = None
            anim_box.layout.display = "none"
            plot_btn.layout.display = None
        else:
            static_box.layout.display = "none"
            anim_box.layout.display = None
            plot_btn.layout.display = "none"

    mode.observe(_sync_input_mode, names="value")

    def _read_static_points():
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
            a = (float(azim) + 180.0) % 360.0 - 180.0
            azim_s.value = a

    def _show_fig(fig, info, extra=""):
        display(fig)
        plt.close(fig)
        print(
            f"(u1,phi1)=({info['u1']:.3f},{info['phi1']:.3f}) "
            f"θ={info['theta1']:.3f} -> {np.round(info['p1'], 4)}"
        )
        print(
            f"(u2,phi2)=({info['u2']:.3f},{info['phi2']:.3f}) "
            f"θ={info['theta2']:.3f} -> {np.round(info['p2'], 4)}"
        )
        print(
            f"cyl sum=({info['u_sum']:.3f},{info['phi_sum']:.3f}) "
            f"θ={info['theta_sum']:.3f} -> {np.round(info['p_cyl'], 4)}"
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
            f"{extra}"
        )

    def _run_static(_=None):
        if busy["on"]:
            return
        busy["on"] = True
        try:
            with out:
                clear_output(wait=True)
                try:
                    u1, phi1, u2, phi2 = _read_static_points()
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
                    _show_fig(fig, info)
                except Exception as err:
                    status.value = (
                        f"<span style='color:#a00'><b>input error:</b> {err}</span>"
                    )
                    print("input error:", err)
        finally:
            busy["on"] = False

    def _read_anim_ranges():
        # θ texts are multiples of π; φ texts are multiples of π too.
        th1_0 = parse_expr(th1_a.value) * np.pi
        th1_1 = parse_expr(th1_b.value) * np.pi
        th2_0 = parse_expr(th2_a.value) * np.pi
        th2_1 = parse_expr(th2_b.value) * np.pi
        ph1_0 = parse_expr(ph1_a.value) * np.pi
        ph1_1 = parse_expr(ph1_b.value) * np.pi
        ph2_0 = parse_expr(ph2_a.value) * np.pi
        ph2_1 = parse_expr(ph2_b.value) * np.pi
        return (
            th1_0, th1_1, ph1_0, ph1_1,
            th2_0, th2_1, ph2_0, ph2_1,
        )

    def _build_frames(_=None):
        if busy["on"]:
            return
        busy["on"] = True
        try:
            try:
                (
                    th1_0, th1_1, ph1_0, ph1_1,
                    th2_0, th2_1, ph2_0, ph2_1,
                ) = _read_anim_ranges()
                frames, trails = build_animation_frames(
                    int(n_frames.value),
                    theta1_0=th1_0,
                    theta1_1=th1_1,
                    phi1_0=ph1_0,
                    phi1_1=ph1_1,
                    theta2_0=th2_0,
                    theta2_1=th2_1,
                    phi2_0=ph2_0,
                    phi2_1=ph2_1,
                    anim_theta1=bool(anim_th1.value),
                    anim_phi1=bool(anim_ph1.value),
                    anim_theta2=bool(anim_th2.value),
                    anim_phi2=bool(anim_ph2.value),
                )
                anim_cache["frames"] = frames
                anim_cache["trails"] = trails
                n = len(frames) - 1
                with play.hold_trait_notifications(), frame_s.hold_trait_notifications():
                    play.max = n
                    frame_s.max = n
                    play.min = 0
                    frame_s.min = 0
                    play.value = 0
                    frame_s.value = 0
                    play.disabled = False
                    frame_s.disabled = False
                    play.interval = int(interval.value)
                status.value = (
                    f"<b>built</b> {len(frames)} frames — press Play or scrub "
                    f"<code>frame</code>"
                )
                _show_frame(0)
            except Exception as err:
                anim_cache["frames"] = None
                anim_cache["trails"] = None
                play.disabled = True
                frame_s.disabled = True
                status.value = (
                    f"<span style='color:#a00'><b>build error:</b> {err}</span>"
                )
                print("build error:", err)
        finally:
            busy["on"] = False

    def _show_frame(k):
        frames = anim_cache["frames"]
        if not frames:
            return
        k = int(np.clip(k, 0, len(frames) - 1))
        u1, phi1, u2, phi2 = frames[k]
        trails = anim_cache["trails"] if show_trail.value else None
        if follow.value:
            p1 = p_cylinder(u1, phi1)
            p2 = p_cylinder(u2, phi2)
            p_cyl = p_cylinder(u1 + u2, wrap_pi(phi1 + phi2))
            _set_cam_quiet(elev_s.value, azim_facing([p1, p2, p_cyl]))
        with out:
            clear_output(wait=True)
            fig, info = draw_cylinder_sum(
                u1, phi1, u2, phi2,
                elev=elev_s.value,
                azim=azim_s.value,
                trails=trails,
                frame_label=rf"frame {k + 1}/{len(frames)}",
            )
            _show_fig(fig, info, extra=f"; frame {k + 1}/{len(frames)}")

    def _on_frame(change):
        if view.value != "animate":
            return
        if anim_cache["frames"] is None:
            return
        if busy["on"]:
            return
        busy["on"] = True
        try:
            _show_frame(change["new"])
        finally:
            busy["on"] = False

    def _snap_cam(_=None):
        try:
            if view.value == "animate" and anim_cache["frames"]:
                u1, phi1, u2, phi2 = anim_cache["frames"][int(frame_s.value)]
            else:
                u1, phi1, u2, phi2 = _read_static_points()
            p1 = p_cylinder(u1, wrap_pi(phi1))
            p2 = p_cylinder(u2, wrap_pi(phi2))
            p_cyl = p_cylinder(u1 + u2, wrap_pi(phi1 + phi2))
            _set_cam_quiet(elev_s.value, azim_facing([p1, p2, p_cyl]))
            if view.value == "animate":
                _show_frame(int(frame_s.value))
            else:
                _run_static()
        except Exception as err:
            status.value = f"<span style='color:#a00'><b>snap error:</b> {err}</span>"

    def _on_view_change(change):
        _sync_view()
        if change["new"] == "static":
            try:
                play.playing = False
            except Exception:
                pass
            _run_static()
        else:
            status.value = (
                "<i>Set θ/φ ranges (×π), tick what to animate, then "
                "<b>Build frames</b>. Scrub with Play / frame.</i>"
            )

    plot_btn.on_click(_run_static)
    snap_btn.on_click(_snap_cam)
    build_btn.on_click(_build_frames)
    frame_s.observe(_on_frame, names="value")
    view.observe(_on_view_change, names="value")

    for w in (u1_s, u2_s, phi1_s, phi2_s):
        w.observe(lambda ch: _run_static() if view.value == "static" else None, names="value")

    def _cam_redraw(_=None):
        if view.value == "static":
            _run_static()
        elif anim_cache["frames"] is not None:
            _show_frame(int(frame_s.value))

    for w in (elev_s, azim_s, follow, show_trail):
        w.observe(lambda ch: _cam_redraw(), names="value")

    ui = W.VBox([
        view,
        static_box,
        anim_box,
        cam_box,
        W.HBox([plot_btn, status]),
        out,
    ])
    display(ui)
    _sync_view()
    _snap_cam()
