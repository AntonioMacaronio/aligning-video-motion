"""The animation layered on the geometry sidecar figure (figures/arch_sidecar.svg): one update, played on a loop.

    from sidecar_animation import overlay; svg = svg.replace("</svg>", overlay() + "</svg>")

The layout constants are those of the thesis's figure-src/architecture-20261024/make_fig_sidecar.py, which draws the
static figure underneath; check_layout() compares them with that script's SVG. The animation uses SVG's own
animation elements (<animate>, <animateMotion>), so it plays inside an <img> without any script. Readers who ask for
reduced motion get make_post_figures.py's static copy (<name>_static.svg, a <picture> source on the page); the
overlay also hides itself under that media query when the SVG is opened on its own.

One loop, LOOP seconds, follows panel 2's timeline:
- 0-3 s: the rank's rollout runs (its box fills).
- 3-3.4 s: z_S and x_S go to the worker.
- The worker decodes and runs VGGT-Omega on the video while the motion goes through the CPF and R_cpf,cam, then
  scores L_geom, then backpropagates (dashed outlines) and sends L_geom, g_z, g_x back.
- 8.4 s: the rank re-runs step n from the inputs the rollout saved.
- 9.7 s: the optimizer step.
Meanwhile the rank trains on its four data microbatches. A playhead sweeps panel 2's time axis.
"""
import re

LOOP = 12.5
PALETTE = {"video": "#1d4ed8", "motion": "#047857", "joint": "#6d28d9", "neutral": "#475569", "grad": "#1e293b"}

# ---- make_fig_sidecar.py's layout ----------------------------------------------------------------------------------
W = 640
TOP, BOT = 32, 246
TX0, TX1 = 14, 216
WX0, WX1 = 290, 626
YR, YS, YP, BH = 98, 148, 216, 38
BX, BW = TX0 + 13, TX1 - TX0 - 26
ZW, MW, GAPT = 44, 40, 40
ZX = BX + (BW - ZW - GAPT - MW) / 2
MX = ZX + ZW + GAPT
FORK = WX0 + 12
YV, YM, CH = 110, 178, 32
B1, BWW = WX0 + 26, 90
B2 = B1 + BWW + 18
LX0, LW, LH = B2 + BWW + 16, 60, 44
LY = (YV + YM) / 2
LC = LX0 + LW / 2
XB = WX1 - 12
P2 = BOT + 18
RY, KY, LH2 = P2 + 54, P2 + 106, 30
X0 = 86
SEGMENTS = [("rollout", 146), *[(f"data {k}", 54) for k in range(1, 5)], ("re-run", 62), ("optimizer", None)]
X = {}
_x = X0
for _name, _w in SEGMENTS:
    _w = W - 14 - _x if _w is None else _w
    X[_name] = (_x, _x + _w)
    _x += _w + (8 if _name in ("rollout", "data 4", "re-run") else 4)
K0, K1 = X["rollout"][1] + 14, X["data 4"][0] + 30
TY = KY + LH2 / 2 + 14

# ---- the schedule (seconds) ---------------------------------------------------------------------------------------
LANE = {"rollout": (0.0, 3.0), "data 1": (3.3, 4.5), "data 2": (4.5, 5.7), "data 3": (5.7, 6.9), "data 4": (6.9, 8.1),
        "re-run": (8.4, 9.4), "optimizer": (9.7, 10.9)}


def x_to_t(x):
    """Panel 2's time axis: each lane box spans its LANE window, the gaps between boxes the time between windows."""
    marks = [(X0, 0.0)]
    for name, (t0, t1) in LANE.items():
        marks += [(X[name][0], t0), (X[name][1], t1)]
    for (xa, ta), (xb, tb) in zip(marks, marks[1:]):
        if xa <= x <= xb:
            return ta + (tb - ta) * (x - xa) / (xb - xa) if xb > xa else ta
    raise ValueError(x)


T_SEND, T_WORK0, T_WORK1 = 3.0, x_to_t(K0), x_to_t(K1)     # request leaves; worker busy from K0 to K1 on the axis
T_BACK = T_WORK1 - 1.5                                      # backward pass starts
T_ARRIVE = LANE["re-run"][0]


# ---- SVG helpers --------------------------------------------------------------------------------------------------
def _frames(points):
    """[(t, value)] -> SMIL values / keyTimes over one loop; clamps to [0, LOOP] and adds the endpoints."""
    pts = sorted((min(max(t, 0.0), LOOP), v) for t, v in points)
    if pts[0][0] > 0:
        pts.insert(0, (0.0, pts[0][1]))
    if pts[-1][0] < LOOP:
        pts.append((LOOP, pts[-1][1]))
    values = ";".join(str(v) for _, v in pts)
    times = ";".join(f"{t / LOOP:.4f}" for t, _ in pts)
    return values, times


def _animate(attr, points):
    values, times = _frames(points)
    return (f'<animate attributeName="{attr}" values="{values}" keyTimes="{times}" dur="{LOOP}s" '
            f'repeatCount="indefinite"/>')


def _visible(t0, t1, fade=0.15):
    """Opacity 1 over [t0, t1], fading in and out."""
    if t0 <= 0:
        return _animate("opacity", [(0, 1), (t1, 1), (t1 + fade, 0)])
    return _animate("opacity", [(0, 0), (t0 - fade, 0), (t0, 1), (t1, 1), (t1 + fade, 0)])


def highlight(x, y, w, h, color, t0, t1, dashed=False, rx=6):
    """A glow around a box while it works; dashed for the backward pass."""
    dash = ' stroke-dasharray="3 2.5"' if dashed else ""
    return (f'<g opacity="0">{_visible(t0, t1)}'
            f'<rect x="{x - 3:.2f}" y="{y - 3:.2f}" width="{w + 6:.2f}" height="{h + 6:.2f}" rx="{rx + 2}" fill="none" '
            f'stroke="{color}" stroke-opacity="0.18" stroke-width="5"/>'
            f'<rect x="{x - 1:.2f}" y="{y - 1:.2f}" width="{w + 2:.2f}" height="{h + 2:.2f}" rx="{rx + 1}" fill="{color}" '
            f'fill-opacity="0.08" stroke="{color}" stroke-width="2"{dash}/></g>')


def progress(x, y, w, color, t0, t1):
    """A bar along the bottom of a box that fills from t0 to t1."""
    return (f'<g opacity="0">{_visible(t0, t1)}<rect x="{x:.2f}" y="{y:.2f}" width="0" height="2.6" rx="1.3" '
            f'fill="{color}">{_animate("width", [(0, 0), (t0, 0), (t1, round(w, 2))])}</rect></g>')


def packet(path, color, t0, t1, ring=False, r=3.4):
    """A dot that travels the path from t0 to t1: filled for data, a ring for gradients."""
    fill, stroke = ("#ffffff", color) if ring else (color, "#ffffff")
    motion = (f'<animateMotion path="{path}" keyPoints="0;0;1;1" keyTimes="0;{t0 / LOOP:.4f};{t1 / LOOP:.4f};1" '
              f'calcMode="linear" dur="{LOOP}s" repeatCount="indefinite"/>')
    return (f'<circle r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{1.8 if ring else 1.2}" opacity="0">'
            f'{_visible(t0, t1, fade=0.08)}{motion}</circle>')


def poly(*points):
    return "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in points)


# ---- the animation ------------------------------------------------------------------------------------------------
def overlay():
    V, M, J, N, G = (PALETTE[k] for k in ("video", "motion", "joint", "neutral", "grad"))
    a = []
    # panel 1, the rank: the rollout runs and leaves z_S and x_S
    a.append(highlight(BX, YR - BH / 2, BW, BH, J, 0.0, T_SEND))
    a.append(progress(BX + 8, YR + BH / 2 - 5.5, BW - 16, J, 0.0, T_SEND))
    a.append(highlight(ZX, YS - 8, ZW, 16, V, T_SEND - 0.3, T_SEND + 0.4, rx=2))
    a.append(highlight(MX, YS - 4.5, MW, 9, M, T_SEND - 0.3, T_SEND + 0.4, rx=2))
    # the request crosses to the worker and forks onto its two rails
    a.append(packet(poly((MX + MW, YS), (FORK, YS), (FORK, YV), (B1, YV)), V, T_SEND, T_WORK0 + 0.2))
    a.append(packet(poly((MX + MW, YS), (FORK, YS), (FORK, YM), (B1, YM)), M, T_SEND, T_WORK0 + 0.2))
    # the worker's forward pass: decoder -> VGGT-Omega on the video rail, CPF -> R_cpf,cam on the motion rail
    f0 = T_WORK0 + 0.2
    t_dec, t_vggt, t_loss = f0 + 0.7, f0 + 1.6, f0 + 2.0
    a.append(highlight(B1, YV - CH / 2, BWW, CH, V, f0, t_dec))
    a.append(highlight(B2, YV - CH / 2, BWW, CH, V, t_dec, t_vggt))
    a.append(highlight(B1, YM - CH / 2, BWW, CH, M, f0, f0 + 0.5))
    a.append(highlight(B2, YM - CH / 2, BWW, CH, M, f0 + 0.5, f0 + 1.0))
    a.append(packet(poly((B2 + BWW, YV), (LC, YV), (LC, LY - LH / 2)), V, t_vggt, t_loss))
    a.append(packet(poly((B2 + BWW, YM), (LC, YM), (LC, LY + LH / 2)), M, t_vggt, t_loss))
    a.append(highlight(LX0, LY - LH / 2, LW, LH, N, t_loss, T_BACK + 0.2))
    # backward through VGGT-Omega, the decoder and the CPF, then the gradients go back to the rank
    b1 = T_BACK + 0.55
    a.append(highlight(B2, YV - CH / 2, BWW, CH, G, T_BACK, b1, dashed=True))
    a.append(highlight(B1, YV - CH / 2, BWW, CH, G, b1, T_WORK1, dashed=True))
    a.append(highlight(B2, YM - CH / 2, BWW, CH, G, T_BACK, b1, dashed=True))
    a.append(highlight(B1, YM - CH / 2, BWW, CH, G, b1, T_WORK1, dashed=True))
    a.append(packet(poly((LX0 + LW, LY), (XB, LY), (XB, YP), (WX0 + 1, YP), (BX + BW, YP)), G, T_WORK1, T_ARRIVE,
                    ring=True))
    # the rank re-runs step n from the inputs the rollout saved
    rerun0, rerun1 = LANE["re-run"]
    a.append(highlight(BX, YP - BH / 2, BW, BH, J, rerun0, rerun1))
    a.append(f'<g opacity="0">{_visible(rerun0 - 0.1, rerun1)}<path d="{poly((BX, YR + 8), (BX - 6, YR + 8), (BX - 6, YP), (BX, YP))}" '
             f'fill="none" stroke="{J}" stroke-width="2" stroke-opacity="0.7"/></g>')
    # panel 2: the same update in time
    for name, (t0, t1) in LANE.items():
        kind = J if name in ("rollout", "re-run") else N
        x0, x1 = X[name]
        a.append(highlight(x0, RY - LH2 / 2, x1 - x0, LH2, kind, t0, t1))
    a.append(progress(X["rollout"][0] + 8, RY + LH2 / 2 - 5, X["rollout"][1] - X["rollout"][0] - 16, J, 0.0, T_SEND))
    a.append(highlight(K0, KY - LH2 / 2, K1 - K0, LH2, N, T_WORK0, T_WORK1))
    a.append(progress(K0 + 8, KY + LH2 / 2 - 5, K1 - K0 - 16, N, T_WORK0, T_WORK1))
    a.append(packet(poly((X["rollout"][1] - 6, RY + LH2 / 2), (K0 + 6, KY - LH2 / 2)), N, T_SEND, T_WORK0))
    a.append(packet(poly((K1 - 6, KY - LH2 / 2), (X["re-run"][0] + 10, RY + LH2 / 2)), G, T_WORK1, T_ARRIVE, ring=True))
    # the playhead on the time axis
    t_end = LANE["optimizer"][1]
    xs = [X0] + [v for name in LANE for v in X[name]]
    head = [(x_to_t(x), round(x, 2)) for x in xs]
    a.append(f'<g opacity="0">{_visible(0.0, t_end, fade=0.3)}<line x1="0" y1="{RY - LH2 / 2 - 6}" x2="0" '
             f'y2="{TY + 4}" stroke="{J}" stroke-width="1.4" stroke-opacity="0.75">'
             f'{_animate("x1", head)}{_animate("x2", head)}</line></g>')
    return ('<g class="sidecar-anim"><style>@media (prefers-reduced-motion: reduce) { .sidecar-anim { display: none; } }'
            '</style>' + "".join(a) + "</g>")


def check_layout(source_svg):
    """The constants above against make_fig_sidecar.py's SVG: the boxes this animation outlines must be there."""
    boxes = {(round(float(x), 1), round(float(y), 1), round(float(w), 1), round(float(h), 1))
             for x, y, w, h in re.findall(r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"', source_svg)}
    expect = [(BX, YR - BH / 2, BW, BH), (BX, YP - BH / 2, BW, BH), (B1, YV - CH / 2, BWW, CH), (B2, YV - CH / 2, BWW, CH),
              (B1, YM - CH / 2, BWW, CH), (B2, YM - CH / 2, BWW, CH), (LX0, LY - LH / 2, LW, LH),
              (K0, KY - LH2 / 2, K1 - K0, LH2), *[(x0, RY - LH2 / 2, x1 - x0, LH2) for x0, x1 in X.values()]]
    missing = [b for b in expect if tuple(round(v, 1) for v in b) not in boxes]
    assert not missing, f"make_fig_sidecar.py's layout changed: {missing}"
