"""The animation layered on the three-tasks figure (figures/arch_tasks.jpg): each task in turn, played on a loop.

    from tasks_animation import overlay; svg = overlay(source_svg, view_box)

source_svg is the thesis's figure-src/architecture-20261024/make_fig_tasks.py output, the drawing the JPEG is rendered
from. The clip's frames (the video-conditioned row's given video) and its bodies (the motion-conditioned row's given
motion) are taken from it. The overlay is a transparent SVG in the drawing's coordinates, cropped to the JPEG's view
box plus PAD. The page lays it over the JPEG in an <object> (whose clock a script can read: the earlier three-column
viewer lit up the matching task with it); the animation itself is SVG's (<animate>), no script. Readers who ask for reduced
motion don't get the overlay (the page's CSS hides it). Layout constants are make_fig_tasks.py's, and check_layout()
compares them with the source.

Each task gets PHASE / SPEED seconds:
- the other two rows dim;
- the given tokens glow, and a label lists what is given, the text prompt included;
- the generated tokens denoise: the clean frame or body fades in over the static (sigma 1 -> 0, the bar shrinking),
  all at once, as in the sampler;
- a short hold, then the row resets.
After the third task the plain figure shows for PAUSE seconds before the loop starts again.
The clean content of a generated token is the clip's own frame or body, which is what the model is trained to produce.
"""
import re

PHASE = 5.5                  # one task, in the schedule's units
SPEED = 1.33                 # played this much faster: 4.1 s per task
PAUSE = 2.0                  # seconds of the plain figure at the end of each loop
LOOP = 3 * PHASE + PAUSE * SPEED   # in the schedule's units: a 14.4 s loop
INK, MUTED = "#1e293b", "#64748b"
VIDEO, MOTION = "#1d4ed8", "#047857"
SANS = "'Helvetica Neue', Helvetica, Arial, sans-serif"
PAD = 4                      # drawing units the overlay reaches past the figure's crop

# ---- make_fig_tasks.py's layout ------------------------------------------------------------------------------------
W, LX, CELL, PITCH, ROW, MH, LAT_T = 640, 206, 30, 32.4, 110, 46, 13
TASKS = [("the first frame, the first pose and the text prompt", False, False),
         ("the whole video, the first pose and the text prompt", True, False),
         ("the whole motion, the first frame and the text prompt", False, True)]


def y_row(t):
    return 14 + ROW * t


def _images(source_svg):
    found = re.findall(r'<image x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"[^>]*'
                       r'href="(data:image/\w+;base64,[^"]+)"', source_svg)
    return {(round(float(x), 2), round(float(y), 2), round(float(w), 2), round(float(h), 2)): href
            for x, y, w, h, href in found}


def check_layout(source_svg):
    """Every frame and body this animation reuses must sit where make_fig_tasks.py's constants put it."""
    images = _images(source_svg)
    for k in range(LAT_T):
        x = round(LX + PITCH * k, 2)
        assert (x, y_row(1) + 8, CELL, CELL) in images, f"frame {k} moved"
        assert (round(x + 1, 2), y_row(2) + 45, CELL - 2, MH - 2) in images, f"body {k} moved"


# ---- SMIL helpers (as in sidecar_animation.py) --------------------------------------------------------------------
def _frames(points):
    pts = sorted((min(max(t, 0.0), LOOP), v) for t, v in points)
    if pts[0][0] > 0:
        pts.insert(0, (0.0, pts[0][1]))
    if pts[-1][0] < LOOP:
        pts.append((LOOP, pts[-1][1]))
    return ";".join(str(v) for _, v in pts), ";".join(f"{t / LOOP:.4f}" for t, _ in pts)


def _animate(attr, points):
    values, times = _frames(points)
    return (f'<animate attributeName="{attr}" values="{values}" keyTimes="{times}" dur="{LOOP / SPEED:.2f}s" '
            f'repeatCount="indefinite"/>')


def _window(t0, t1, fade=0.3, peak=1):
    """Opacity peak over [t0, t1], fading in and out."""
    return _animate("opacity", [(0, 0), (t0 - fade, 0), (t0, peak), (t1, peak), (t1 + fade, 0)])


def overlay(source_svg, view_box):
    check_layout(source_svg)
    images = _images(source_svg)
    frame = [images[(round(LX + PITCH * k, 2), y_row(1) + 8, CELL, CELL)] for k in range(LAT_T)]
    body = [images[(round(LX + PITCH * k + 1, 2), y_row(2) + 45, CELL - 2, MH - 2)] for k in range(LAT_T)]
    defs = "".join(f'<image id="f{k}" width="{CELL}" height="{CELL}" preserveAspectRatio="xMidYMid slice" href="{h}"/>'
                   for k, h in enumerate(frame))
    defs += "".join(f'<image id="b{k}" width="{CELL - 2}" height="{MH - 2}" href="{h}"/>' for k, h in enumerate(body))
    a = []
    for t, (given_text, video_given, motion_given) in enumerate(TASKS):
        p0 = PHASE * t
        y0 = y_row(t)
        g0, d0, d1, end = p0 + 0.2, p0 + 1.6, p0 + 4.2, p0 + PHASE - 0.4
        # the other rows dim while this one plays
        for other in range(3):
            if other != t:
                a.append(f'<rect x="8" y="{y_row(other) + 2}" width="{W - 16}" height="{ROW - 8}" fill="#ffffff" '
                         f'opacity="0">{_window(p0 + 0.1, end, peak=0.72)}</rect>')
        # the given tokens glow: a thicker border just outside each one (PAD keeps it whole at the figure's edges);
        # the label lists them
        glow = []
        for k in range(LAT_T):
            x = LX + PITCH * k
            if video_given or k == 0:
                glow.append(f'<rect x="{x - 1.5:.2f}" y="{y0 + 6.5}" width="{CELL + 3}" height="{CELL + 3}" rx="2" fill="none" '
                            f'stroke="{VIDEO}" stroke-width="2.4"/>')
            if motion_given or k == 0:
                glow.append(f'<rect x="{x - 1.5:.2f}" y="{y0 + 42.5}" width="{CELL + 3}" height="{MH + 3}" rx="2" fill="none" '
                            f'stroke="{MOTION}" stroke-width="2.4"/>')
        a.append(f'<g opacity="0">{_window(g0, d0 + 0.2)}{"".join(glow)}</g>')
        a.append(f'<g opacity="0">{_window(g0, end)}'
                 f'<text x="14" y="{y0 + 55}" font-family="{SANS}" font-size="8.2" fill="{INK}" font-weight="600">given:'
                 f'<tspan font-weight="400" fill="{MUTED}"> {given_text.replace(" and the text prompt", ",")}</tspan></text>'
                 f'<text x="14" y="{y0 + 66}" font-family="{SANS}" font-size="8.2" fill="{MUTED}">and the text prompt '
                 f'<tspan font-style="italic" font-family="\'STIX Two Text\', \'Times New Roman\', serif">c</tspan></text></g>')
        # the denoising: sigma's bar empties while the clean tokens fade in over the static
        a.append(f'<g opacity="0">{_window(d0 - 0.1, end)}'
                 f'<text x="14" y="{y0 + 81}" font-family="\'STIX Two Text\', \'Times New Roman\', serif" font-style="italic" '
                 f'font-size="9.5" fill="{MUTED}">σ</text>'
                 f'<rect x="24" y="{y0 + 75.5}" width="90" height="5" rx="2.5" fill="#e2e8f0"/>'
                 f'<rect x="24" y="{y0 + 75.5}" width="90" height="5" rx="2.5" fill="{MUTED}">'
                 f'{_animate("width", [(0, 90), (d0, 90), (d1, 0), (end + 0.3, 0), (end + 0.31, 90)])}</rect></g>')
        clean = []
        for k in range(LAT_T):
            x = LX + PITCH * k
            if not (video_given or k == 0):
                clean.append(f'<use href="#f{k}" x="{x:.2f}" y="{y0 + 8}"/>')
            if not (motion_given or k == 0):
                clean.append(f'<rect x="{x + 0.4:.2f}" y="{y0 + 44.4}" width="{CELL - 0.8}" height="{MH - 0.8}" fill="#ffffff"/>'
                             f'<use href="#b{k}" x="{x + 1:.2f}" y="{y0 + 45}"/>')
        ramp = _animate("opacity", [(0, 0), (d0, 0), (d1, 1), (end, 1), (end + 0.3, 0)])
        a.append(f'<g opacity="0">{ramp}{"".join(clean)}</g>')
    # The overlay reaches PAD units past the figure on every side, so the glows on the first row and the last column are
    # not clipped; data-pad tells build_site.py how far to extend the overlay past the figure's box, and data-width and
    # data-height its size in drawing units (it fills its box: width and height 100%).
    x0, y0, w, h = view_box
    x0, y0, w, h = round(x0 - PAD, 2), round(y0 - PAD, 2), round(w + 2 * PAD, 2), round(h + 2 * PAD, 2)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="100%" height="100%" '
            f'data-width="{w}" data-height="{h}" viewBox="{x0} {y0} {w} {h}" data-pad="{PAD}"><defs>{defs}</defs>'
            f'{"".join(a)}</svg>\n')
