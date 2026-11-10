"""The teaser video at the top of the main page: the three tasks, one thesis figure each, as inputs -> MM-DiT -> outputs.

    PYTHONPATH=<LVP tree with datasets/head_correction.py> python make_teaser.py \
        <gallery dir> <figure-src gallery dir> <site dir> [--chrome-flag ...]

The arguments are make_thesis_scenes.py's, without --tasks: each figure shows its own task. One segment per figure, on
a white WIDTH x HEIGHT frame under its task's name:
1. its inputs appear on the left, one above the other: the text prompt, then the first frame or the recorded video, then
   the first pose or the recorded motion;
2. their tokens (gray for text, blue for video, green for motion) flow into the MM-DiT box, which lights up;
3. tokens flow out to the outputs on the right, which start as noise and are denoised to their first frame;
4. the outputs play: the figure's 145 motion frames at 30 fps (video frame t // 3, as on the page), a given video or
   motion playing along with them.
A video is its 49 frames (scenes/<name>/frames), framed in the time ramp's color like the page's videos. A 3D scene is
the page's scene for the figure (make_thesis_scenes.py's functions: the figure's cut point cloud, the floor and the body
colored along the ramp), rendered by viser from a square version of the page's opening camera that swings ORBIT degrees
around it while it plays. A pose or a given motion is the body alone on the floor grid, seen by make_task_inputs.py's
follow camera. Writes <site>/media/teaser.mp4 (H.264, frames piped to ffmpeg) and its poster teaser.jpg.
"""
import json
import subprocess
import sys
import tempfile
import time

import numpy as np
import viser
import viser.transforms as vtf
from PIL import Image, ImageDraw, ImageFont

CHROME_FLAGS = [a for a in sys.argv[4:] if a.startswith("--") and a not in ("--videos-only", "--cameras-only")]
sys.argv = sys.argv[:4] + [a for a in sys.argv[4:] if a not in CHROME_FLAGS]
import make_task_inputs as inputs  # noqa: E402  the follow camera for a pose or a given motion
import make_thesis_scenes as m  # noqa: E402  parses the same command line

# The layout is in units of a 1920 x 960 frame; everything is drawn SCALE times larger, so the video stays sharp where it
# fills a laptop's Retina screen (about 1400 px wide on the page, so 2800 device pixels).
SCALE = 1.5
LAYOUT_W, LAYOUT_H, FPS = 1920, 960, 30
WIDTH, HEIGHT = round(LAYOUT_W * SCALE), round(LAYOUT_H * SCALE)


def px(v):
    return round(v * SCALE)
INK, MUTED, RULE, PANEL = (31, 35, 40), (100, 116, 139), (220, 223, 228), (245, 246, 248)
TEXT_C, VIDEO_C, MOTION_C = (148, 163, 184), (29, 78, 216), (4, 120, 87)
MODEL_FILL, MODEL_EDGE, MODEL_LIT, MODEL_INK = (247, 245, 255), (196, 181, 253), (124, 58, 237), (76, 29, 149)
FONTS = "/System/Library/Fonts/HelveticaNeue.ttc"   # faces: 0 regular, 1 bold, 2 italic
HEADINGS = {"joint": "Generates video and motion", "video_cond": "Given video, generates motion",
            "motion_cond": "Given motion, generates video"}
SEGMENTS = ["joint_1", "video_cond_1", "motion_cond_3"]   # one per task: the staircase, the hallway, the laptop
ORBIT = 16.0                               # degrees the scene camera swings while the outputs play

# ---- layout (px) ---------------------------------------------------------------------------------------------------
IN_X, IN_R = 80, 560                       # the inputs' column
PROMPT = (IN_X, 170, IN_R, 345)            # the prompt's box
TILE = 190                                 # an input tile; two of them below the prompt, each with its label beside it
TILE_Y = (375, 595)
BOX = (700, 170, 920, 785)                 # the MM-DiT
OUT_X, OUT_R, OUT_SIDE = 1000, 1860, 420   # the outputs' area and each output's square
OUT_Y = (BOX[1] + BOX[3]) // 2 - OUT_SIDE // 2
BORDER, RADIUS = 8, 12
# ---- timeline (frames) ---------------------------------------------------------------------------------------------
APPEAR, FLOW, REVEAL, OUTRO = 15, 30, 24, 12   # before the 145 frames of playback, and after


def font(size, face=0):
    """A face at `size` layout units."""
    return ImageFont.truetype(FONTS, px(size), index=face)


class Pen:
    """ImageDraw in layout units: coordinates, widths and radii are scaled on the way in, text lengths on the way out."""

    def __init__(self, image):
        self.d = ImageDraw.Draw(image)

    def text(self, xy, text, **kw):
        self.d.text((xy[0] * SCALE, xy[1] * SCALE), text, **kw)

    def textlength(self, text, font):
        return self.d.textlength(text, font=font) / SCALE

    def line(self, points, width=1, **kw):
        self.d.line([v * SCALE for v in points], width=max(1, px(width)), **kw)

    def rounded_rectangle(self, box, radius, width=1, **kw):
        self.d.rounded_rectangle([v * SCALE for v in box], px(radius), width=max(1, px(width)), **kw)


def rounded(size, radius):
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius, fill=255)
    return mask


def framed_video(frame, side, color, border=BORDER):
    """A video frame `side` layout units square, in a `border`-wide frame of the ramp's color (drawn at SCALE)."""
    side, border = px(side), px(border)
    out = Image.new("RGB", (side, side), color)
    inner = frame.convert("RGB").resize((side - 2 * border, side - 2 * border), Image.LANCZOS)
    out.paste(inner, (border, border), rounded(inner.size, max(px(RADIUS) - border // 2, 2)))
    return out


def outlined(image, side):
    """An image `side` layout units square, rounded, with a light outline (drawn at SCALE)."""
    side = px(side)
    out = Image.new("RGB", (side, side), "white")
    image = image.resize((side, side), Image.LANCZOS) if image.size != (side, side) else image
    out.paste(image, (0, 0), rounded(image.size, px(RADIUS)))
    ImageDraw.Draw(out).rounded_rectangle((0, 0, side - 1, side - 1), px(RADIUS), outline=RULE, width=px(2))
    return out


def noisy(image, sigma, noise):
    """The image under static, sigma 1 (all static) to 0 (clean), as the sampler denoises."""
    if sigma <= 0:
        return image
    return Image.fromarray(np.uint8(np.clip((1 - sigma) * np.asarray(image, np.float32) + sigma * noise, 0, 255)))


def wrap(text, face, width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and face.getlength(trial) / SCALE > width:   # width in layout units
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + [line]


def render(client, side, position, look_at, fov):
    wxyz = vtf.SO3.from_matrix(m.look_at_rotation(np.asarray(position), np.asarray(look_at))).wxyz
    rgba = Image.fromarray(client.get_render(side, side, wxyz=wxyz, position=np.asarray(position), fov=fov,
                                             transport_format="png")).convert("RGBA")
    return Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")


def orbit(position, look_at, degrees):
    a, d = np.radians(degrees), np.asarray(position) - np.asarray(look_at)
    return np.asarray(look_at) + np.array([d[0] * np.cos(a) - d[1] * np.sin(a), d[0] * np.sin(a) + d[1] * np.cos(a), d[2]])


def arrow(draw, x0, x1, y, color=MUTED, width=4):
    draw.line((x0, y, x1 - 3, y), fill=color, width=width)
    draw.line((x1 - 18, y - 13, x1, y, x1 - 18, y + 13), fill=color, width=width, joint="curve")


def token(draw, x, y, color):
    draw.rounded_rectangle((x - 7, y - 7, x + 7, y + 7), 3, fill=color)


def flow(draw, frame, sources, x0, x1, start, count=5, every=4, travel=16):
    """Tokens from x0 to x1 along each source's row (y, color), `count` per source, one every `every` frames."""
    for y, color in sources:
        for k in range(count):
            f = (frame - start - k * every) / travel
            if 0 <= f <= 1:
                token(draw, x0 + (x1 - x0) * f, y, color)


class Segment:
    """One figure's frames: its inputs, outputs and the given clips, already rendered, and how to draw each step."""

    def __init__(self, e, frames, first_pose, motion, scene, caption):
        self.e, self.mode, self.frames, self.first_pose = e, e["mode"], frames, first_pose
        self.motion, self.scene, self.caption = motion, scene, caption
        rng = np.random.default_rng(0)
        self.noise = rng.normal(150, 55, (px(OUT_SIDE), px(OUT_SIDE), 3)).clip(0, 255).astype(np.float32)
        self.last = len(scene or motion) - 1

    def video(self, t):
        return self.frames[min(t // 3, len(self.frames) - 1)]

    def inputs(self, t):
        """The two input tiles at playback frame t (a given clip plays; a first frame or pose holds)."""
        color = m.ramp(t / self.last)
        if self.mode == "joint":
            return [("first frame", framed_video(self.frames[0], TILE, m.ramp(0.0), 6), VIDEO_C),
                    ("first pose", outlined(self.first_pose, TILE), MOTION_C)]
        if self.mode == "video_cond":
            return [("recorded video", framed_video(self.video(t), TILE, color, 6), VIDEO_C),
                    ("first pose", outlined(self.first_pose, TILE), MOTION_C)]
        return [("first frame", framed_video(self.frames[0], TILE, m.ramp(0.0), 6), VIDEO_C),
                ("recorded motion", outlined(self.motion[t], TILE), MOTION_C)]

    def outputs(self, t):
        """(label, image, token color) per output at playback frame t."""
        video = ("generated video", framed_video(self.video(t), OUT_SIDE, m.ramp(t / self.last)), VIDEO_C)
        scene = ("generated motion", outlined(self.scene[t], OUT_SIDE), MOTION_C) if self.scene else None
        return {"joint": [video, scene], "video_cond": [scene], "motion_cond": [video]}[self.mode]

    def draw(self, i):
        """Frame i of the segment."""
        play0 = APPEAR + FLOW + REVEAL
        t = min(max(i - play0, 0), self.last)
        out = Image.new("RGB", (WIDTH, HEIGHT), "white")
        d = Pen(out)
        heading = HEADINGS[self.mode]
        d.text(((LAYOUT_W - d.textlength(heading, font=font(48, 1))) / 2, 46), heading, font=font(48, 1), fill=INK)
        d.text((IN_X, 132), "INPUTS", font=font(20, 1), fill=MUTED)
        # the inputs
        d.rounded_rectangle(PROMPT, 10, fill=PANEL)
        lines = wrap(f"“{self.caption}”", font(23, 2), PROMPT[2] - PROMPT[0] - 36)
        top = (PROMPT[1] + PROMPT[3]) / 2 - len(lines) * 29 / 2
        for k, line in enumerate(lines):
            d.text((PROMPT[0] + 18, top + 29 * k), line, font=font(23, 2), fill=INK)
        rows = [((PROMPT[1] + PROMPT[3]) // 2, TEXT_C)]
        for (label, image, color), y in zip(self.inputs(t), TILE_Y):
            out.paste(image, (px(IN_X), px(y)), rounded(image.size, px(RADIUS)))
            d.text((IN_X + TILE + 22, y + TILE / 2 - 14), label, font=font(25), fill=INK)
            rows.append((y + TILE // 2, color))
        for y, _ in rows:
            arrow(d, IN_R + 12, BOX[0] - 10, y, RULE, 3)
        # the MM-DiT, lit while the tokens pass
        lit = APPEAR <= i < play0
        if lit:
            d.rounded_rectangle((BOX[0] - 7, BOX[1] - 7, BOX[2] + 7, BOX[3] + 7), 20, fill=(237, 233, 254))
        d.rounded_rectangle(BOX, 16, fill=MODEL_FILL, outline=MODEL_LIT if lit else MODEL_EDGE, width=5 if lit else 3)
        label = "MM-DiT"
        cx, cy = (BOX[0] + BOX[2]) / 2, (BOX[1] + BOX[3]) / 2
        d.text((cx - d.textlength(label, font=font(38, 1)) / 2, cy - 24), label, font=font(38, 1), fill=MODEL_INK)
        flow(d, i, rows, IN_R + 14, BOX[0] - 4, APPEAR)
        # the outputs: static during the flow, denoised as the tokens come out, then playing
        outs = self.outputs(t)
        xs = [OUT_X, OUT_X + OUT_SIDE + 20] if len(outs) == 2 else [(OUT_X + OUT_R - OUT_SIDE) // 2]
        d.text((xs[0], OUT_Y - 38), "OUTPUTS", font=font(20, 1), fill=MUTED)   # above the first output, left-aligned
        for (label, image, color), x in zip(outs, xs):
            if i < APPEAR:
                continue
            sigma = 1.0 if i < APPEAR + FLOW else max(0.0, 1 - (i - APPEAR - FLOW) / REVEAL)
            shown = noisy(image, sigma, self.noise)
            alpha = min(1.0, (i - APPEAR) / 10)
            if alpha < 1:
                shown = Image.blend(Image.new("RGB", shown.size, "white"), shown, alpha)
            out.paste(shown, (px(x), px(OUT_Y)), rounded(shown.size, px(RADIUS)))
            w = d.textlength(label, font=font(25))
            d.text((x + (OUT_SIDE - w) / 2, OUT_Y + OUT_SIDE + 14), label, font=font(25), fill=INK)
        yo = OUT_Y + OUT_SIDE // 2   # the tokens come out along one row to the first output, in the outputs' colors
        arrow(d, BOX[2] + 12, xs[0] - 12, yo, RULE, 3)
        for k, (_, _, color) in enumerate(outs):
            flow(d, i, [(yo, color)], BOX[2] + 14, xs[0] - 6, APPEAR + FLOW + 2 * k, count=4, every=4 * len(outs),
                 travel=14)
        # in and out
        n = self.length()
        fade = min(1.0, i / APPEAR, (n - 1 - i) / OUTRO if i >= n - OUTRO else 1.0)
        return out if fade >= 1 else Image.blend(Image.new("RGB", out.size, "white"), out, max(fade, 0.0))

    def length(self):
        return APPEAR + FLOW + REVEAL + self.last + 1 + OUTRO


def main():
    examples = {e["name"]: e for e in json.loads((m.FIGSRC / "examples.json").read_text())}
    captions = {e["name"]: e["caption"] for e in json.loads((m.SITE / "thesis_examples.json").read_text())}
    skinner, identities = m.MhrMeshSkinner(), m.SubjectIdentities.load(m.SUBJECT_IDENTITY_PATH)
    server = viser.ViserServer(host="127.0.0.1", port=9492, verbose=False)
    server.scene.set_up_direction("+z")
    chrome = subprocess.Popen(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--headless=new",
                               *(CHROME_FLAGS or ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]),
                               f"--window-size={px(2 * TILE)},{px(2 * TILE)}", f"--user-data-dir={tempfile.mkdtemp()}",
                               f"http://127.0.0.1:{server.get_port()}"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out = m.SITE / "media" / "teaser.mp4"
    encoder = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                                f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "24",
                                "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)],
                               stdin=subprocess.PIPE)
    try:
        deadline = time.time() + 60
        while not server.get_clients() and time.time() < deadline:
            time.sleep(0.5)
        client = next(iter(server.get_clients().values()))
        total, poster = 0, None
        for name in [n for n in SEGMENTS if not m.NAMES or n in m.NAMES]:   # names on the command line: a preview
            started = time.time()
            e = examples[name]
            scene, points, colors, middle, args, view, _ = m.figure_scene(name, name, m.GALLERY)
            position, look_at, fov = m.site_camera(points, scene["vertices"], dict(view, aspect=1.0))
            points, colors = m.thin(points, colors)
            vertices, error = m.posed_body(m.GALLERY, e, scene, skinner, identities)
            assert error < 1e-3, f"{name}: posed body differs from the figure's snapshots by {error:.4f} m"
            frames = [Image.open(p).convert("RGB") for p in sorted((m.GALLERY / "scenes" / name / "frames").glob("*.jpg"))]
            last = len(vertices) - 1
            # the body alone on the grid, followed from the figure camera's side: the first pose, or the given motion
            path = inputs.follow_path(vertices)
            heading = np.array(view["look_at"][:2]) - np.array(view["position"][:2])
            heading /= np.linalg.norm(heading)
            server.scene.reset()
            server.scene.add_grid("/floor", width=24.0, height=24.0, plane="xy", cell_size=0.5, section_size=1.0,
                                  cell_color=(215, 215, 215), section_color=(185, 185, 185), shadow_opacity=0.25,
                                  position=(float(path[:, 0].mean()), float(path[:, 1].mean()), 0.0))
            body = server.scene.add_mesh_simple("/body", vertices=vertices[0], faces=skinner.faces, color=m.ramp(0.0),
                                                side="double")
            follow = lambda t: (path[t] + np.array([*(-inputs.DIST * heading), inputs.RISE]), path[t])  # noqa: E731
            time.sleep(2.0)
            first_pose = render(client, px(2 * TILE), *follow(0), inputs.FOV)
            motion = None
            if e["mode"] == "motion_cond":
                motion = []
                for t in range(len(vertices)):
                    body.vertices, body.color = vertices[t], m.ramp(t / last)
                    time.sleep(0.1)
                    motion.append(render(client, px(2 * TILE), *follow(t), inputs.FOV))
            scene_frames = None
            if e["mode"] != "motion_cond":   # the generated motion in the video's 3D reconstruction
                server.scene.reset()
                server.scene.add_point_cloud("/scene", points=points.astype(np.float32), colors=colors,
                                             point_size=args.point_size, point_shape="circle", precision="float16")
                if not args.no_floor:
                    server.scene.add_grid("/floor", width=9.0, height=6.0, plane="xy", cell_size=0.5, section_size=1.0,
                                          cell_color=(215, 215, 215), section_color=(185, 185, 185),
                                          shadow_opacity=0.25, position=(float(middle[0]), float(middle[1]), 0.0))
                body = server.scene.add_mesh_simple("/body", vertices=vertices[0], faces=skinner.faces,
                                                    color=m.ramp(0.0), side="double")
                scene_frames = []
                for t in range(len(vertices)):
                    body.vertices, body.color = vertices[t], m.ramp(t / last)
                    time.sleep(0.1 if t else 2.0)
                    cam = orbit(position, look_at, ORBIT * (t / last - 0.5))
                    scene_frames.append(render(client, px(OUT_SIDE), cam, look_at, fov))
            segment = Segment(e, frames, first_pose, motion, scene_frames, captions[name])
            for i in range(segment.length()):
                frame = segment.draw(i)
                encoder.stdin.write(frame.tobytes())
                if poster is None and i == APPEAR + FLOW + REVEAL:
                    poster = frame
            total += segment.length()
            print(name, f"{segment.length()} frames, {time.time() - started:.0f} s", flush=True)
        encoder.stdin.close()
        encoder.wait()
        poster.save(out.with_suffix(".jpg"), quality=88)
        print(out, f"{out.stat().st_size / 1e6:.1f} MB, {total} frames, {total / FPS:.1f} s", flush=True)
    finally:
        if encoder.poll() is None:
            encoder.kill()
        chrome.terminate()
        server.stop()


if __name__ == "__main__":
    main()
