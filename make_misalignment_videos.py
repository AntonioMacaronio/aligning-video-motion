"""Comparison videos for "Why align video and motion?": where the generated head looks vs where the generated video's
camera looks, for finetuning alone (A v2 continued, 12.5k) and with the curriculum and post-training (K v2b, 12.5k), on
held-out clips.

    PYTHONPATH=<LVP tree> python make_misalignment_videos.py <eval dir> <camera-to-CPF table> <site dir> [clip ...]

<eval dir> holds the grid-2 evaluation's val64 outputs for the clips (synced from
s3://far-research-internal/antzhan/eval/motion-v2-grid2-20261020/val64/): media/<label>/<clip>_undistorted/
{joint.mp4, meta.json, vggt.npz} and motion/<label>/<clip>_undistorted.pt, for label acont_12500 and k2b_12500. Both
models sampled each clip from the same seed. The page's comparison uses the same layout from samples drawn with the
negative prompt in the video stream (dev node /data/sky/projects/large-video-planner/tmp/misalign-neg-20261110/:
neg_multi.py with each clip's evaluation seed, VGGT-Omega by joint_cameras.py; on S3 under
genvideomotion/thesis-site-scenes-20261030/misalign-neg-20261110/), with no camera/ shards, so its labels are the
per-frame means below.

The comparison is the thesis's camera-head consistency (camera_head_alignment_loss): the generated head's CPF pose on
the 49 video frames (motion_to_geometry_head_poses, head_kimodo_v2) and VGGT-Omega's cameras on the joint video
(vggt.npz, 448 px), the camera rotations taken relative to frame 0 and mapped into the head world through hr[0] and the
clip's calibrated R_{CPF<-camera}. Per model, one row of a WIDTH x HEIGHT frame:
- the generated video, framed in blue;
- a 3D view: the generated body (head-corrected MHR mesh) on the floor grid, followed from behind its first heading,
  with two view cones at its eyes: green, the camera the head implies (hr[t] R_{CPF<-camera}); blue, the camera the
  video shows (VGGT, in the head world); each with a ray along its optical axis;
- the turn since the first frame (yaw of the optical axis) of both, drawn as the clip plays;
and, as its label, the clip's self-consistency error from the evaluation itself (camera/<label>.shard*.json in <eval
dir>, VGGT-Omega on the decoded latent: the number the thesis's table averages), else the mean of the per-frame errors
above.
Writes <site>/media/misalignment/<clip>.mp4 (30 fps; the body moves at the motion's 30 fps, the video and the cones at
its 10) and a poster, summary.json (each model's mean error and final turns), and comparison.mp4, the clips in turn.
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import viser
import viser.transforms as vtf
from PIL import Image, ImageDraw, ImageFont

from algorithms.wan22.camera_to_cpf_table import load_camera_to_cpf_table
from algorithms.wan22.geometry_loss import motion_to_geometry_head_poses
from datasets.head_correction import correct_head_rotation, head_to_cpf_offset
from utils.mhr_mesh import MhrMeshSkinner, SUBJECT_IDENTITY_PATH, SubjectIdentities, subject_id_from_clip_name
from utils.motion_decode import motion_v2_to_joints

EVAL, TABLE, SITE = Path(sys.argv[1]), load_camera_to_cpf_table(Path(sys.argv[2])), Path(sys.argv[3])
CLIPS = sys.argv[4:] or ["20230907_s0_margaret_mccormick_act2_mgk2q0_00020", "20231010_s0_albert_hammond_act2_op7b8a_00149",
                         "20230907_s0_margaret_mccormick_act1_rwed0h_00169", "20231010_s1_cheryl_johnson_act2_xssrc8_00145"]
MODELS = (("acont_12500", "Finetuning", ""),
          ("k2b_12500", "Curriculum + post-training", ""))
WIDTH, HEIGHT, FPS = 1920, 1080, 30
SIDE, PLOT_W = 420, 640                      # px: the video and the 3D view; the plot's width
LABEL_W, X0, ROW_Y = 330, 40, (150, 600)     # the rows' label column, the left margin, each row's top
INK, MUTED, RULE = (31, 35, 40), (100, 116, 139), (220, 223, 228)
HEAD_C, CAMERA_C = (4, 150, 105), (37, 99, 235)
BODY_C = (188, 192, 200)
FONTS = "/System/Library/Fonts/HelveticaNeue.ttc"   # faces: 0 regular, 1 bold, 2 italic
DIST, RISE, AHEAD = 2.4, 1.1, 0.6            # m: the 3D camera behind and above the head, looking this far ahead of it


def font(size, face=0):
    return ImageFont.truetype(FONTS, size, index=face)


def rounded(size, radius=12):
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius, fill=255)
    return mask


def look_at_rotation(position, target):
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(forward, right), forward], axis=1)


def official(clip, label):
    """The evaluation's self-consistency for the clip (camera/<label>.shard*.json): VGGT-Omega on the decoded latent, the
    number the thesis's table averages; None if the shards are not in <eval dir>."""
    for path in sorted((EVAL / "camera").glob(f"{label}.shard*.json")):
        for c in json.loads(path.read_text())["clips"]:
            if c["sample_id"] == clip + "_undistorted":
                return float(c["joint_video_vs_joint_head"]["rotation_loss_deg"])
    return None


def analyse(clip, label):
    """Video frames, body vertices (145), and per video frame: head position, the two cameras' rotations, yaws."""
    sid = clip + "_undistorted"
    media = EVAL / "media" / label / sid
    meta = json.loads((media / "meta.json").read_text())
    motion = torch.load(EVAL / "motion" / label / f"{sid}.pt", map_location="cpu", weights_only=False)["joint"]
    hp, hr = motion_to_geometry_head_poses(motion.float()[None], 49, head_track="v2")
    hp, hr = hp[0].double().numpy(), hr[0].double().numpy()
    extrinsics = np.load(media / "vggt.npz")["joint_extrinsics"].astype(np.float64)
    a = TABLE.resolve([meta["device_serial"]])[0][0].double().numpy()     # R_{CPF<-camera}
    world_from_camera = np.transpose(extrinsics[:, :3, :3], (0, 2, 1))
    camera = hr[0] @ a @ (world_from_camera[0].T @ world_from_camera)     # the video's camera, in the head world
    head = hr @ a                                                          # the camera the head implies
    rel = np.einsum("tji,tjk->tik", camera @ a.T, hr)
    error = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))

    def yaw(r):
        f = r[:, :, 2]
        y = np.degrees(np.unwrap(np.arctan2(f[:, 1], f[:, 0])))
        return y - y[0]

    raw = motion.double().numpy()
    corrected = correct_head_rotation(raw, head_to_cpf_offset(sid))
    joints, _, _ = motion_v2_to_joints(corrected)
    vertices = SKINNER.from_motion_v2(corrected, IDENTITIES.bind_state(subject_id_from_clip_name(sid)),
                                      positions_mode="fk", joints=joints).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(media / "joint.mp4"), f"{tmp}/%02d.png"], check=True)
        frames = [Image.open(p).convert("RGB") for p in sorted(Path(tmp).glob("*.png"))]
    return dict(meta=meta, frames=frames, vertices=vertices, hp=hp, head=head, camera=camera, error=error,
                official=official(clip, label),
                head_yaw=yaw(head), camera_yaw=yaw(camera))


def follow(hp, heading, k):
    """The 3D view's camera at video frame k: behind the (smoothed) head along the first heading, looking ahead."""
    lo, hi = max(0, k - 4), min(len(hp), k + 5)
    centre = hp[lo:hi].mean(0)
    target = centre + AHEAD * np.array([*heading, 0.0]) - np.array([0, 0, 0.35])
    position = centre + np.array([*(-DIST * heading), RISE])
    return position, target


def render_3d(server, client, d):
    """The 3D view for every motion frame."""
    hp, verts = d["hp"], d["vertices"]
    f0 = d["head"][0][:, 2]
    heading = f0[:2] / np.linalg.norm(f0[:2])
    server.scene.reset()
    server.scene.add_grid("/floor", width=20.0, height=20.0, plane="xy", cell_size=0.5, section_size=1.0,
                          cell_color=(215, 215, 215), section_color=(185, 185, 185), shadow_opacity=0.25,
                          position=(float(hp[:, 0].mean()), float(hp[:, 1].mean()), 0.0))
    body = server.scene.add_mesh_simple("/body", vertices=verts[0], faces=SKINNER.faces, color=BODY_C, side="double")
    cones = {}
    for name, color in (("head", HEAD_C), ("camera", CAMERA_C)):
        cones[name] = server.scene.add_camera_frustum(f"/{name}", fov=np.radians(55), aspect=1.0, scale=0.32,
                                                      line_width=4.0, color=color)
    ray_colors = np.array([[HEAD_C, HEAD_C], [CAMERA_C, CAMERA_C]], np.uint8)
    out = []
    for t in range(len(verts)):
        k = min(t // 3, len(hp) - 1)
        body.vertices = verts[t]
        segments = []
        for name in ("head", "camera"):
            r = d[name][k]
            cones[name].wxyz = vtf.SO3.from_matrix(r).wxyz
            cones[name].position = hp[k]
            segments.append([hp[k], hp[k] + 1.6 * r[:, 2]])
        server.scene.add_line_segments("/rays", points=np.asarray(segments, np.float32), colors=ray_colors,
                                       line_width=4.0)   # the same name replaces last frame's rays
        time.sleep(0.1 if t else 2.0)
        position, target = follow(hp, heading, k)
        wxyz = vtf.SO3.from_matrix(look_at_rotation(position, target)).wxyz
        rgba = Image.fromarray(client.get_render(SIDE, SIDE, wxyz=wxyz, position=position, fov=np.radians(55),
                                                 transport_format="png")).convert("RGBA")
        out.append(Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB"))
    return out


class Plot:
    """The turn since the first frame of both cameras, drawn up to the current frame, on axes shared by both rows."""

    def __init__(self, rows, w=PLOT_W, h=SIDE):
        values = np.concatenate([np.concatenate([d["head_yaw"], d["camera_yaw"]]) for d in rows])
        lo, hi = float(values.min()), float(values.max())
        pad = max(10.0, 0.08 * (hi - lo))
        self.lo, self.hi, self.w, self.h = lo - pad, hi + pad, w, h
        self.left, self.right, self.top, self.bottom = 86, w - 20, 20, h - 56
        span = self.hi - self.lo
        self.step = next(s for s in (15, 30, 45, 90, 180) if span / s <= 7)

    def xy(self, k, v):
        x = self.left + (self.right - self.left) * k / 48
        y = self.bottom - (self.bottom - self.top) * (v - self.lo) / (self.hi - self.lo)
        return x, y

    def draw(self, d, k):
        img = Image.new("RGB", (self.w, self.h), "white")
        g = ImageDraw.Draw(img)
        for v in range(int(np.ceil(self.lo / self.step)) * self.step, int(self.hi) + 1, self.step):
            y = self.xy(0, v)[1]
            g.line((self.left, y, self.right, y), fill=(236, 238, 241) if v else (190, 196, 204), width=2 if not v else 1)
            g.text((self.left - 12 - g.textlength(f"{v}°", font=font(26)), y - 15), f"{v}°", font=font(26), fill=MUTED)
        for s in range(0, 5):
            x = self.xy(s * 10, 0)[0] if s * 10 <= 48 else None
            if x is not None:
                g.text((x - 12, self.bottom + 10), f"{s} s", font=font(26), fill=MUTED)
        for name, color in (("head_yaw", HEAD_C), ("camera_yaw", CAMERA_C)):
            faint = tuple(int(c + (255 - c) * 0.78) for c in color)
            pts = [self.xy(i, v) for i, v in enumerate(d[name])]
            g.line(pts, fill=faint, width=3, joint="curve")
            g.line(pts[:k + 1] if k > 0 else pts[:1] * 2, fill=color, width=5, joint="curve")
            x, y = pts[k]
            g.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)
        x = self.xy(k, 0)[0]
        g.line((x, self.top, x, self.bottom), fill=(203, 213, 225), width=2)
        return img


def compose(clip, rows, renders, plot, t):
    k = min(t // 3, 48)
    out = Image.new("RGB", (WIDTH, HEIGHT), "white")
    g = ImageDraw.Draw(out)
    # the prompt, centred, on one line or wrapped onto two (smaller); the legend under it
    caption = f"“{rows[0]['meta']['caption']}”"
    face = font(32, 2)
    lines = [caption]
    if g.textlength(caption, font=face) > WIDTH - 2 * X0:
        face, lines = font(25, 2), [""]
        for word in caption.split():
            if lines[-1] and g.textlength(f"{lines[-1]} {word}", font=face) > WIDTH - 2 * X0:
                lines.append(word)
            else:
                lines[-1] = f"{lines[-1]} {word}".strip()
    top = 34 if len(lines) == 1 else 14
    for line in lines:
        g.text(((WIDTH - g.textlength(line, font=face)) / 2, top), line, font=face, fill=INK)
        top += face.size + 8
    legend_y = max(88, top + 8)
    legend = [("where the generated head looks", HEAD_C), ("where the generated video's camera looks (VGGT-Ω)", CAMERA_C)]
    x = (WIDTH - sum(g.textlength(s, font=font(28)) + 64 for s, _ in legend)) / 2
    for s, color in legend:
        g.rounded_rectangle((x, legend_y + 10, x + 32, legend_y + 24), 4, fill=color)
        g.text((x + 42, legend_y), s, font=font(28), fill=INK)
        x += g.textlength(s, font=font(28)) + 64
    for (label, title, subtitle), d, frames3d, y in zip(MODELS, rows, renders, ROW_Y):
        # the label: the model's name (wrapped to the column), what it adds, and the clip's error
        words, lines = title.split(), []
        for word in words:
            if lines and g.textlength(f"{lines[-1]} {word}", font=font(34, 1)) <= LABEL_W - 16:
                lines[-1] = f"{lines[-1]} {word}"
            else:
                lines.append(word)
        top = y + 40
        for line in lines:
            g.text((X0, top), line, font=font(34, 1), fill=INK)
            top += 42
        for line in filter(None, subtitle.split("\n")):
            g.text((X0, top + 6), line, font=font(26), fill=MUTED)
            top += 32
        apart = d["official"] if d["official"] is not None else float(d["error"].mean())
        g.text((X0, top + 40), "head vs camera", font=font(26), fill=MUTED)
        g.text((X0, top + 72), f"{apart:.0f}° apart", font=font(46, 1), fill=INK)
        x = X0 + LABEL_W
        video = Image.new("RGB", (SIDE, SIDE), CAMERA_C)
        inner = d["frames"][k].resize((SIDE - 16, SIDE - 16), Image.LANCZOS)
        video.paste(inner, (8, 8), rounded(inner.size, 8))
        out.paste(video, (x, y), rounded(video.size))
        view = frames3d[t]
        out.paste(view, (x + SIDE + 24, y), rounded(view.size))
        g.rounded_rectangle((x + SIDE + 24, y, x + 2 * SIDE + 23, y + SIDE - 1), 12, outline=RULE, width=2)
        out.paste(plot.draw(d, k), (x + 2 * SIDE + 48, y))
    g.text((X0 + LABEL_W + 2 * SIDE + 48 + 86, ROW_Y[1] + SIDE + 26), "turn since the first frame",
           font=font(26), fill=MUTED)
    return out


def main():
    global SKINNER, IDENTITIES
    SKINNER, IDENTITIES = MhrMeshSkinner(), SubjectIdentities.load(SUBJECT_IDENTITY_PATH)
    out_dir = SITE / "media" / "misalignment"
    out_dir.mkdir(parents=True, exist_ok=True)
    server = viser.ViserServer(host="127.0.0.1", port=9493, verbose=False)
    server.scene.set_up_direction("+z")
    chrome = subprocess.Popen(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--headless=new",
                               "--use-angle=metal", f"--window-size={SIDE},{SIDE}", f"--user-data-dir={tempfile.mkdtemp()}",
                               f"http://127.0.0.1:{server.get_port()}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    summary = {}
    try:
        deadline = time.time() + 60
        while not server.get_clients() and time.time() < deadline:
            time.sleep(0.5)
        client = next(iter(server.get_clients().values()))
        for clip in CLIPS:
            started = time.time()
            rows = [analyse(clip, label) for label, _, _ in MODELS]
            renders = [render_3d(server, client, d) for d in rows]
            plot = Plot(rows)
            out = out_dir / f"{clip}.mp4"
            encoder = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                                        f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "23",
                                        "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
                                        str(out)], stdin=subprocess.PIPE)
            for t in range(len(rows[0]["vertices"])):
                frame = compose(clip, rows, renders, plot, t)
                encoder.stdin.write(frame.tobytes())
                if t == 96:
                    frame.save(out.with_suffix(".jpg"), quality=88)
            encoder.stdin.close()
            encoder.wait()
            if clip == CLIPS[0]:
                (out_dir / "comparison.jpg").write_bytes(out.with_suffix(".jpg").read_bytes())
            summary[clip] = {label: dict(mean_deg=round(float(d["error"].mean()), 1), evaluation_deg=d["official"],
                                         head_turn_deg=round(float(d["head_yaw"][-1]), 1),
                                         camera_turn_deg=round(float(d["camera_yaw"][-1]), 1))
                             for (label, _, _), d in zip(MODELS, rows)}
            print(clip, summary[clip], f"{out.stat().st_size / 1e6:.1f} MB, {time.time() - started:.0f} s", flush=True)
    finally:
        chrome.terminate()
        server.stop()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    # all the clips, one after another, for the page (same encoding, so the streams are copied)
    listing = out_dir / "comparison.txt"
    listing.write_text("".join(f"file '{clip}.mp4'\n" for clip in CLIPS))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy",
                    "-movflags", "+faststart", str(out_dir / "comparison.mp4")], check=True)
    listing.unlink()


if __name__ == "__main__":
    main()
