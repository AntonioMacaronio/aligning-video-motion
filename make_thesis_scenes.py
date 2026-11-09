"""Animated scenes of the thesis's Section 4.4 gallery for the site's main page.

    PYTHONPATH=<LVP tree with datasets/head_correction.py> python make_thesis_scenes.py \
        <gallery dir> <figure-src gallery dir> <site dir> [--tasks <tasks dir>] [--videos-only | --cameras-only] [name ...]

<gallery dir> holds gallery_scene.py's scenes/<name>/{scene.npz,frames/} and the examples' inputs
(<cohort>/motion/<clip>.pt); <figure-src gallery dir> is the thesis repo's figure-src/gallery-44-20261025
(examples.json, views.json, cameras/), and its sibling data-figure-20261023/render_datafig.py supplies the figure's
color ramp and occluder cut. For each of the twelve examples:
- the point cloud is cut exactly as for the thesis figure: render_all.sh's flags, the example's views.json options and
  its figure camera (cameras/<name>.json, --cut-from-view), at the figure's aspect ratio;
- the body is the example's motion at every one of the 145 motion frames, posed with gallery_scene.py's recipe (the
  glasses-anchored head, the subject's identity, the joint rotations; checked against the scene's stored snapshots)
  and colored along the figure's green -> blue -> purple ramp by time;
- the video is the example's 49 frames (scenes/<name>/frames) as a 10 fps mp4 in which every frame is a keyframe,
  because the page seeks it to the scene's playback time on every frame (video frame k = motion frame 3k).
Writes <site>/recordings/thesis/<name>.viser and .json (the figure camera, and the site's opening camera: see
site_camera) and <site>/media/thesis/<name>.mp4; --videos-only rewrites just the videos, --cameras-only just the site
cameras in the .json files.

--tasks adds each example's other two tasks (the page's tabs): <tasks dir>/examples_tasks.json lists them as
<example>__<task> with the evaluation's samples of the same clip, and <tasks dir> holds their scenes (built by
gallery_scene.py, or gallery_scene_local.py on a laptop) and inputs. Each uses its example's figure camera and
views.json options, so the three tasks of a clip open at the same view, unless <tasks dir>/cameras/<name>.json holds a
view saved for that tab in the figure viewer (render_datafig.py --interactive), which then replaces the figure camera.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import viser
import viser.transforms as vtf
from scipy.spatial import cKDTree

GALLERY, FIGSRC, SITE = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
_rest = sys.argv[4:]
TASKS = None
if "--tasks" in _rest:
    _i = _rest.index("--tasks")
    TASKS = Path(_rest[_i + 1])
    _rest = _rest[:_i] + _rest[_i + 2:]
VIDEOS_ONLY = "--videos-only" in _rest
CAMERAS_ONLY = "--cameras-only" in _rest
NAMES = [name for name in _rest if name not in ("--videos-only", "--cameras-only")]
sys.path.insert(0, str(FIGSRC.parent / "data-figure-20261023"))
from render_datafig import occluders, ramp  # noqa: E402  the figure's own color ramp and occluder cut

from datasets.head_correction import correct_head_rotation, head_to_cpf_offset  # noqa: E402
from utils.mhr_mesh import MhrMeshSkinner, SUBJECT_IDENTITY_PATH, SubjectIdentities, subject_id_from_clip_name  # noqa: E402
from utils.motion_decode import motion_v2_to_joints  # noqa: E402

FPS = 30.0
MAX_POINTS = 300_000
# render_all.sh's flags for every example, before the example's views.json options
COMMON = ["--align-floor", "--min-opacity", "0.7", "--min-z", "-0.08", "--drop-occluders", "--self-radius", "0.07",
          "--color-gamma", "0.8", "--color-max", "235"]


def figure_options(argv):
    """The subset of render_datafig.py's options that shapes the scene, with its defaults."""
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=2400)
    p.add_argument("--height", type=int, default=1000)
    p.add_argument("--distance", type=float, default=4.2)
    p.add_argument("--elevation", type=float, default=1.6)
    p.add_argument("--view", default="side")
    p.add_argument("--view-yaw", type=float, default=None)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--pick", type=str, default=None)
    p.add_argument("--fov", type=float, default=50.0)
    p.add_argument("--front-margin", type=float, default=0.4)
    p.add_argument("--ceiling", type=float, default=2.3)
    p.add_argument("--min-opacity", type=float, default=0.25)
    p.add_argument("--no-floor", action="store_true")
    p.add_argument("--drop-occluders", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--hide-behind-bodies", action="store_true")
    p.add_argument("--occlusion-margin", type=float, default=0.05)
    p.add_argument("--self-radius", type=float, default=0.0)
    p.add_argument("--color-gamma", type=float, default=1.0)
    p.add_argument("--color-max", type=int, default=255)
    p.add_argument("--align-floor", action="store_true")
    p.add_argument("--min-z", type=float, default=-0.2)
    p.add_argument("--point-size", type=float, default=0.012)
    p.add_argument("--camera-json", type=str, default=None)
    p.add_argument("--yaw-offset", type=float, default=0.0)
    p.add_argument("--max-range", type=float, default=None)
    p.add_argument("--look-height", type=float, default=0.9)
    p.add_argument("--cut-from-view", action="store_true")
    return p.parse_known_args(argv)[0]


def figure_points(scene, args):
    """render_datafig.py's camera and point cuts, line for line; returns the kept points and colors, and the grid centre."""
    if args.pick:
        frames = [int(f) for f in args.pick.split(",")]
        index = [int(np.flatnonzero(scene["snapshots"] == f)[0]) for f in frames]
        scene["vertices"], scene["snapshots"] = scene["vertices"][index], scene["snapshots"][index]
    head = scene["head"]
    walk = head[-1, :2] - head[0, :2]
    walk /= np.linalg.norm(walk)
    d = np.array([walk[0], walk[1], 0.0])
    forward = np.cross([0.0, 0.0, 1.0], d)
    if args.view == "behind":
        facing = scene["head_rot"][:, :2, 2]
        yaw = np.arctan2(facing[:, 1], facing[:, 0])
        yaw = np.radians(args.view_yaw) if args.view_yaw is not None else np.arctan2(np.sin(yaw).mean(), np.cos(yaw).mean())
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    if args.view == "chase":
        travel = head[:, :2] - head[0, :2]
        reach = np.linalg.norm(travel, axis=1)
        if reach.max() > 0.5:
            out = travel[int(np.argmax(reach > 0.5 * reach.max()))]
            yaw = np.arctan2(out[1], out[0])
        else:
            facing = scene["head_rot"][:, :2, 2]
            yaw = np.arctan2(np.sin(np.arctan2(facing[:, 1], facing[:, 0])).mean(),
                             np.cos(np.arctan2(facing[:, 1], facing[:, 0])).mean())
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    if args.yaw_offset:
        c, s = np.cos(np.radians(args.yaw_offset)), np.sin(np.radians(args.yaw_offset))
        forward = np.array([c * forward[0] - s * forward[1], s * forward[0] + c * forward[1], 0.0])
    middle = np.array([*(0.5 * (head[0, :2] + head[-1, :2])), args.look_height])
    if args.view == "chase":
        middle = np.array([*head[0, :2], args.look_height])
    saved = json.loads(Path(args.camera_json).read_text())
    position, wxyz, fov = np.array(saved["position"]), np.array(saved["wxyz"]), float(saved["fov"])
    if not args.cut_from_view:
        gaze = vtf.SO3(wxyz).as_matrix()[:, 2]
        forward = np.array([gaze[0], gaze[1], 0.0]) / np.linalg.norm(gaze[:2])

    points, colors = scene["points"], scene["colors"]
    if args.align_floor:
        low = points[points[:, 2] < 0.3, 2]
        counts, edges = np.histogram(low, bins=np.arange(-0.3, 0.3, 0.02))
        floor = float(edges[counts.argmax()] + 0.01)
        points = points - np.array([0.0, 0.0, floor], dtype=points.dtype)
    toward_camera = (points - middle) @ (-forward)
    keep = (toward_camera < args.front_margin) & (points[:, 2] < args.ceiling) & (points[:, 2] > args.min_z)
    if args.max_range is not None:
        keep &= cKDTree(head[:, :2]).query(points[:, :2])[0] < args.max_range
    if args.self_radius > 0:
        body = np.concatenate([scene["joints"].reshape(-1, 3), scene["vertices"].reshape(-1, 3)])
        keep &= ~(cKDTree(body).query(points, distance_upper_bound=args.self_radius)[0] < np.inf)
    if args.drop_occluders:
        keep &= ~occluders(points, scene["vertices"][::args.every], scene["faces"], position,
                           vtf.SO3(wxyz).as_matrix(), fov, args.width, args.height, args.occlusion_margin,
                           behind=args.hide_behind_bodies)
    points, colors = points[keep], colors[keep]
    if args.color_gamma != 1.0:
        colors = (255 * (colors / 255.0) ** args.color_gamma).round().astype(np.uint8)
    if args.color_max < 255:
        colors = (colors.astype(np.float64) * args.color_max / 255.0).round().astype(np.uint8)
    return points, colors, middle, saved


def look_at_rotation(position, target):
    """Camera-to-world rotation (OpenCV axes: x right, y down, z forward) of a camera at position looking at target, z up."""
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(forward, right), forward], axis=1)


def site_camera(points, bodies, view, fill=0.88):
    """The site's opening camera: compose_gallery.py's --tight-crop as a camera. It stays at the figure camera's
    position, turns to the centre of the content's box and narrows its field of view until the box fills `fill` of
    the view, so the site shows what the figure shows, at the same perspective. The content is the cut point cloud
    (1-99% of it, and none within 1 m of the camera, so stray and near-plane points do not count) and every stored body
    snapshot (all of it). The look-at point, where the viewer orbits, sits on the centre ray at the content's depth; the
    centring is damped, since a cloud spread in depth does not move linearly in the image as the camera turns. Returns
    (position, look_at, vertical fov)."""
    position = np.asarray(view["position"], float)
    t, aspect = np.tan(float(view["fov"]) / 2), float(view.get("aspect") or 1.8)
    bodies = bodies.reshape(-1, 3)
    centroid = np.concatenate([points, bodies]).mean(0)
    look_at = np.asarray(view["look_at"], float)

    def box(rotation):
        def extent(xyz, low, high, near):
            cam = (xyz - position) @ rotation
            cam = cam[cam[:, 2] > near]
            x, y = cam[:, 0] / cam[:, 2] / (t * aspect), cam[:, 1] / cam[:, 2] / t
            return np.percentile(x, low), np.percentile(x, high), np.percentile(y, low), np.percentile(y, high)
        p, q = extent(points, 1, 99, 1.0), extent(bodies, 0, 100, 0.05)
        return min(p[0], q[0]), max(p[1], q[1]), min(p[2], q[2]), max(p[3], q[3])

    for _ in range(12):  # aim at the box centre, half a step at a time
        rotation = look_at_rotation(position, look_at)
        x0, x1, y0, y1 = box(rotation)
        ray = rotation @ np.array([t * aspect * (x0 + x1) / 4, t * (y0 + y1) / 4, 1.0])
        ray /= np.linalg.norm(ray)
        look_at = position + ray * float((centroid - position) @ ray)
    x0, x1, y0, y1 = box(look_at_rotation(position, look_at))
    zoom = float(np.clip(fill / max(abs(x0), abs(x1), abs(y0), abs(y1)), 0.65, 4.0))
    return position, look_at, float(2 * np.arctan(t / zoom))


def encode_video(scene_dir, video):
    """The example's 49 frames as a 10 fps H.264 mp4, all keyframes so that seeking to any frame is immediate."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "10", "-i", str(scene_dir / "frames" / "%02d.jpg"),
                    "-vf", "scale=640:640:flags=lanczos", "-c:v", "libx264", "-crf", "23", "-preset", "slow", "-g", "1",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(video)], check=True)


def figure_scene(name, base, root):
    """A scene's cloud as its figure cuts it: (scene.npz's arrays, points, colors, the floor's middle, the figure
    options, the figure camera's view, and the site's opening camera (position, look_at, fov))."""
    scene = dict(np.load(root / "scenes" / name / "scene.npz"))
    camera = FIGSRC / "cameras" / f"{base}.json"
    if root is TASKS and (TASKS / "cameras" / f"{name}.json").exists():  # a tab with its own saved view
        camera = TASKS / "cameras" / f"{name}.json"
    views = json.loads((FIGSRC / "views.json").read_text())
    extras = views.get(base, views["default"])
    height = round(2400 / (json.loads(camera.read_text()).get("aspect") or 2.4))
    argv = ["--width", "2400", "--height", str(height), *COMMON,
            *([] if "--pick" in extras else ["--every", "2"]), *extras, "--camera-json", str(camera), "--cut-from-view"]
    args = figure_options(argv)
    stored = scene["vertices"].copy()
    points, colors, middle, view = figure_points(dict(scene), args)
    return scene, points, colors, middle, args, view, site_camera(points, stored, view)


def thin(points, colors):
    """At most MAX_POINTS of the cloud, the same ones on every run."""
    if len(points) > MAX_POINTS:
        pick = np.sort(np.random.default_rng(0).choice(len(points), MAX_POINTS, replace=False))
        points, colors = points[pick], colors[pick]
    return points, colors


def posed_body(root, example, scene, skinner, identities):
    """The body at every motion frame, as gallery_scene.py posed its snapshots, and its largest distance from them."""
    clip = example["clip"]
    motion = torch.load(root / example["cohort"] / "motion" / f"{clip}.pt", map_location="cpu",
                        weights_only=False)[example["motion"]].double().numpy()
    motion = correct_head_rotation(motion, head_to_cpf_offset(clip))
    joints, _, _ = motion_v2_to_joints(motion)
    vertices = skinner.from_motion_v2(motion, identities.bind_state(subject_id_from_clip_name(clip)),
                                      positions_mode="fk", joints=joints).astype(np.float32)
    return vertices, float(np.abs(vertices[scene["snapshots"]] - scene["vertices"]).max())


def main():
    examples = json.loads((FIGSRC / "examples.json").read_text())
    skinner, identities = MhrMeshSkinner(), SubjectIdentities.load(SUBJECT_IDENTITY_PATH)
    (SITE / "recordings" / "thesis").mkdir(parents=True, exist_ok=True)
    (SITE / "media" / "thesis").mkdir(parents=True, exist_ok=True)
    # (name, the example whose figure camera and views it uses, its scene directory's root, its entry)
    jobs = [(e["name"], e["name"], GALLERY, e) for e in examples]
    if TASKS is not None:
        jobs += [(e["name"], e["example"], TASKS, e) for e in json.loads((TASKS / "examples_tasks.json").read_text())]
    port = 9400
    for name, base, root, example in jobs:
        if NAMES and name not in NAMES:
            continue
        scene_dir = root / "scenes" / name
        video = SITE / "media" / "thesis" / f"{name}.mp4"
        if VIDEOS_ONLY:
            encode_video(scene_dir, video)
            print(name, f"video {video.stat().st_size / 1e6:.2f} MB", flush=True)
            continue
        scene, points, colors, middle, args, view, (site_position, site_look_at, site_fov) = figure_scene(name, base, root)
        if CAMERAS_ONLY:
            target = SITE / "recordings" / "thesis" / f"{name}.json"
            info = json.loads(target.read_text())
            info.update(site_position=np.round(site_position, 3).tolist(), site_look_at=np.round(site_look_at, 3).tolist(),
                        site_fov=round(site_fov, 4))
            target.write_text(json.dumps(info))
            print(name, "site camera", info["site_look_at"], f"fov {np.degrees(site_fov):.1f} deg", flush=True)
            continue
        points, colors = thin(points, colors)
        vertices, error = posed_body(root, example, scene, skinner, identities)
        assert error < 1e-3, f"{name}: posed body differs from the figure's snapshots by {error:.4f} m"

        server = viser.ViserServer(host="127.0.0.1", port=port, verbose=False)
        server.scene.set_up_direction("+z")
        serializer = server.get_scene_serializer()
        server.scene.add_point_cloud("/scene", points=points.astype(np.float32), colors=colors,
                                     point_size=args.point_size, point_shape="circle", precision="float16")
        if not args.no_floor:
            server.scene.add_grid("/floor", width=9.0, height=6.0, plane="xy", cell_size=0.5, section_size=1.0,
                                  cell_color=(215, 215, 215), section_color=(185, 185, 185), shadow_opacity=0.25,
                                  position=(float(middle[0]), float(middle[1]), 0.0))
        last = len(vertices) - 1
        body = server.scene.add_mesh_simple("/body", vertices=vertices[0], faces=skinner.faces, color=ramp(0.0),
                                            side="double")
        for t in range(len(vertices)):
            body.vertices = vertices[t]
            body.color = ramp(t / last)
            serializer.insert_sleep(1.0 / FPS)
        target = SITE / "recordings" / "thesis" / f"{name}.viser"
        target.write_bytes(serializer.serialize())
        server.stop()
        info = dict(position=np.round(view["position"], 3).tolist(), look_at=np.round(view["look_at"], 3).tolist(),
                    up=[0.0, 0.0, 1.0], fov=float(view["fov"]), aspect=float(view.get("aspect") or 1.8),
                    site_position=np.round(site_position, 3).tolist(), site_look_at=np.round(site_look_at, 3).tolist(),
                    site_fov=round(site_fov, 4),
                    points=int(len(points)), frames=len(vertices), snapshot_error_m=error,
                    megabytes=round(target.stat().st_size / 1e6, 2))
        target.with_suffix(".json").write_text(json.dumps(info))

        encode_video(scene_dir, video)
        print(name, json.dumps(info), f"video {video.stat().st_size / 1e6:.2f} MB", flush=True)
        port += 1


if __name__ == "__main__":
    main()
