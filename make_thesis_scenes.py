"""Animated scenes of the thesis's Section 4.4 gallery for the site's main page.

    PYTHONPATH=<LVP tree with datasets/head_correction.py> python make_thesis_scenes.py \
        <gallery dir> <figure-src gallery dir> <site dir> [name ...]

<gallery dir> holds gallery_scene.py's scenes/<name>/{scene.npz,frames/} and the examples' inputs
(<cohort>/motion/<clip>.pt); <figure-src gallery dir> is the thesis repo's figure-src/gallery-44-20261025
(examples.json, views.json, cameras/), and its sibling data-figure-20261023/render_datafig.py supplies the figure's
color ramp and occluder cut. For each of the twelve examples:
- the point cloud is cut exactly as for the thesis figure: render_all.sh's flags, the example's views.json options and
  its figure camera (cameras/<name>.json, --cut-from-view), at the figure's aspect ratio;
- the body is the example's motion at every one of the 145 motion frames, posed with gallery_scene.py's recipe (the
  glasses-anchored head, the subject's identity, the joint rotations; checked against the scene's stored snapshots)
  and colored along the figure's green -> blue -> purple ramp by time;
- the video is the example's 49 frames (scenes/<name>/frames) as a 10 fps mp4.
Writes <site>/recordings/thesis/<name>.viser and .json (the figure camera) and <site>/media/thesis/<name>.mp4.
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
NAMES = sys.argv[4:]
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


def main():
    examples = json.loads((FIGSRC / "examples.json").read_text())
    views = json.loads((FIGSRC / "views.json").read_text())
    skinner, identities = MhrMeshSkinner(), SubjectIdentities.load(SUBJECT_IDENTITY_PATH)
    (SITE / "recordings" / "thesis").mkdir(parents=True, exist_ok=True)
    (SITE / "media" / "thesis").mkdir(parents=True, exist_ok=True)
    port = 9400
    for example in examples:
        name, clip = example["name"], example["clip"]
        if NAMES and name not in NAMES:
            continue
        scene_dir = GALLERY / "scenes" / name
        scene = dict(np.load(scene_dir / "scene.npz"))
        camera = FIGSRC / "cameras" / f"{name}.json"
        extras = views.get(name, views["default"])
        height = round(2400 / (json.loads(camera.read_text()).get("aspect") or 2.4))
        argv = ["--width", "2400", "--height", str(height), *COMMON,
                *([] if "--pick" in extras else ["--every", "2"]), *extras, "--camera-json", str(camera), "--cut-from-view"]
        args = figure_options(argv)
        snapshots, stored = scene["snapshots"].copy(), scene["vertices"].copy()
        points, colors, middle, view = figure_points(dict(scene), args)
        if len(points) > MAX_POINTS:
            pick = np.random.default_rng(0).choice(len(points), MAX_POINTS, replace=False)
            points, colors = points[np.sort(pick)], colors[np.sort(pick)]

        # The body at every frame, as gallery_scene.py posed its snapshots.
        motion = torch.load(GALLERY / example["cohort"] / "motion" / f"{clip}.pt", map_location="cpu",
                            weights_only=False)[example["motion"]].double().numpy()
        motion = correct_head_rotation(motion, head_to_cpf_offset(clip))
        joints, _, _ = motion_v2_to_joints(motion)
        vertices = skinner.from_motion_v2(motion, identities.bind_state(subject_id_from_clip_name(clip)),
                                          positions_mode="fk", joints=joints).astype(np.float32)
        error = float(np.abs(vertices[snapshots] - stored).max())
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
                    points=int(len(points)), frames=len(vertices), snapshot_error_m=error,
                    megabytes=round(target.stat().st_size / 1e6, 2))
        target.with_suffix(".json").write_text(json.dumps(info))

        video = SITE / "media" / "thesis" / f"{name}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "10", "-i", str(scene_dir / "frames" / "%02d.jpg"),
                        "-vf", "scale=640:640:flags=lanczos", "-c:v", "libx264", "-crf", "23", "-preset", "slow",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(video)], check=True)
        print(name, json.dumps(info), f"video {video.stat().st_size / 1e6:.2f} MB", flush=True)
        port += 1


if __name__ == "__main__":
    main()
