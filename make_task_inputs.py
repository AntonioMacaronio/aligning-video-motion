"""The viewer's input panels: each Section 4.4 example's recorded motion as a short video in the 3D scenes' look.

    PYTHONPATH=<LVP tree with datasets/head_correction.py> python make_task_inputs.py \
        <gallery dir> <figure-src gallery dir> <site dir> --tasks <tasks dir> [name ...]

The arguments are make_thesis_scenes.py's. For each example, the recorded motion of its motion-conditioned task (the
example itself when that is its figure's task, else <example>__motion_cond in the tasks list) is posed exactly as in
its scene. viser renders it in headless Chrome the way the scenes look: the shaded MHR body on the floor grid, green to
purple over time. A camera follows the body from the figure camera's side, so the body stays large in a small panel.
Writes <site>/media/thesis/<example>__input_motion.mp4: 49 frames at 10 fps, motion frame 3k at frame k, all keyframes,
like the scene videos, so the page can seek it in step with the 3D scene. Its first frame is the starting pose, which
the viewer shows as an input on every tab (build_site.py makes it the video's poster).
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
from PIL import Image

import make_thesis_scenes as m   # parses the same command line

SIZE, OUT = 480, 360         # px, square: rendered, and encoded (the panel shows it at about 150 px)
DIST, RISE = 2.9, 0.9      # m: camera distance behind the body's centre, and height above it
FOV = np.radians(50.0)
SMOOTH = 15                 # motion frames: the camera follows a moving average of the body's centre


def follow_path(vertices):
    """The body's centre per motion frame, smoothed so the camera does not sway with each step; it rises with the body,
    so a climb stays in frame."""
    centre = vertices.mean(1)
    pad = np.pad(centre, ((SMOOTH // 2, SMOOTH // 2), (0, 0)), mode="edge")
    kernel = np.ones(SMOOTH) / SMOOTH
    return np.stack([np.convolve(pad[:, i], kernel, mode="valid") for i in range(3)], 1)


def main():
    examples = json.loads((m.FIGSRC / "examples.json").read_text())
    tasks = {e["name"]: e for e in json.loads((m.TASKS / "examples_tasks.json").read_text())} if m.TASKS else {}
    skinner, identities = m.MhrMeshSkinner(), m.SubjectIdentities.load(m.SUBJECT_IDENTITY_PATH)
    server = viser.ViserServer(host="127.0.0.1", port=9480, verbose=False)
    server.scene.set_up_direction("+z")
    chrome = subprocess.Popen(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--headless=new",
                               "--use-angle=swiftshader", "--enable-unsafe-swiftshader", f"--window-size={SIZE},{SIZE}",
                               f"--user-data-dir={tempfile.mkdtemp()}", f"http://127.0.0.1:{server.get_port()}"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 60
        while not server.get_clients() and time.time() < deadline:
            time.sleep(0.5)
        client = next(iter(server.get_clients().values()))
        for e in examples:
            if m.NAMES and e["name"] not in m.NAMES:
                continue
            entry, root = (e, m.GALLERY) if e["mode"] == "motion_cond" else (tasks[f"{e['name']}__motion_cond"], m.TASKS)
            clip = entry["clip"]
            motion = torch.load(root / entry["cohort"] / "motion" / f"{clip}.pt", map_location="cpu",
                                weights_only=False)[entry["motion"]].double().numpy()
            motion = m.correct_head_rotation(motion, m.head_to_cpf_offset(clip))
            joints, _, _ = m.motion_v2_to_joints(motion)
            vertices = skinner.from_motion_v2(motion, identities.bind_state(m.subject_id_from_clip_name(clip)),
                                              positions_mode="fk", joints=joints).astype(np.float32)
            # the figure camera's horizontal heading, so the panel looks at the body from the scene's side
            camera = json.loads((m.FIGSRC / "cameras" / f"{e['name']}.json").read_text())
            heading = np.array(camera["look_at"][:2]) - np.array(camera["position"][:2])
            heading /= np.linalg.norm(heading)
            path = follow_path(vertices)
            server.scene.reset()
            server.scene.add_grid("/floor", width=24.0, height=24.0, plane="xy", cell_size=0.5, section_size=1.0,
                                  cell_color=(215, 215, 215), section_color=(185, 185, 185), shadow_opacity=0.25,
                                  position=(float(path[:, 0].mean()), float(path[:, 1].mean()), 0.0))
            body = server.scene.add_mesh_simple("/body", vertices=vertices[0], faces=skinner.faces,
                                                color=m.ramp(0.0), side="double")
            last = len(vertices) - 1
            with tempfile.TemporaryDirectory() as tmp:
                for k, t in enumerate(range(0, len(vertices), 3)):
                    body.vertices = vertices[t]
                    body.color = m.ramp(t / last)
                    target = path[t]
                    position = target + np.array([*(-DIST * heading), RISE])
                    wxyz = vtf.SO3.from_matrix(m.look_at_rotation(position, target)).wxyz
                    time.sleep(0.15 if k else 1.5)
                    rgba = Image.fromarray(client.get_render(SIZE, SIZE, wxyz=wxyz, position=position, fov=FOV,
                                                             transport_format="png")).convert("RGBA")
                    Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB").save(
                        Path(tmp) / f"{k:02d}.png")
                out = m.SITE / "media" / "thesis" / f"{e['name']}__input_motion.mp4"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "10", "-i", str(Path(tmp) / "%02d.png"),
                                "-vf", f"scale={OUT}:{OUT}:flags=lanczos", "-c:v", "libx264", "-crf", "26",
                                "-preset", "slow", "-g", "1", "-pix_fmt", "yuv420p",
                                "-movflags", "+faststart", "-an", str(out)], check=True)
            print(e["name"], f"{out.name} {out.stat().st_size / 1e3:.0f} kB", flush=True)
    finally:
        chrome.terminate()
        server.stop()


if __name__ == "__main__":
    main()
