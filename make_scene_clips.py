"""Square video clips of the thesis figures' 3D scenes, for the main page's three side-by-side tasks.

    PYTHONPATH=<LVP tree with datasets/head_correction.py> python make_scene_clips.py \
        <gallery dir> <figure-src gallery dir> <site dir> [--chrome-flag ...] [name ...]

The arguments are make_thesis_scenes.py's, without --tasks: a clip shows the example's figure task. The scene is the
page's scene for the example, built by make_thesis_scenes.py's own functions: the figure's cut point cloud, the floor
and the body at all 145 motion frames, colored along the time ramp. viser renders it in headless Chrome from a square
version of the page's opening camera (make_thesis_scenes.site_camera at aspect 1: the figure camera's position, turned
to the content and zoomed until it fills the square). Writes <site>/media/scenes/<name>.mp4, 145 frames at 30 fps,
every frame a keyframe so that the page can seek it to any frame (motion frame t at frame t), and its poster.
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import viser
import viser.transforms as vtf
from PIL import Image

CHROME_FLAGS = [a for a in sys.argv[4:] if a.startswith("--") and a not in ("--videos-only", "--cameras-only")]
sys.argv = sys.argv[:4] + [a for a in sys.argv[4:] if a not in CHROME_FLAGS]
import make_thesis_scenes as m  # noqa: E402  parses the same command line

SIDE = 420                   # px: rendered and encoded (the page shows it at about 210 px)
FPS = 30


def main():
    examples = json.loads((m.FIGSRC / "examples.json").read_text())
    skinner, identities = m.MhrMeshSkinner(), m.SubjectIdentities.load(m.SUBJECT_IDENTITY_PATH)
    out_dir = m.SITE / "media" / "scenes"
    out_dir.mkdir(parents=True, exist_ok=True)
    server = viser.ViserServer(host="127.0.0.1", port=9491, verbose=False)
    server.scene.set_up_direction("+z")
    chrome = subprocess.Popen(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--headless=new",
                               *(CHROME_FLAGS or ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]),
                               f"--window-size={SIDE},{SIDE}", f"--user-data-dir={tempfile.mkdtemp()}",
                               f"http://127.0.0.1:{server.get_port()}"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 60
        while not server.get_clients() and time.time() < deadline:
            time.sleep(0.5)
        client = next(iter(server.get_clients().values()))
        for e in examples:
            name = e["name"]
            if m.NAMES and name not in m.NAMES:
                continue
            started = time.time()
            scene, points, colors, middle, args, view, _ = m.figure_scene(name, name, m.GALLERY)
            position, look_at, fov = m.site_camera(points, scene["vertices"], dict(view, aspect=1.0))
            points, colors = m.thin(points, colors)
            vertices, error = m.posed_body(m.GALLERY, e, scene, skinner, identities)
            assert error < 1e-3, f"{name}: posed body differs from the figure's snapshots by {error:.4f} m"
            wxyz = vtf.SO3.from_matrix(m.look_at_rotation(position, look_at)).wxyz
            server.scene.reset()
            server.scene.add_point_cloud("/scene", points=points.astype(np.float32), colors=colors,
                                         point_size=args.point_size, point_shape="circle", precision="float16")
            if not args.no_floor:
                server.scene.add_grid("/floor", width=9.0, height=6.0, plane="xy", cell_size=0.5, section_size=1.0,
                                      cell_color=(215, 215, 215), section_color=(185, 185, 185), shadow_opacity=0.25,
                                      position=(float(middle[0]), float(middle[1]), 0.0))
            body = server.scene.add_mesh_simple("/body", vertices=vertices[0], faces=skinner.faces, color=m.ramp(0.0),
                                                side="double")
            last = len(vertices) - 1
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                for t in range(len(vertices)):
                    body.vertices = vertices[t]
                    body.color = m.ramp(t / last)
                    time.sleep(0.1 if t else 2.0)
                    rgba = Image.fromarray(client.get_render(SIDE, SIDE, wxyz=wxyz, position=position, fov=fov,
                                                             transport_format="png")).convert("RGBA")
                    Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB").save(
                        tmp / f"{t:03d}.png")
                out = out_dir / f"{name}.mp4"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(tmp / "%03d.png"),
                                "-c:v", "libx264", "-crf", "24", "-preset", "slow", "-g", "1", "-pix_fmt", "yuv420p",
                                "-movflags", "+faststart", "-an", str(out)], check=True)
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp / "000.png"), "-q:v", "3",
                                str(out.with_suffix(".jpg"))], check=True)
            print(name, f"{out.stat().st_size / 1e3:.0f} kB, fov {np.degrees(fov):.1f} deg, {time.time() - started:.0f} s",
                  flush=True)
    finally:
        chrome.terminate()
        server.stop()


if __name__ == "__main__":
    main()
