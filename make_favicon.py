"""The tab icon: a body mesh seen through a fisheye lens, framed like an egocentric NymeriaPlus frame.

    galenv/python make_favicon.py <scene dir> <site dir> [--preview preview.png]

The body is one snapshot of a gallery scene (scene.npz from gallery_scene.py; the site's icon uses the fridge example's
joint sample, snapshot 3 = motion frame 30), in the figures' green-to-purple colour at that time, standing on the
figures' floor grid. viser renders it with a wide perspective camera in headless Chrome; the render is then remapped to
an equidistant fisheye (angle proportional to radius, 62 deg at the edge midpoints) and the corners outside radius 1.315
(in half-widths) are blacked out, the image circle of the dataset's undistorted 640 x 640 frames. Writes favicon.ico
(16, 32, 48 px), favicon-32.png and apple-touch-icon.png (180 px), and optionally a 512 px preview.
"""
import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, "/Users/antzhan/lab42/src/masters_thesis/figure-src/data-figure-20261023")
import viser  # noqa: E402
import viser.transforms as vtf  # noqa: E402
from render_datafig import ramp  # noqa: E402

RENDER = 2048         # perspective render, px
HALF_FOV = np.radians(74.0)
EDGE = np.radians(62.0)  # fisheye angle at the edge midpoints
CIRCLE = 1.315        # image-circle radius of the NymeriaPlus frames, in half-widths


def look_at(position, target):
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross(forward, [0, 0, 1.0]); right /= np.linalg.norm(right)
    return vtf.SO3.from_matrix(np.stack([right, np.cross(forward, right), forward], axis=1)).wxyz


def fisheye(perspective, size):
    """Equidistant fisheye from a square perspective render with half field of view HALF_FOV."""
    n = perspective.shape[0]
    f = (n / 2) / np.tan(HALF_FOV)
    yy, xx = np.mgrid[:size, :size]
    x, y = (xx + 0.5 - size / 2) / (size / 2), (yy + 0.5 - size / 2) / (size / 2)
    r = np.hypot(x, y)
    theta = np.minimum(r * EDGE, HALF_FOV * 1.414)
    scale = np.where(r > 0, f * np.tan(np.minimum(theta, np.radians(89))) / np.maximum(r, 1e-9), 0)
    u, v = n / 2 + x * scale - 0.5, n / 2 + y * scale - 0.5
    out = np.stack([ndimage.map_coordinates(perspective[..., c].astype(np.float32), [v, u], order=1, mode="nearest")
                    for c in range(3)], -1)
    out[r > CIRCLE] = 0
    return out.clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--snapshot", type=int, default=3)
    parser.add_argument("--distance", type=float, default=0.85, help="m from the body's vertical axis")
    parser.add_argument("--azimuth", type=float, default=-40.0, help="deg from the body's facing direction")
    parser.add_argument("--height", type=float, default=1.5, help="camera height, m")
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()

    scene = np.load(args.scene / "scene.npz")
    vertices, faces = scene["vertices"][args.snapshot].copy(), scene["faces"]
    vertices[:, 2] -= vertices[:, 2].min()                       # feet on the grid
    frame = int(scene["snapshots"][args.snapshot])
    center = np.array([*vertices[:, :2].mean(0), 0.0])
    facing = scene["head_rot"][frame][:, 2]
    yaw = np.arctan2(facing[1], facing[0]) + np.radians(args.azimuth)
    position = center + np.array([args.distance * np.cos(yaw), args.distance * np.sin(yaw), args.height])
    target = center + np.array([0, 0, 0.82])

    server = viser.ViserServer(host="127.0.0.1", port=9497, verbose=False)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/floor", width=12.0, height=12.0, plane="xy", cell_size=0.5, section_size=1.0,
                          cell_color=(200, 200, 200), section_color=(165, 165, 165), shadow_opacity=0.3,
                          position=(float(center[0]), float(center[1]), 0.0))
    server.scene.add_mesh_simple("/body", vertices=vertices, faces=faces, color=ramp(frame / 144), side="double")
    chrome = subprocess.Popen(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--headless=new",
                               "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--window-size=800,800",
                               f"--user-data-dir={tempfile.mkdtemp()}", f"http://127.0.0.1:{server.get_port()}"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 60
        while not server.get_clients() and time.time() < deadline:
            time.sleep(0.5)
        client = next(iter(server.get_clients().values()))
        time.sleep(2.0)
        rgba = Image.fromarray(client.get_render(RENDER, RENDER, wxyz=np.asarray(look_at(position, target)),
                                                 position=position, fov=2 * HALF_FOV, transport_format="png"))
    finally:
        chrome.terminate()
        server.stop()
    perspective = np.asarray(Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba.convert("RGBA")).convert("RGB"))
    args.out.mkdir(parents=True, exist_ok=True)
    big = Image.fromarray(fisheye(perspective, 1024))
    if args.preview:
        big.resize((512, 512), Image.LANCZOS).save(args.preview)
    big.resize((180, 180), Image.LANCZOS).save(args.out / "apple-touch-icon.png")
    big.resize((32, 32), Image.LANCZOS).save(args.out / "favicon-32.png")
    big.resize((48, 48), Image.LANCZOS).save(args.out / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
    print("wrote favicon.ico, favicon-32.png and apple-touch-icon.png to", args.out)


if __name__ == "__main__":
    main()
