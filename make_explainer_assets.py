"""The geometric post-training explainer's images (explainer/), from the staircase example's generated video.

    python3 make_explainer_assets.py [site dir]

- explainer/frames.jpg: 12 frames of media/thesis/joint_1.mp4 (the joint sample of the viewer's opening example), evenly
  spaced over its 49, side by side at 200 px. The explainer's sample shows frame 0 while it is denoised, then plays them.
- explainer/noise0.png ... noise2.png: 33 x 33 Gaussian RGB noise, one per rollout in the story; the page scales it up
  with crisp pixels, so it reads as a noisy latent rather than as film grain.
Needs ffmpeg, numpy and Pillow.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

FRAMES, SIZE, NOISE = 12, 200, 33

site = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
out = site / "explainer"
out.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(site / "media/thesis/joint_1.mp4"),
                    f"{tmp}/f%02d.png"], check=True)
    frames = sorted(Path(tmp).glob("f*.png"))
    picks = [frames[round(i * (len(frames) - 1) / (FRAMES - 1))] for i in range(FRAMES)]
    sprite = Image.new("RGB", (SIZE * FRAMES, SIZE))
    for i, p in enumerate(picks):
        sprite.paste(Image.open(p).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS), (i * SIZE, 0))
    sprite.save(out / "frames.jpg", quality=82, optimize=True, progressive=True)
for k in range(3):
    noise = np.random.default_rng(k).normal(127, 70, (NOISE, NOISE, 3)).clip(0, 255).astype(np.uint8)
    Image.fromarray(noise).save(out / f"noise{k}.png", optimize=True)
print("wrote", ", ".join(sorted(p.name for p in out.iterdir())), "to", out)
