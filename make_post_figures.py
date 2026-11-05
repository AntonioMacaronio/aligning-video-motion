"""Render the thesis figures that the project page shows, from the thesis repo's figures/*.pdf (macOS, Quartz).

    python3 make_post_figures.py <thesis repo> <site repo>

Writes <site>/figures/<name>.jpg for figures with renders or photos in them and .png for line drawings, at a width
of about twice the width the page shows them at; and figures/thumbs/<example>.jpg, the thesis figures viewer's
thumbnails: the 3D panel of each Section 4.4 figure (figures/gallery/<example>.pdf), cropped to a square around its bodies.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import Quartz
from PIL import Image

THESIS, SITE = Path(sys.argv[1]), Path(sys.argv[2])
# name: (format, pixel width)
FIGURES = {"arch_tasks": ("jpg", 2000), "data_example_kitchen_walk": ("jpg", 2000), "arch_block": ("png", 1600),
           "arch_sidecar": ("png", 2000)}
# Figures shown at the text column's width: their white page margins are trimmed, so the drawing meets the text's edges.
TRIM = {"arch_block"}


def render(pdf, out, width, kind):
    path = str(pdf).encode()
    page = Quartz.CGPDFDocumentGetPage(Quartz.CGPDFDocumentCreateWithURL(
        Quartz.CFURLCreateFromFileSystemRepresentation(None, path, len(path), False)), 1)
    box = Quartz.CGPDFPageGetBoxRect(page, Quartz.kCGPDFCropBox)
    scale = width / box.size.width
    w, h = int(round(box.size.width * scale)), int(round(box.size.height * scale))
    ctx = Quartz.CGBitmapContextCreate(None, w, h, 8, 0, Quartz.CGColorSpaceCreateDeviceRGB(),
                                       Quartz.kCGImageAlphaNoneSkipLast)
    Quartz.CGContextSetInterpolationQuality(ctx, Quartz.kCGInterpolationHigh)
    Quartz.CGContextSetRGBFillColor(ctx, 1, 1, 1, 1)
    Quartz.CGContextFillRect(ctx, Quartz.CGRectMake(0, 0, w, h))
    Quartz.CGContextScaleCTM(ctx, scale, scale)
    Quartz.CGContextTranslateCTM(ctx, -box.origin.x, -box.origin.y)
    Quartz.CGContextDrawPDFPage(ctx, page)
    target = str(out).encode()
    dest = Quartz.CGImageDestinationCreateWithURL(
        Quartz.CFURLCreateFromFileSystemRepresentation(None, target, len(target), False),
        "public.jpeg" if kind == "jpg" else "public.png", 1, None)
    options = {Quartz.kCGImageDestinationLossyCompressionQuality: 0.82} if kind == "jpg" else None
    Quartz.CGImageDestinationAddImage(dest, Quartz.CGBitmapContextCreateImage(ctx), options)
    assert Quartz.CGImageDestinationFinalize(dest), out
    return w, h


def thumbnail(im, size=176):
    """The figure's 3D panel (below the frame strip, without its label and the caption), as a square around the bodies."""
    rgb = np.asarray(im).astype(int)
    h, w, _ = rgb.shape
    ink = rgb.min(axis=2) < 200                                # bodies and points; not the light grid or the page
    top = int(0.37 * h)                                        # below the frames and their time labels
    ink[:top] = False
    ink[top:top + int(0.05 * h), :int(0.32 * w)] = False      # the panel label ("Generated motion", "Input motion ...")
    ink[int(0.925 * h):] = False                               # the caption (one or two lines)
    ys, xs = np.nonzero(ink)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    hsv = np.asarray(im.convert("HSV")).astype(int)            # hue 0-255
    hue, sat, val = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    body = ink & (hue > 130) & (hue < 300) & (sat > 0.4) & (val > 0.3)   # the green-to-purple bodies
    bx = np.nonzero(body)[1]
    cx = float(np.median(bx)) if len(bx) > 200 else (x0 + x1) / 2
    side = max(y1 - y0, min(x1 - x0, 1.2 * (y1 - y0))) * 1.04 # a wide panel is cut to a near-square around the bodies
    cx = min(max(cx, x0 + side / 2), x1 - side / 2) if x1 - x0 > side else (x0 + x1) / 2
    cy = (y0 + y1) / 2
    box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
    canvas = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), "white")
    inside = (max(box[0], x0), max(box[1], y0), min(box[2], x1), min(box[3], y1))  # never past the panel
    canvas.paste(im.crop(inside), (inside[0] - box[0], inside[1] - box[1]))
    return canvas.resize((size, size), Image.LANCZOS)


(SITE / "figures").mkdir(exist_ok=True)
for name, (kind, width) in FIGURES.items():
    out = SITE / "figures" / f"{name}.{kind}"
    w, h = render(THESIS / "figures" / f"{name}.pdf", out, width, kind)
    if name in TRIM:
        im = Image.open(out).convert("RGB")
        ys, xs = np.nonzero(np.asarray(im).min(axis=2) < 245)
        im = im.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
        im.save(out)
        w, h = im.size
    print(f"{out.name}: {w} x {h}, {out.stat().st_size / 1e3:.0f} kB")
(SITE / "figures" / "thumbs").mkdir(exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    for example in json.loads((SITE / "thesis_examples.json").read_text()):
        full = Path(tmp) / f"{example['name']}.png"
        render(THESIS / "figures" / "gallery" / f"{example['name']}.pdf", full, 1400, "png")
        out = SITE / "figures" / "thumbs" / f"{example['name']}.jpg"
        thumbnail(Image.open(full).convert("RGB")).save(out, quality=86)
        print(f"thumbs/{out.name}: {out.stat().st_size / 1e3:.0f} kB")
