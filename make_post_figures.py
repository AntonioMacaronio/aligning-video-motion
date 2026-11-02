"""Render the thesis figures that the project page shows, from the thesis repo's figures/*.pdf (macOS, Quartz).

    python3 make_post_figures.py <thesis repo> <site repo>

Writes <site>/figures/<name>.jpg for figures with renders or photos in them and .png for line drawings, at a width
of about twice the width the page shows them at.
"""
import sys
from pathlib import Path

import Quartz

THESIS, SITE = Path(sys.argv[1]), Path(sys.argv[2])
# name: (format, pixel width)
FIGURES = {"arch_tasks": ("jpg", 2000), "data_example_kitchen_walk": ("jpg", 2000), "arch_block": ("png", 1600),
           "arch_sidecar": ("png", 2000)}


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


(SITE / "figures").mkdir(exist_ok=True)
for name, (kind, width) in FIGURES.items():
    out = SITE / "figures" / f"{name}.{kind}"
    w, h = render(THESIS / "figures" / f"{name}.pdf", out, width, kind)
    print(f"{out.name}: {w} x {h}, {out.stat().st_size / 1e3:.0f} kB")
