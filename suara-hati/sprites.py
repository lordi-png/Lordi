"""Potong sprite dari lembar karakter berlatar solid (putih / pola kotak-kotak).

Latar dianggap = piksel terang (min RGB >= bright) dan nyaris abu-abu (saturasi <= sat)
yang terhubung ke tepi potongan; celah tertutup yang besar (mis. di antara lengan dan
badan) juga dibuang. Hasil berupa RGBA dengan tepi alpha yang dihaluskan.
"""
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage


def cutout(im, box=None, bright=185, sat=8, hole=300, keep_frac=0.12, erode=1):
    if box:
        im = im.crop(box)
    a = np.array(im.convert("RGB")).astype(np.int16)
    bgl = (a.min(2) >= bright) & ((a.max(2) - a.min(2)) <= sat)
    lab, k = ndimage.label(bgl)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    sizes = ndimage.sum(bgl, lab, range(1, k + 1)) if k else []
    bg = np.zeros_like(bgl)
    for i in range(1, k + 1):
        if i in border or sizes[i - 1] >= hole:
            bg |= lab == i
    fg = ~bg
    fg = ndimage.binary_opening(fg, iterations=1)
    lab, k = ndimage.label(fg)
    if k:
        sizes = ndimage.sum(fg, lab, range(1, k + 1))
        big = sizes.max()
        keep = [i + 1 for i, s_ in enumerate(sizes) if s_ >= big * keep_frac]
        fg = np.isin(lab, keep)
    fg = ndimage.binary_erosion(fg, iterations=erode)
    alpha = Image.fromarray((fg * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
    out = Image.fromarray(a.astype(np.uint8)).convert("RGBA")
    out.putalpha(alpha)
    bb = out.getbbox()
    return out.crop(bb) if bb else out


def split_strip(im, box, min_gap=4, min_w=25, **kw):
    """Split a horizontal strip of figures at empty columns; returns list of cutouts."""
    full = cutout(im, box, keep_frac=0.0, **kw)
    # recompute on the uncropped strip to keep x positions
    x0, y0, x1, y1 = box
    a = np.array(im.crop(box).convert("RGB")).astype(np.int16)
    bgl = (a.min(2) >= kw.get("bright", 185)) & ((a.max(2) - a.min(2)) <= kw.get("sat", 8))
    col = (~bgl).sum(0) > 2
    runs, s = [], None
    for i, v in enumerate(col):
        if v and s is None:
            s = i
        if not v and s is not None:
            runs.append([s, i])
            s = None
    if s is not None:
        runs.append([s, len(col)])
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < min_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    del full
    return [cutout(im, (x0 + r[0] - 2, y0, x0 + r[1] + 2, y1), **kw) for r in merged if r[1] - r[0] >= min_w]


def split_by_body(im, box, thr=40, **kw):
    """Split a strip of overlapping figures (feet may touch) midway between torso columns."""
    x0, y0, x1, y1 = box
    a = np.array(im.crop(box).convert("RGB")).astype(np.int16)
    bgl = (a.min(2) >= kw.get("bright", 185)) & ((a.max(2) - a.min(2)) <= kw.get("sat", 8))
    col = (~bgl).sum(0) > thr
    runs, s = [], None
    for i, v in enumerate(col):
        if v and s is None:
            s = i
        if not v and s is not None:
            runs.append((s, i))
            s = None
    if s is not None:
        runs.append((s, len(col)))
    cs = [(r[0] + r[1]) / 2 for r in runs if r[1] - r[0] > 6]
    cuts = [0] + [int((cs[i] + cs[i + 1]) / 2) for i in range(len(cs) - 1)] + [x1 - x0]
    return [cutout(im, (x0 + cuts[i], y0, x0 + cuts[i + 1], y1), **kw) for i in range(len(cuts) - 1)]
