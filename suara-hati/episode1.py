"""SUARA HATI — Episode 1: Mereka yang Selalu Kita Abaikan.

Animasi cut-out bergerak, 720x1280 (9:16) @30fps, durasi ~4 menit.

Aset yang dipakai (folder assets/):
  arka_sheet2.png  — lembar karakter Arka (pose & ekspresi)
  arka_sheet.png   — lembar ekspresi Arka lama (untuk ekspresi tersenyum)
  arka_walk.gif    — animasi Arka berjalan (8 frame)
Anak kecil, ibu warung, pejalan kaki, mobil, dan latar kota digambar prosedural.

Pemakaian:
  python3 episode1.py out.mp4                # render penuh
  python3 episode1.py out 12.5 40 150        # simpan still frame (out_12.5.png, …)
"""
import math
import os
import random
import subprocess
import sys
import wave
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageSequence
from scipy import ndimage

W, H, FPS = 720, 1280, 30
HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "assets")

FD = "/usr/share/fonts/truetype/crosextra/"
F_TITLE = ImageFont.truetype(FD + "Caladea-Bold.ttf", 86)
F_EP = ImageFont.truetype(FD + "Caladea-Italic.ttf", 30)
F_CHAP_S = ImageFont.truetype(FD + "Caladea-Bold.ttf", 26)
F_CHAP_L = ImageFont.truetype(FD + "Caladea-Bold.ttf", 44)
F_TEXT = ImageFont.truetype(FD + "Carlito-Regular.ttf", 33)
F_NARR = ImageFont.truetype(FD + "Caladea-Italic.ttf", 33)
F_NAME = ImageFont.truetype(FD + "Carlito-Bold.ttf", 24)
F_BUB = ImageFont.truetype(FD + "Carlito-Bold.ttf", 30)
F_SIGN = ImageFont.truetype(FD + "Carlito-Bold.ttf", 22)
F_SIGN_L = ImageFont.truetype(FD + "Carlito-Bold.ttf", 34)
F_QUOTE = ImageFont.truetype(FD + "Caladea-Italic.ttf", 38)

SUB_BOTTOM = H - 290          # TikTok safe area (bottom UI)
NARR_TOP = 150                # below TikTok top bar


# ======================================================================
# helpers
# ======================================================================
def ease(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def lerp(a, b, x):
    return a + (b - a) * x


def fade(t, a, b, f=0.35):
    if t < a or t > b:
        return 0.0
    return ease((t - a) / f) * ease((b - t) / f)


def wrap(text, font, maxw):
    out = []
    for para in text.split("\n"):
        cur = ""
        for w_ in para.split():
            test = (cur + " " + w_).strip()
            if font.getlength(test) <= maxw:
                cur = test
            else:
                out.append(cur)
                cur = w_
        out.append(cur)
    return out


def set_alpha(im, k):
    if k >= 0.999:
        return im
    im = im.copy()
    a = np.array(im.getchannel("A"), np.float32) * k
    im.putalpha(Image.fromarray(a.astype(np.uint8)))
    return im


def scaled(im, s):
    if abs(s - 1) < 1e-3:
        return im
    return im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BILINEAR)


def put(img, sp, x, y, anchor="bottom", alpha=1.0, scale=1.0, flip=False, rot=0.0):
    """Composite sprite; anchor bottom-center (feet) or center."""
    if alpha <= 0.003:
        return
    s = scaled(sp, scale)
    if flip:
        s = s.transpose(Image.FLIP_LEFT_RIGHT)
    if rot:
        s = s.rotate(rot, Image.BICUBIC, expand=True)
    s = set_alpha(s, alpha)
    if anchor == "bottom":
        px, py = int(x - s.width / 2), int(y - s.height)
    else:
        px, py = int(x - s.width / 2), int(y - s.height / 2)
    if px >= W or py >= H or px + s.width <= 0 or py + s.height <= 0:
        return
    # manual clip (alpha_composite does not accept negative dest)
    sx, sy = max(0, -px), max(0, -py)
    dx, dy = max(0, px), max(0, py)
    w_ = min(s.width - sx, W - dx)
    h_ = min(s.height - sy, H - dy)
    if w_ <= 0 or h_ <= 0:
        return
    img.alpha_composite(s.crop((sx, sy, sx + w_, sy + h_)), (dx, dy))


def shadow_ellipse(img, x, y, w_, alpha=90):
    ov = Image.new("RGBA", (int(w_ * 1.4) + 20, int(w_ * 0.35) + 20))
    ImageDraw.Draw(ov).ellipse((10, 10, ov.width - 10, ov.height - 10), fill=(0, 0, 0, alpha))
    ov = ov.filter(ImageFilter.GaussianBlur(6))
    put(img, ov, x, y + ov.height / 2, "bottom")


def text_layer(lines, font, color, cy, alpha, cx=W // 2, spacing=10, shadow=True, stroke=0):
    ov = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(ov)
    lh = font.size + spacing
    y = cy - lh * len(lines) / 2
    for ln in lines:
        x = cx - font.getlength(ln) / 2
        if shadow:
            d.text((x + 2, y + 3), ln, font=font, fill=(0, 0, 0, int(190 * alpha)))
        d.text((x, y), ln, font=font, fill=tuple(color) + (int(255 * alpha),),
               stroke_width=stroke, stroke_fill=(0, 0, 0, int(160 * alpha)))
        y += lh
    return ov


def heart(d, cx, cy, r, fill):
    d.ellipse((cx - r, cy - r * 0.8, cx, cy + r * 0.2), fill=fill)
    d.ellipse((cx, cy - r * 0.8, cx + r, cy + r * 0.2), fill=fill)
    d.polygon([(cx - r * 0.97, cy - 0.1 * r), (cx + r * 0.97, cy - 0.1 * r), (cx, cy + r)], fill=fill)


# ======================================================================
# Arka sprites
# ======================================================================
def clean(im):
    a = np.array(im)
    m = a[:, :, 3] > 60
    lab, k = ndimage.label(ndimage.binary_dilation(m, iterations=2))
    if k:
        sizes = ndimage.sum(m, lab, range(1, k + 1))
        a[:, :, 3] = np.where(lab == (np.argmax(sizes) + 1), a[:, :, 3], 0)
    out = Image.fromarray(a)
    return out.crop(out.getbbox())


SHEET2 = Image.open(os.path.join(A, "arka_sheet2.png")).convert("RGBA")
SHEET1 = Image.open(os.path.join(A, "arka_sheet.png")).convert("RGBA")
BOX = {"front": (10, 0, 215, 708), "side": (216, 0, 335, 708), "back": (330, 0, 527, 708),
       "sit": (525, 760, 692, 1020), "sitdown": (1068, 780, 1280, 1022)}
ARKA = {k: clean(SHEET2.crop(b)) for k, b in BOX.items()}
for k in ARKA:  # normalise standing heights to 700 px
    if k in ("front", "side", "back"):
        s = 700 / ARKA[k].height
        ARKA[k] = ARKA[k].resize((int(ARKA[k].width * s), 700), Image.LANCZOS)
_a = np.array(ARKA["side"])
_a[560:, :16, 3] = 0          # drop a neighbouring shoe fragment from the sheet
ARKA["side"] = Image.fromarray(_a)
ARKA["side"] = ARKA["side"].crop(ARKA["side"].getbbox())

GIF = []
_g = Image.open(os.path.join(A, "arka_walk.gif"))
for f in ImageSequence.Iterator(_g):
    GIF.append(f.convert("RGBA").crop((150, 0, 358, 512)))


def gif_frame(t, fps=7.0):
    return GIF[int(t * fps) % len(GIF)]


def face_tile(sheet, r, c):
    if sheet == 2:
        t = SHEET2.crop((528 + c * 145, r * 172, 528 + (c + 1) * 145, r * 172 + 168))
    else:
        x0, x1, h0 = 370, 1212, 1296
        tw, th = (x1 - x0) / 5, h0 / 6
        t = SHEET1.crop((int(x0 + c * tw + 9), int(r * th + 9),
                         int(x0 + (c + 1) * tw - 9), int((r + 1) * th - 15)))
    return clean(t)


def rounded_mask(size, radius, feather):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle((feather, feather, size[0] - feather, size[1] - feather),
                                        radius, fill=255)
    return m.filter(ImageFilter.GaussianBlur(feather / 2))


def card(tile, size=(400, 460), panel=(200, 210, 230)):
    s = max(size[0] / tile.width, size[1] / tile.height) * 1.02
    t = tile.resize((int(tile.width * s), int(tile.height * s)), Image.LANCZOS)
    t = t.filter(ImageFilter.UnsharpMask(2, 60, 2))
    c = Image.new("RGBA", size, panel + (240,))
    g = Image.new("L", size, 0)
    ImageDraw.Draw(g).ellipse((-80, -60, size[0] + 80, size[1] + 60), fill=110)
    c = Image.composite(Image.new("RGBA", size, (250, 250, 255, 240)), c, g.filter(ImageFilter.GaussianBlur(50)))
    c.alpha_composite(t, ((size[0] - t.width) // 2, size[1] - t.height))
    m = np.minimum(np.array(c.getchannel("A")), np.array(rounded_mask(size, 30, 6)))
    c.putalpha(Image.fromarray(m))
    out = Image.new("RGBA", (size[0] + 40, size[1] + 40))
    sh = Image.new("L", out.size, 0)
    ImageDraw.Draw(sh).rounded_rectangle((22, 28, size[0] + 22, size[1] + 28), 30, fill=160)
    out.putalpha(sh.filter(ImageFilter.GaussianBlur(12)))
    out.alpha_composite(c, (20, 20))
    ImageDraw.Draw(out).rounded_rectangle((25, 25, size[0] + 15, size[1] + 15), 28,
                                          outline=(255, 255, 255, 150), width=2)
    return out


FACE = {
    "neutral": card(face_tile(2, 0, 0)),
    "notice": card(face_tile(2, 0, 1)),
    "skeptic": card(face_tile(2, 2, 0)),
    "doubt": card(face_tile(2, 0, 5)),
    "frown": card(face_tile(2, 1, 5)),
    "conflict": card(face_tile(2, 3, 1)),
    "worry": card(face_tile(2, 1, 0)),
    "down": card(face_tile(2, 3, 5)),
    "regret": card(face_tile(2, 3, 4)),
    "resolve": card(face_tile(2, 1, 1)),
    "soft": card(face_tile(1, 5, 1), panel=(240, 220, 200)),
    "smile": card(face_tile(1, 2, 2), panel=(245, 220, 190)),
    "calm": card(face_tile(1, 3, 3), panel=(245, 220, 190)),
}


# side walk: scissor the legs for a stride
def _side_parts():
    s = ARKA["side"]
    hip = int(s.height * 0.47)
    top = s.crop((0, 0, s.width, hip + 14))
    legs = s.crop((0, hip, s.width, s.height))
    return s, top, legs, hip


_SIDE = _side_parts()
_side_cache = {}


def side_walk(phase):
    """phase in [0,1) — one full stride cycle. Returns sprite (faces left)."""
    q = int(phase * 16) % 16
    if q in _side_cache:
        return _side_cache[q]
    s, top, legs, hip = _SIDE
    ang = 13 * math.sin(q / 16 * 2 * math.pi)
    pad = 60
    canvas = Image.new("RGBA", (s.width + pad * 2, s.height + 20))
    hx = s.width // 2 + pad
    for a, dark in ((-ang, 0.7), (ang, 1.0)):
        L = legs.copy()
        if dark < 1:
            arr = np.array(L).astype(np.float32)
            arr[:, :, :3] *= dark
            L = Image.fromarray(arr.astype(np.uint8))
        big = Image.new("RGBA", (canvas.width, canvas.height))
        big.alpha_composite(L, (pad, hip))
        big = big.rotate(a, Image.BICUBIC, center=(hx, hip + 10))
        canvas.alpha_composite(big)
    canvas.alpha_composite(top, (pad, 0))
    canvas = canvas.crop(canvas.getbbox())
    _side_cache[q] = canvas
    return canvas


def walk_bob(phase):
    return -abs(math.sin(phase * 2 * math.pi)) * 7


# ======================================================================
# procedural child
# ======================================================================
SKIN = (198, 142, 102)
SKIN_D = (170, 116, 84)
HAIR = (34, 27, 30)
SHIRT = (199, 162, 98)
SHIRT_D = (160, 128, 78)
PATCH = (118, 138, 150)
SHORTS = (92, 72, 60)
SANDAL = (60, 62, 74)
IRIS = (66, 40, 28)
LINE = (70, 45, 38)


def draw_head(expr="neutral", blink=False, look_up=0.0, k=3, hijab=False):
    """Head on 200x210 unit canvas; neck point = (100, 200)."""
    im = Image.new("RGBA", (200 * k, 215 * k))
    d = ImageDraw.Draw(im)

    def P(*pts):
        return [v * k for v in pts]

    if hijab:
        d.ellipse(P(14, 6, 186, 196), fill=(48, 98, 88))
        d.polygon(P(20, 120, 100, 214, 180, 120), fill=(48, 98, 88))
    else:
        d.ellipse(P(22, 10, 178, 172), fill=HAIR)
        d.ellipse(P(20, 96, 42, 132), fill=SKIN)
        d.ellipse(P(158, 96, 180, 132), fill=SKIN)
    d.ellipse(P(32, 32, 168, 186), fill=SKIN)
    d.ellipse(P(58, 128, 142, 200), fill=SKIN)
    if hijab:
        d.arc(P(26, 18, 174, 150), 190, 350, fill=(38, 82, 74), width=10 * k)
    else:
        d.polygon(P(24, 98, 28, 44, 58, 16, 100, 8, 142, 16, 172, 44, 176, 98, 162, 70, 152, 90,
                    136, 60, 121, 88, 103, 56, 86, 88, 70, 58, 55, 90, 40, 66), fill=HAIR)
    # blush & dirt smudge
    ov = Image.new("RGBA", im.size)
    od = ImageDraw.Draw(ov)
    od.ellipse(P(44, 136, 70, 152), fill=(235, 120, 110, 80))
    od.ellipse(P(130, 136, 156, 152), fill=(235, 120, 110, 80))
    if not hijab:
        od.ellipse(P(134, 152, 150, 162), fill=(110, 80, 60, 70))
    im.alpha_composite(ov.filter(ImageFilter.GaussianBlur(2 * k)))
    d = ImageDraw.Draw(im)
    ly = -5 * look_up
    for cx in (70, 130):
        inner = 1 if cx == 70 else -1
        # brows
        if expr in ("sad", "tired", "hope"):
            d.line(P(cx - 14 * inner, 94, cx + 12 * inner, 88), fill=LINE, width=3 * k)
        elif expr == "surprised":
            d.arc(P(cx - 15, 80, cx + 15, 96), 200, 340, fill=LINE, width=3 * k)
        else:
            d.arc(P(cx - 15, 86, cx + 15, 100), 200, 340, fill=LINE, width=3 * k)
        if blink or expr in ("happy", "kind"):
            d.arc(P(cx - 15, 108, cx + 15, 128), 200, 340, fill=LINE, width=4 * k)
            continue
        ry = 20 if expr == "surprised" else 18
        d.ellipse(P(cx - 16, 118 - ry, cx + 16, 118 + ry), fill=(250, 246, 240))
        d.ellipse(P(cx - 12, 102 + ly, cx + 12, 134 + ly), fill=IRIS)
        d.ellipse(P(cx - 6, 110 + ly, cx + 6, 126 + ly), fill=(25, 15, 12))
        d.ellipse(P(cx - 8, 104 + ly, cx - 1, 111 + ly), fill=(255, 255, 255))
        d.ellipse(P(cx + 3, 124 + ly, cx + 7, 128 + ly), fill=(255, 255, 255, 200))
        if expr == "tired":
            d.rectangle(P(cx - 18, 96, cx + 18, 113), fill=SKIN)
            d.line(P(cx - 17, 113, cx + 17, 113), fill=LINE, width=4 * k)
        else:
            d.arc(P(cx - 18, 99, cx + 18, 137), 195, 345, fill=(40, 25, 22), width=4 * k)
    d.line(P(101, 136, 98, 146), fill=SKIN_D, width=2 * k)
    m = {"neutral": "line", "hope": "o", "sad": "down", "tired": "down", "surprised": "O",
         "happy": "smile", "kind": "smile", "smile": "smile"}.get(expr, "line")
    if m == "line":
        d.line(P(92, 166, 108, 166), fill=LINE, width=3 * k)
    elif m == "o":
        d.ellipse(P(95, 160, 105, 170), fill=(150, 60, 60))
    elif m == "O":
        d.ellipse(P(92, 158, 108, 176), fill=(140, 50, 50))
    elif m == "down":
        d.arc(P(88, 164, 112, 180), 200, 340, fill=LINE, width=3 * k)
    elif m == "smile":
        d.chord(P(86, 150, 114, 176), 0, 180, fill=(160, 60, 60))
        d.arc(P(86, 150, 114, 176), 0, 180, fill=LINE, width=2 * k)
    if expr == "smile":   # tears of joy
        d.ellipse(P(52, 132, 60, 146), fill=(170, 210, 255))
        d.ellipse(P(140, 132, 148, 146), fill=(170, 210, 255))
    return im


def draw_body(pose="stand", k=3):
    """Body on 260x440 unit canvas; neck point = (130, 150). Ground y≈430."""
    im = Image.new("RGBA", (260 * k, 440 * k))
    d = ImageDraw.Draw(im)

    def P(*pts):
        return [v * k for v in pts]

    def limb(pts, w_, col=SKIN):
        d.line(P(*pts), fill=col, width=int(w_ * k), joint="curve")
        for i in range(0, len(pts), 2):
            r = w_ / 2
            d.ellipse(P(pts[i] - r, pts[i + 1] - r, pts[i] + r, pts[i + 1] + r), fill=col)

    if pose in ("stand", "ask", "hold"):
        limb([108, 330, 108, 418], 17)
        limb([152, 330, 152, 418], 17)
        d.ellipse(P(94, 414, 124, 428), fill=SANDAL)
        d.ellipse(P(138, 414, 168, 428), fill=SANDAL)
        d.polygon(P(94, 276, 166, 276, 170, 346, 134, 348, 130, 322, 126, 348, 90, 346), fill=SHORTS)
        d.rectangle(P(120, 138, 140, 162), fill=SKIN_D)
        shirt = [78, 166, 104, 152, 156, 152, 182, 166, 200, 214, 178, 224, 172, 200, 176, 300,
                 84, 300, 88, 200, 82, 224, 60, 214]
        d.polygon(P(*shirt), fill=SHIRT)
        d.polygon(P(84, 300, 92, 290, 100, 302, 110, 292, 118, 302), fill=SHIRT)
        d.polygon(P(150, 210, 172, 200, 176, 300, 150, 300), fill=SHIRT_D)
        d.rectangle(P(138, 246, 160, 266), fill=PATCH)
        d.arc(P(110, 146, 150, 170), 0, 180, fill=SHIRT_D, width=3 * k)
        if pose == "stand":
            limb([72, 222, 70, 296], 15)
            limb([188, 222, 190, 296], 15)
        elif pose == "ask":
            limb([72, 222, 70, 296], 15)
            limb([188, 220, 214, 262, 236, 258], 15)
            d.chord(P(222, 246, 254, 270), 0, 180, fill=SKIN)
        elif pose == "hold":
            limb([72, 222, 92, 262, 116, 268], 15)
            limb([188, 222, 168, 262, 144, 268], 15)
    else:  # sitting, seen from the front
        d.rectangle(P(120, 138, 140, 162), fill=SKIN_D)
        d.polygon(P(76, 168, 104, 154, 156, 154, 184, 168, 192, 380, 68, 380), fill=SHIRT)
        d.polygon(P(150, 200, 184, 180, 192, 380, 150, 380), fill=SHIRT_D)
        d.ellipse(P(52, 350, 208, 424), fill=SHORTS)
        if pose in ("sit", "sit_tired"):
            for kx in (104, 156):
                limb([kx, 300, kx, 404], 30)
                d.ellipse(P(kx - 20, 398, kx + 20, 420), fill=SANDAL)
            limb([70, 210, 76, 322, 124, 330], 16)
            limb([190, 210, 184, 322, 136, 330], 16)
        else:  # sit_hold — holding food in front
            for kx, fx in ((96, 86), (164, 174)):
                limb([kx, 318, fx, 404], 30)
                d.ellipse(P(fx - 20, 398, fx + 20, 420), fill=SANDAL)
            limb([70, 210, 82, 280, 114, 292], 16)
            limb([190, 210, 178, 280, 146, 292], 16)
    return im


_child_cache = {}


def child(pose="sit", expr="neutral", t=0.0, look_up=0.0, k=1.0, tilt=0.0):
    """Full child sprite (bottom = feet). k: final scale (1 ≈ 440 px tall standing)."""
    blink = (t % 3.7) < 0.12
    key = (pose, expr, blink, round(look_up, 1), round(k, 2), round(tilt, 0))
    if key in _child_cache:
        return _child_cache[key]
    rk = 2
    body = draw_body(pose, rk)
    head = draw_head(expr, blink, look_up, rk).resize((int(200 * rk * 0.66), int(215 * rk * 0.66)),
                                                     Image.LANCZOS)
    sitting = pose.startswith("sit")
    nx, ny = 130 * rk, 150 * rk
    canvas = Image.new("RGBA", (body.width, body.height + int(140 * rk)))
    oy = int(140 * rk)
    canvas.alpha_composite(body, (0, oy))
    h = head.rotate(tilt + (-8 if pose == "sit_tired" else 0), Image.BICUBIC, expand=False,
                    center=(head.width / 2, head.height * 0.93))
    canvas.alpha_composite(h, (int(nx - h.width / 2), int(ny + oy - h.height * 0.93 + (18 if sitting else 0))))
    canvas = canvas.crop(canvas.getbbox())
    out = canvas.resize((int(canvas.width * k / rk * 0.85), int(canvas.height * k / rk * 0.85)),
                        Image.LANCZOS)
    if len(_child_cache) > 400:
        _child_cache.clear()
    _child_cache[key] = out
    return out


def child_closeup(expr, t, look_up=0.0, pose="stand"):
    """Head-and-shoulders crop of a large child render."""
    sp = child(pose, expr, t, look_up, k=2.6)
    c = sp.crop((0, 0, sp.width, int(sp.height * 0.6)))
    s_ = 820 / c.height
    return c.resize((int(c.width * s_), 820), Image.BILINEAR)


def ibu(expr="neutral", t=0.0):
    blink = (t % 4.1) < 0.12
    k = 2
    head = draw_head("kind" if expr == "smile" else "neutral", blink, 0, k, hijab=True)
    im = Image.new("RGBA", (300 * k, 420 * k))
    d = ImageDraw.Draw(im)
    d.polygon([(40 * k, 420 * k), (60 * k, 250 * k), (110 * k, 215 * k), (190 * k, 215 * k),
               (240 * k, 250 * k), (260 * k, 420 * k)], fill=(160, 70, 70))
    d.polygon([(95 * k, 260 * k), (205 * k, 260 * k), (215 * k, 420 * k), (85 * k, 420 * k)], fill=(235, 225, 205))
    d.polygon([(90 * k, 205 * k), (150 * k, 250 * k), (210 * k, 205 * k)], fill=(48, 98, 88))
    im.alpha_composite(head, (int(150 * k - head.width / 2), int(20 * k)))
    return im.resize((300, 420), Image.LANCZOS)


# ======================================================================
# props
# ======================================================================
def nasi_bungkus(k=1.0):
    w_, h_ = int(120 * k), int(80 * k)
    im = Image.new("RGBA", (w_ + 10, h_ + 10))
    d = ImageDraw.Draw(im)
    d.polygon([(5, h_ * 0.45), (w_ * 0.5, 5), (w_, h_ * 0.45), (w_ * 0.5, h_)], fill=(196, 150, 100))
    d.polygon([(5, h_ * 0.45), (w_ * 0.5, h_ * 0.3), (w_, h_ * 0.45), (w_ * 0.5, h_)], fill=(180, 135, 88))
    d.line([(w_ * 0.5, 5), (w_ * 0.5, h_)], fill=(140, 100, 60), width=max(1, int(2 * k)))
    d.line([(w_ * 0.2, h_ * 0.62), (w_ * 0.8, h_ * 0.28)], fill=(210, 60, 60), width=max(2, int(4 * k)))
    return im


BUNGKUS = nasi_bungkus(1.0)


def phone_wallet_layer(alpha, t):
    ov = Image.new("RGBA", (W, H))
    if alpha <= 0:
        return ov
    d = ImageDraw.Draw(ov)
    a = int(255 * alpha)
    pop = 1 + 0.06 * math.sin(t * 6)
    # phone
    cx, cy = 230, 390
    d.rounded_rectangle((cx - 45 * pop, cy - 80 * pop, cx + 45 * pop, cy + 80 * pop), 14, fill=(30, 30, 40, a))
    d.rounded_rectangle((cx - 38, cy - 68, cx + 38, cy + 64), 8, fill=(110, 170, 230, a))
    # wallet
    cx2 = 470
    d.rounded_rectangle((cx2 - 80, cy - 50, cx2 + 80, cy + 55), 14, fill=(110, 70, 45, a))
    d.rounded_rectangle((cx2 - 10, cy - 20, cx2 + 80, cy + 20), 10, fill=(140, 92, 60, a))
    d.ellipse((cx2 + 50, cy - 7, cx2 + 64, cy + 7), fill=(230, 200, 120, a))
    # no cash chip
    d.rounded_rectangle((300, 500, 620, 560), 30, fill=(160, 40, 50, int(220 * alpha)))
    txt = "Tidak ada uang tunai"
    d.text((460 - F_NAME.getlength(txt) / 2, 516), txt, font=F_NAME, fill=(255, 255, 255, a))
    return ov


# ======================================================================
# backgrounds
# ======================================================================
WW = 1800
THEMES = {
    "dusk":    dict(sky=((70, 92, 140), (228, 160, 130)), win=(255, 210, 140), lamp=0.6, light=(1.0, 0.95, 0.95)),
    "grey":    dict(sky=((148, 158, 172), (205, 205, 210)), win=None, lamp=0.0, light=(0.95, 0.97, 1.0)),
    "rain":    dict(sky=((54, 62, 80), (108, 114, 128)), win=(230, 200, 140), lamp=0.5, light=(0.75, 0.8, 0.9)),
    "evening": dict(sky=((38, 40, 80), (176, 104, 112)), win=(255, 200, 120), lamp=1.0, light=(0.75, 0.72, 0.85)),
    "sunset":  dict(sky=((92, 70, 132), (255, 152, 92)), win=(255, 210, 150), lamp=0.7, light=(1.05, 0.9, 0.85)),
    "golden":  dict(sky=((255, 186, 120), (255, 228, 172)), win=(255, 230, 170), lamp=0.3, light=(1.1, 1.0, 0.88)),
}
SHOPS = ["TOKO", "PULSA", "LAUNDRY", "KAFE", "BENGKEL", "FOTOKOPI", "WARUNG MAKAN", "APOTEK"]


def grad(top, bottom, h, w):
    t = np.linspace(0, 1, h)[:, None, None]
    a = np.array(top, np.float32)[None, None]
    b = np.array(bottom, np.float32)[None, None]
    return np.repeat(a * (1 - t) + b * t, w, axis=1)


def build_street(theme):
    th = THEMES[theme]
    rnd = random.Random(7)
    sky = Image.fromarray(grad(*th["sky"], H, W).astype(np.uint8)).convert("RGBA")
    if theme in ("sunset", "golden"):
        g = Image.new("RGBA", (W, H))
        ImageDraw.Draw(g).ellipse((W / 2 - 150, 380, W / 2 + 150, 680), fill=(255, 235, 180, 255))
        sky.alpha_composite(g.filter(ImageFilter.GaussianBlur(30)))
    # far skyline
    far = Image.new("RGBA", (WW, H))
    d = ImageDraw.Draw(far)
    x = -20
    fc = tuple(int(c * 0.55) for c in th["sky"][0])
    while x < WW:
        bw, bh = rnd.randint(70, 150), rnd.randint(250, 520)
        d.rectangle((x, 720 - bh, x + bw, 720), fill=fc + (255,))
        if th["win"]:
            for wy in range(720 - bh + 14, 700, 22):
                for wx in range(x + 8, x + bw - 8, 18):
                    if rnd.random() < 0.25:
                        d.rectangle((wx, wy, wx + 7, wy + 10), fill=th["win"] + (150,))
        x += bw + rnd.randint(-10, 8)
    far = far.filter(ImageFilter.GaussianBlur(1.5))
    # main: shops, road, sidewalk
    main = Image.new("RGBA", (WW, H))
    d = ImageDraw.Draw(main)
    x = 0
    i = 0
    warung_x = None
    while x < WW:
        bw = rnd.randint(220, 300)
        top = rnd.randint(470, 560)
        wall = tuple(int(c) for c in np.array([rnd.randint(150, 220), rnd.randint(140, 200),
                                                rnd.randint(130, 190)]) * np.array(th["light"]))
        name = SHOPS[i % len(SHOPS)]
        if name == "WARUNG MAKAN":
            warung_x = x + bw / 2
            wall = (205, 170, 120)
        d.rectangle((x, top, x + bw, 770), fill=wall + (255,))
        d.rectangle((x, top, x + bw, top + 10), fill=tuple(int(c * 0.8) for c in wall) + (255,))
        # sign
        sc = (180, 40, 40) if name == "WARUNG MAKAN" else (rnd.randint(40, 90), rnd.randint(70, 120), rnd.randint(110, 160))
        d.rectangle((x + 20, top + 22, x + bw - 20, top + 70), fill=sc + (255,))
        d.text((x + bw / 2 - F_SIGN.getlength(name) / 2, top + 33), name, font=F_SIGN, fill=(255, 245, 225))
        # awning stripes
        ay = top + 92
        stripes = ((200, 50, 50), (245, 240, 230)) if name == "WARUNG MAKAN" else (
            (rnd.randint(60, 200), rnd.randint(60, 160), rnd.randint(60, 160)), (235, 230, 220))
        for j, sx in enumerate(range(x, x + bw, 26)):
            d.polygon([(sx, ay), (sx + 26, ay), (sx + 30, ay + 40), (sx - 4, ay + 40)], fill=stripes[j % 2] + (255,))
        # window / door
        glow = th["win"] or (120, 130, 140)
        d.rectangle((x + 26, ay + 60, x + bw - 90, 750), fill=tuple(int(c * 0.85) for c in glow) + (255,))
        d.rectangle((x + bw - 74, ay + 60, x + bw - 26, 770), fill=(70, 60, 55, 255))
        x += bw
        i += 1
    # far-side pavement, road, curb, near sidewalk
    d.rectangle((0, 770, WW, 790), fill=(130, 125, 120, 255))
    d.rectangle((0, 790, WW, 905), fill=(56, 56, 64, 255))
    for dx in range(0, WW, 90):
        d.rectangle((dx, 845, dx + 50, 851), fill=(230, 220, 180, 255))
    d.rectangle((0, 905, WW, 922), fill=(185, 180, 170, 255))
    pv = grad((150, 142, 132), (120, 112, 104), H - 922, WW).astype(np.uint8)
    main.paste(Image.fromarray(pv).convert("RGBA"), (0, 922))
    d = ImageDraw.Draw(main)
    y, step = 922, 18
    while y < H:
        d.line((0, y, WW, y), fill=(100, 94, 88, 255), width=2)
        step *= 1.35
        y += int(step)
    for vx in range(-400, WW + 400, 130):
        d.line((vx, 922, vx + (vx - WW / 2) * 0.6, H), fill=(100, 94, 88, 255), width=2)
    # lamp posts
    lamps = []
    for lx in (160, 760, 1360):
        d.rectangle((lx - 6, 540, lx + 6, 915), fill=(50, 52, 60, 255))
        d.rectangle((lx - 6, 540, lx + 70, 552), fill=(50, 52, 60, 255))
        d.rounded_rectangle((lx + 50, 548, lx + 90, 566), 6, fill=(70, 72, 80, 255))
        lamps.append((lx + 70, 572))
    tint = np.array(th["light"] + (1.0,), np.float32)
    arr = np.clip(np.array(main, np.float32) * tint, 0, 255).astype(np.uint8)
    main = Image.fromarray(arr)
    if th["lamp"] > 0:
        gl = Image.new("RGBA", (WW, H))
        gd = ImageDraw.Draw(gl)
        for lx, ly in lamps:
            gd.ellipse((lx - 110, ly - 50, lx + 110, ly + 170), fill=(255, 220, 150, int(110 * th["lamp"])))
            gd.ellipse((lx - 110, 900, lx + 110, 960), fill=(255, 220, 150, int(70 * th["lamp"])))
        main.alpha_composite(gl.filter(ImageFilter.GaussianBlur(30)))
    return dict(sky=sky, far=far, main=main, warung_x=warung_x, theme=theme)


STREETS = {k: build_street(k) for k in THEMES}


def draw_world(img, st, cam):
    img.alpha_composite(st["sky"])
    fx = int(cam * 0.35) % (WW - W)
    img.alpha_composite(st["far"].crop((fx, 0, fx + W, H)))
    cx = int(max(0, min(WW - W, cam)))
    img.alpha_composite(st["main"].crop((cx, 0, cx + W, H)))
    return cx


# ---------------------------------------------------------------- traffic
CAR_COLORS = [(190, 60, 60), (60, 90, 160), (220, 220, 225), (50, 50, 55), (200, 160, 60), (90, 140, 110)]


def draw_cars(img, t, cam, density=1.0, lights=0.5):
    ov = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(ov)
    for i in range(int(6 * density)):
        rr = random.Random(100 + i)
        lane = i % 2
        speed = rr.uniform(160, 260) * (1 if lane else -1)
        w_ = rr.uniform(150, 200)
        span = WW + 800
        x = (rr.uniform(0, span) + speed * t) % span - 400 - cam
        y = 862 if lane else 900
        if lane == 0:
            w_ *= 1.05
        col = CAR_COLORS[i % len(CAR_COLORS)]
        h_ = w_ * 0.3
        d.rounded_rectangle((x, y - h_, x + w_, y - 12), 14, fill=col + (255,))
        d.polygon([(x + w_ * 0.22, y - h_), (x + w_ * 0.32, y - h_ * 1.75), (x + w_ * 0.72, y - h_ * 1.75),
                   (x + w_ * 0.84, y - h_)], fill=tuple(int(c * 0.85) for c in col) + (255,))
        d.polygon([(x + w_ * 0.27, y - h_ - 2), (x + w_ * 0.35, y - h_ * 1.62), (x + w_ * 0.69, y - h_ * 1.62),
                   (x + w_ * 0.79, y - h_ - 2)], fill=(150, 180, 200, 255))
        for wx in (x + w_ * 0.2, x + w_ * 0.78):
            d.ellipse((wx - 20, y - 34, wx + 20, y + 6), fill=(25, 25, 28, 255))
            d.ellipse((wx - 8, y - 22, wx + 8, y - 6), fill=(120, 120, 125, 255))
        front = x + w_ if speed > 0 else x
        back = x if speed > 0 else x + w_
        d.ellipse((front - 10, y - h_ + 10, front + 10, y - h_ + 26), fill=(255, 240, 190, 255))
        d.ellipse((back - 8, y - h_ + 12, back + 8, y - h_ + 24), fill=(230, 40, 40, 255))
    img.alpha_composite(ov)


def walker(d, x, y, h, phase, col, facing=1):
    sw = math.sin(phase * 2 * math.pi)
    hip = (x, y - h * 0.47)
    for s_, c in ((1, tuple(int(v * 0.8) for v in col[:3]) + col[3:]), (-1, col)):
        a = 0.45 * sw * s_
        fx = hip[0] + math.sin(a) * h * 0.47
        fy = hip[1] + math.cos(a) * h * 0.47
        d.line((hip[0], hip[1], fx, fy), fill=c, width=max(2, int(h * 0.075)))
    d.rounded_rectangle((x - h * 0.11, y - h * 0.84, x + h * 0.11, y - h * 0.44), int(h * 0.06), fill=col)
    sh = (x, y - h * 0.8)
    for s_ in (1, -1):
        a = -0.4 * sw * s_
        d.line((sh[0], sh[1], sh[0] + math.sin(a) * h * 0.33, sh[1] + math.cos(a) * h * 0.33),
               fill=col, width=max(2, int(h * 0.055)))
    r = h * 0.075
    d.ellipse((x - r, y - h * 0.84 - r * 2.1, x + r, y - h * 0.84 - 0.1 * r), fill=col)


def draw_people(img, t, cam, n=6, alpha=200, near=True, far_side=True, tint=(40, 38, 50)):
    ov = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(ov)
    for i in range(n):
        rr = random.Random(500 + i)
        speed = rr.uniform(45, 75) * (1 if i % 2 else -1)
        span = WW + 400
        x = (rr.uniform(0, span) + speed * t) % span - 200 - cam
        ph = t * abs(speed) / 70 + rr.random()
        col = tuple(min(255, int(c + rr.randint(-10, 30))) for c in tint) + (alpha,)
        if far_side and i % 3 == 0:
            walker(d, x, 782, 90, ph, col)
        elif near:
            walker(d, x, 955 + (i % 2) * 12, 190, ph, col)
    img.alpha_composite(ov.filter(ImageFilter.GaussianBlur(1.2)))


RND = random.Random(11)
DROPS = [(RND.uniform(0, W + 300), RND.uniform(0, H), RND.uniform(900, 1300), RND.uniform(18, 36))
         for _ in range(320)]
MOTES = [(RND.uniform(0, W), RND.uniform(0, H), RND.uniform(8, 25), RND.uniform(0, 6.28),
          RND.uniform(1.5, 4)) for _ in range(70)]


def rain(img, t, alpha=120):
    ov = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(ov)
    for x, y, v, ln in DROPS:
        yy = (y + v * t) % (H + 40) - 40
        xx = (x - 0.2 * v * t) % (W + 300) - 150
        d.line((xx, yy, xx - ln * 0.2, yy + ln), fill=(200, 215, 255, alpha), width=1)
    img.alpha_composite(ov)


def motes(img, t, color=(255, 236, 190)):
    ov = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(ov)
    for x, y, v, ph, r in MOTES:
        yy = (y - v * t) % H
        xx = x + 15 * math.sin(t + ph)
        a = int(110 + 100 * math.sin(t * 2 + ph))
        d.ellipse((xx - r, yy - r, xx + r, yy + r), fill=color + (max(a, 0),))
    img.alpha_composite(ov.filter(ImageFilter.GaussianBlur(1.2)))


def _vignette():
    y, x = np.mgrid[0:H, 0:W]
    dd = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2)
    v = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    v.putalpha(Image.fromarray(np.clip((dd - 0.7) * 260, 0, 190).astype(np.uint8)))
    return v


VIG = _vignette()

# blurred backdrops for close-ups
BLUR = {k: None for k in THEMES}


def blurred(theme, cam=900):
    if BLUR[theme] is None:
        img = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        draw_world(img, STREETS[theme], cam)
        BLUR[theme] = img.filter(ImageFilter.GaussianBlur(14))
    return BLUR[theme]


# ---------------------------------------------------------------- warung interior
def build_warung():
    im = Image.fromarray(grad((120, 82, 50), (70, 45, 30), H, W).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 300), fill=(150, 105, 65))
    for y in (330, 470):  # shelves
        d.rectangle((60, y, 660, y + 14), fill=(90, 60, 38))
        for jx in range(80, 640, 70):
            d.rounded_rectangle((jx, y - 70, jx + 46, y), 8, fill=(200 - jx % 60, 120 + jx % 50, 70, 255))
    d.rounded_rectangle((170, 120, 550, 270), 12, fill=(40, 50, 45))
    d.text((360 - F_SIGN_L.getlength("MENU HARI INI") / 2, 135), "MENU HARI INI", font=F_SIGN_L, fill=(250, 240, 200))
    for j, ln in enumerate(["Nasi • Sayur • Tempe", "Telur • Ayam • Es Teh"]):
        d.text((360 - F_SIGN.getlength(ln) / 2, 190 + j * 34), ln, font=F_SIGN, fill=(230, 230, 210))
    glow = Image.new("RGBA", (W, H))
    ImageDraw.Draw(glow).ellipse((60, -200, 660, 500), fill=(255, 220, 150, 90))
    im.alpha_composite(glow.filter(ImageFilter.GaussianBlur(60)))
    counter = Image.new("RGBA", (W, H))
    cd = ImageDraw.Draw(counter)
    cd.rectangle((0, 780, W, H), fill=(120, 80, 48))
    cd.rectangle((0, 760, W, 790), fill=(170, 120, 75))
    for vx in range(0, W, 80):
        cd.line((vx, 790, vx, H), fill=(95, 62, 38), width=3)
    for px, col in ((150, (170, 170, 175)), (560, (150, 150, 160))):
        cd.rounded_rectangle((px - 80, 650, px + 80, 770), 20, fill=col)
        cd.ellipse((px - 85, 635, px + 85, 670), fill=tuple(c + 20 for c in col))
    cd.rectangle((290, 690, 430, 765), fill=(220, 220, 225, 200))  # glass display
    for j in range(3):
        cd.ellipse((300 + j * 42, 725, 336 + j * 42, 755), fill=(230, 180, 90))
    return im, counter


WARUNG_BG, WARUNG_COUNTER = build_warung()


def steam(img, t, x, y):
    ov = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(ov)
    for j in range(4):
        ph = (t * 0.5 + j / 4) % 1
        yy = y - ph * 160
        xx = x + 20 * math.sin(t * 2 + j) + (j - 1.5) * 18
        a = int(110 * math.sin(ph * math.pi))
        d.ellipse((xx - 18, yy - 24, xx + 18, yy + 24), fill=(255, 255, 255, a))
    img.alpha_composite(ov.filter(ImageFilter.GaussianBlur(8)))


# ======================================================================
# text overlays
# ======================================================================
def say(img, t, a, b, text, name=None, col=(255, 200, 120)):
    al = fade(t, a, b, 0.3)
    if al <= 0:
        return
    lines = wrap(text, F_TEXT, 570)
    lh = F_TEXT.size + 11
    box_h = lh * len(lines) + (50 if name else 28)
    ov = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(ov)
    top = SUB_BOTTOM - box_h
    d.rounded_rectangle((36, top, W - 36, SUB_BOTTOM), 22, fill=(12, 12, 22, int(185 * al)))
    if name:
        d.text((62, top + 12), name, font=F_NAME, fill=col + (int(255 * al),))
    img.alpha_composite(ov)
    cy = top + (44 if name else 14) + lh * len(lines) / 2 - 3
    img.alpha_composite(text_layer(lines, F_TEXT, (250, 248, 240), cy, al, spacing=11))


def narr(img, t, a, b, text, y=None):
    al = fade(t, a, b, 0.45)
    if al <= 0:
        return
    lines = wrap(text, F_NARR, 590)
    lh = F_NARR.size + 12
    top = NARR_TOP + 20 if y is None else y
    ov = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(ov)
    d.rounded_rectangle((36, top, W - 36, top + lh * len(lines) + 30), 22, fill=(0, 0, 0, int(120 * al)))
    img.alpha_composite(ov)
    img.alpha_composite(text_layer(lines, F_NARR, (255, 246, 228), top + 15 + lh * len(lines) / 2, al, spacing=12))


def bubble(img, t, a, b, text, kind="pikiran", center=(360, 330), tail_to=None):
    al = fade(t, a, b, 0.25)
    if al <= 0:
        return
    n = max(1, int((t - a) * 40))
    shown = text[:n]
    pop = 0.85 + 0.15 * ease((t - a) / 0.25)
    lines = wrap(text, F_BUB, 470)
    lh = F_BUB.size + 8
    bw = max(F_BUB.getlength(l) for l in lines) + 70
    bh = lh * len(lines) + 70
    bw, bh = bw * pop, bh * pop
    cx, cy = center
    if kind == "pikiran":
        fill, edge, lab, lab_col = (30, 48, 86), (150, 190, 255), "SUARA PIKIRAN", (170, 205, 255)
    else:
        fill, edge, lab, lab_col = (110, 36, 58), (255, 160, 175), "SUARA HATI", (255, 190, 200)
    ov = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(ov)
    A_ = int(225 * al)
    x0, y0, x1, y1 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
    # cloud bumps
    if kind == "pikiran":
        for i in range(14):
            ang = i / 14 * 2 * math.pi
            bx = cx + math.cos(ang) * bw / 2 * 0.95
            by = cy + math.sin(ang) * bh / 2 * 0.95
            r = 34 * pop
            d.ellipse((bx - r, by - r, bx + r, by + r), fill=edge + (A_,))
        for i in range(14):
            ang = i / 14 * 2 * math.pi
            bx = cx + math.cos(ang) * bw / 2 * 0.95
            by = cy + math.sin(ang) * bh / 2 * 0.95
            r = 31 * pop
            d.ellipse((bx - r, by - r, bx + r, by + r), fill=fill + (A_,))
        d.rounded_rectangle((x0, y0, x1, y1), 40, fill=fill + (A_,))
        if tail_to:
            for j, rr in enumerate((16, 11, 7)):
                f_ = (j + 1) / 4
                tx = lerp(cx, tail_to[0], f_ + 0.15)
                ty = lerp(y1, tail_to[1], f_ + 0.15)
                d.ellipse((tx - rr, ty - rr, tx + rr, ty + rr), fill=fill + (A_,), outline=edge + (A_,), width=2)
    else:
        d.rounded_rectangle((x0 - 3, y0 - 3, x1 + 3, y1 + 3), 44, fill=edge + (A_,))
        d.rounded_rectangle((x0, y0, x1, y1), 42, fill=fill + (A_,))
        if tail_to:
            for j, rr in enumerate((14, 10, 6)):
                f_ = (j + 1) / 4
                tx = lerp(cx, tail_to[0], f_ + 0.15)
                ty = lerp(y1, tail_to[1], f_ + 0.15)
                heart(d, tx, ty - rr / 2, rr, edge + (A_,))
        heart(d, x0 + 34, y0 + 24, 11, (255, 120, 140, A_))
    lx = x0 + (52 if kind != "pikiran" else 26)
    d.text((lx, y0 + 12), lab, font=F_NAME, fill=lab_col + (A_,))
    img.alpha_composite(ov)
    wl = wrap(shown, F_BUB, 470)
    img.alpha_composite(text_layer(wl, F_BUB, (255, 255, 255), cy + 14 - (len(lines) - len(wl)) * lh / 2,
                                   al, cx=cx, spacing=8))


def chapter_card(img, t, a, b, num, title):
    al = fade(t, a, b, 0.5)
    if al <= 0:
        return
    ov = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(ov)
    d.rectangle((0, 560, W, 760), fill=(0, 0, 0, int(150 * al)))
    d.line((200, 655, 520, 655), fill=(255, 220, 180, int(160 * al)), width=1)
    img.alpha_composite(ov)
    img.alpha_composite(text_layer([num], F_CHAP_S, (255, 214, 170), 618, al))
    img.alpha_composite(text_layer(wrap(title, F_CHAP_L, 640), F_CHAP_L, (255, 248, 236), 705, al))


def day_card(img, t, a, b, text):
    al = fade(t, a, b, 0.4)
    if al <= 0:
        return
    ov = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(ov)
    w_ = F_CHAP_L.getlength(text) + 60
    d.rounded_rectangle((W / 2 - w_ / 2, 170, W / 2 + w_ / 2, 246), 38, fill=(0, 0, 0, int(140 * al)))
    img.alpha_composite(ov)
    img.alpha_composite(text_layer([text], F_CHAP_L, (255, 245, 230), 208, al))


# ======================================================================
# scenes  (each: fn(img, lt) -> dict(grade=..., zoom=..., focus=...))
# ======================================================================
CHILD_X = 1250          # world x where the child sits
COLD = (0.9, 0.96, 1.08)
NEUTRAL = (1.0, 1.0, 1.0)
WARM = (1.08, 1.0, 0.88)
GOLD = (1.12, 1.0, 0.82)


def sc_title(img, t):
    img.alpha_composite(blurred("dusk"))
    img.alpha_composite(Image.new("RGBA", (W, H), (10, 12, 30, 150)))
    motes(img, t, (210, 220, 255))
    a = fade(t, 0.5, 7.0, 1.0)
    g = Image.new("RGBA", (W, H))
    r = 190 + 12 * math.sin(t * 3)
    ImageDraw.Draw(g).ellipse((W / 2 - r, 560 - r, W / 2 + r, 560 + r), fill=(255, 130, 150, int(70 * a)))
    img.alpha_composite(g.filter(ImageFilter.GaussianBlur(60)))
    img.alpha_composite(text_layer(["SUARA", "HATI"], F_TITLE, (255, 245, 235), 545, a, spacing=2))
    a2 = fade(t, 1.6, 7.0, 1.0)
    img.alpha_composite(text_layer(["Episode 1"], F_CHAP_S, (255, 205, 170), 690, a2))
    img.alpha_composite(text_layer(["Mereka yang Selalu", "Kita Abaikan"], F_EP, (235, 220, 235), 760, a2))
    return dict(grade=COLD)


def street_shot(img, t, theme, cam, people=6, cars=1.0, lights=0.5):
    st = STREETS[theme]
    cx = draw_world(img, st, cam)
    draw_cars(img, t, cx, cars, lights)
    draw_people(img, t, cx, people, alpha=170, near=True)
    return cx


def sc_a1(img, lt, t):
    """Adegan 1 — 7..48"""
    if lt < 8:  # establishing, pan right, Arka enters walking toward camera
        cam = lerp(700, 950, ease(lt / 8))
        cx = street_shot(img, t, "dusk", cam)
        cxs = CHILD_X - cx
        shadow_ellipse(img, cxs, 1110, 150)
        put(img, child("sit", "tired", t, k=0.55), cxs, 1112)
        chapter_card(img, lt, 0.3, 4.2, "ADEGAN 1", "Pertemuan yang Biasa")
        narr(img, lt, 4.4, 8.0, "Sore hari. Jalanan kota ramai seperti biasa. Di pinggir jalan, seorang anak kecil duduk meminta-minta.")
        if lt > 5:
            p = (lt - 5) / 3
            put(img, gif_frame(t), lerp(200, 260, p), lerp(1000, 1060, p), scale=lerp(0.55, 0.75, p))
        return dict(grade=COLD)
    if lt < 15:  # Arka approaches, child stands and walks over
        p = (lt - 8) / 7
        cam = 950
        cx = street_shot(img, t, "dusk", cam)
        ax = lerp(260, 250, p)
        ay = lerp(1060, 1180, p)
        put(img, gif_frame(t), ax, ay, scale=lerp(0.75, 1.25, ease(p)))
        cxs = CHILD_X - cx
        if lt < 10.5:
            put(img, child("sit", "tired", t, look_up=0.5, k=0.55), cxs, 1112)
        else:
            q = ease((lt - 10.5) / 4)
            bob = -abs(math.sin(lt * 7)) * 6 * (1 - q * 0.5) if q < 1 else 0
            put(img, child("stand", "hope", t, look_up=0.8, k=lerp(0.6, 0.92, q)),
                lerp(cxs, 520, q), lerp(1120, 1180, q) + bob)
        return dict(grade=COLD)
    if lt < 19.5:  # child close-up asking
        img.alpha_composite(blurred("dusk"))
        q = (lt - 15) / 4.5
        put(img, child_closeup("hope", t, look_up=1.0), 360, 1010 + 6 * math.sin(t * 2), scale=1 + 0.04 * q)
        say(img, lt, 15.3, 19.4, "Om... boleh minta uang untuk makan?", "ANAK KECIL", (255, 210, 140))
        return dict(grade=COLD)
    if lt < 22:  # Arka reaction
        img.alpha_composite(blurred("dusk", 300))
        q = (lt - 19.5) / 2.5
        put(img, FACE["notice"], 360, 560, "center", scale=1.0 + 0.05 * q)
        narr(img, lt, 19.7, 22, "Arka berhenti sebentar. Ia merogoh sakunya.")
        return dict(grade=COLD)
    if lt < 29:  # pockets: phone & wallet, no cash; apology
        cam = 950
        cx = street_shot(img, t, "dusk", cam, people=4)
        jig = 3 * math.sin(t * 18) if lt < 24 else 0
        put(img, ARKA["front"], 250 + jig, 1200, scale=1.0)
        put(img, child("stand", "hope" if lt < 26 else "sad", t, look_up=0.8, k=0.92), 540, 1185)
        img.alpha_composite(phone_wallet_layer(fade(lt, 22.3, 25.8, 0.3), t))
        narr(img, lt, 22.2, 25.8, "Hanya ada ponsel dan dompet — tanpa uang tunai.")
        say(img, lt, 25.9, 29, "Maaf ya, Dek. Om nggak bawa uang.", "ARKA", (150, 195, 255))
        return dict(grade=COLD)
    # 29..41 Arka walks away; thoughts
    p = (lt - 29) / 12
    cam = 950
    cx = street_shot(img, t, "dusk", cam)
    cxs = CHILD_X - cx
    ph = t * 1.6
    if p < 0.35:
        put(img, child("stand", "sad", t, k=lerp(0.92, 0.6, ease(p / 0.35))), lerp(540, cxs, ease(p / 0.35)), lerp(1170, 1112, ease(p / 0.35)))
    else:
        put(img, child("sit", "sad", t, k=0.55), cxs, 1112)
    put(img, ARKA["back"], lerp(260, 150, p), lerp(1230, 1010, p) + walk_bob(ph), scale=lerp(0.95, 0.5, p),
        rot=1.5 * math.sin(ph * 2 * math.pi))
    tail = (lerp(260, 150, p), lerp(500, 680, p))
    bubble(img, lt, 29.6, 32.4, "Ah, paling juga penipu.", "pikiran", (380, 330), tail)
    bubble(img, lt, 32.6, 36.6, "Zaman sekarang banyak orang memanfaatkan anak kecil untuk meminta uang.",
           "pikiran", (380, 330), tail)
    bubble(img, lt, 36.8, 41, "Kalau dikasih, belum tentu uangnya benar-benar untuk makan.",
           "pikiran", (380, 330), tail)
    return dict(grade=COLD)


def pass_by(img, t, lt, theme, dur, cam=950, child_pose="sit", child_expr="tired", rain_on=False,
            people=5, lights=0.5):
    cx = street_shot(img, t, theme, cam, people=people, lights=lights)
    cxs = CHILD_X - cx
    shadow_ellipse(img, cxs, 1110, 150)
    put(img, child(child_pose, child_expr, t, k=0.58), cxs, 1112)
    p = lt / dur
    x = lerp(-120, W + 120, p)
    ph = t * 1.5
    put(img, side_walk(ph % 1), x, 1215 + walk_bob(ph), scale=1.0, flip=True)
    if rain_on:
        rain(img, t, 130)
    return x


def sc_a2(img, lt, t):
    """Adegan 2 — 48..80 (Hari demi hari)"""
    if lt < 1.8:
        img.alpha_composite(blurred("grey"))
        chapter_card(img, lt, 0.0, 1.8, "ADEGAN 2", "Hari demi Hari")
        return dict(grade=NEUTRAL)
    if lt < 9:
        l = lt - 1.8
        x = pass_by(img, t, l, "grey", 7.2, child_expr="neutral", lights=0)
        day_card(img, l, 0.1, 2.4, "Hari ke-2")
        bubble(img, l, 2.4, 7.0, "Masih anak itu.", "pikiran", (360, 330), (x, 470))
        return dict(grade=NEUTRAL)
    if lt < 16.5:
        l = lt - 9
        x = pass_by(img, t, l, "rain", 7.5, child_expr="sad", rain_on=True, people=3, lights=0.7)
        day_card(img, l, 0.1, 2.4, "Hari ke-3")
        bubble(img, l, 2.4, 7.3, "Kenapa setiap hari di sini?", "pikiran", (360, 330), (x, 470))
        return dict(grade=COLD)
    if lt < 24:
        l = lt - 16.5
        x = pass_by(img, t, l, "evening", 7.5, child_pose="sit_tired", child_expr="tired", lights=1.0)
        day_card(img, l, 0.1, 2.2, "Hari ke-4")
        bubble(img, l, 2.2, 4.6, "Sudahlah, bukan urusanku.", "pikiran", (360, 330), (x, 470))
        narr(img, l, 4.8, 7.5, "Anak itu terlihat lebih lelah. Tetapi Arka tetap berjalan melewatinya.")
        return dict(grade=COLD)
    l = lt - 24  # 8s reflective close-up
    img.alpha_composite(blurred("evening"))
    img.alpha_composite(Image.new("RGBA", (W, H), (0, 0, 20, 120)))
    put(img, FACE["down"], 360, 600, "center", scale=1.0 + 0.012 * l)
    narr(img, l, 0.3, 3.8, "Bukan karena ia tidak peduli.", y=170)
    narr(img, l, 3.9, 8.0, "Melainkan karena ia terlalu sibuk mencari alasan untuk tidak peduli.", y=170)
    return dict(grade=COLD)


def sc_a3(img, lt, t):
    """Adegan 3 — 80..113"""
    if lt < 9:  # sunset, Arka walks in and stops
        cam = 950
        cx = street_shot(img, t, "sunset", cam, people=2, cars=0.5, lights=0.6)
        cxs = CHILD_X - cx
        shadow_ellipse(img, cxs, 1110, 150)
        put(img, child("sit_tired", "tired", t, k=0.58), cxs, 1112)
        p = min(1, lt / 6)
        x = lerp(-120, 230, ease(p) if p < 1 else 1)
        if lt < 6:
            ph = t * 1.5
            put(img, side_walk(ph % 1), x, 1215 + walk_bob(ph), flip=True)
        else:
            put(img, ARKA["side"], x, 1215, flip=True)
        chapter_card(img, lt, 0.2, 3.0, "ADEGAN 3", "Sesuatu yang Berubah")
        narr(img, lt, 3.2, 6.2, "Suatu sore, Arka kembali melewati tempat itu.")
        narr(img, lt, 6.3, 9.0, "Anak itu duduk sendirian. Tak banyak orang yang memperhatikannya.")
        return dict(grade=WARM, zoom=1 + 0.06 * ease(max(0, lt - 6) / 3), focus=(300, 900))
    if lt < 12:
        img.alpha_composite(blurred("sunset", 300))
        put(img, FACE["worry"], 360, 560, "center", scale=1 + 0.03 * (lt - 9))
        narr(img, lt, 9.2, 12, "Arka berhenti. Ia memperhatikan anak itu lebih lama dari biasanya.")
        return dict(grade=WARM)
    if lt < 26.5:  # inner voices with heartbeat
        l = lt - 12
        img.alpha_composite(blurred("sunset", 300))
        mix = ease((l - 8) / 3)
        img.alpha_composite(Image.new("RGBA", (W, H), (int(lerp(20, 120, mix)), 20, int(lerp(70, 40, mix)), 110)))
        beat = max(0, math.sin(l * math.pi * 2 * 1.15)) ** 10
        img.alpha_composite(Image.new("RGBA", (W, H), (150, 20, 40, int(50 * beat))))
        face = FACE["conflict"] if l < 8 else FACE["resolve"]
        put(img, face, 360 + 3 * beat, 860, "center", scale=0.8 + 0.02 * beat)
        bubble(img, l, 0.2, 2.6, "Jangan ikut campur.", "pikiran", (360, 330), (300, 600))
        bubble(img, l, 2.7, 5.2, "Kamu tidak tahu siapa mereka.", "pikiran", (360, 330), (300, 640))
        bubble(img, l, 5.3, 8.0, "Bagaimana kalau ini hanya modus penipuan?", "pikiran", (360, 330), (300, 640))
        bubble(img, l, 8.2, 11.1, "Tapi bagaimana kalau dia benar-benar lapar?", "hati", (360, 330), (420, 640))
        bubble(img, l, 11.2, 14.5, "Bagaimana kalau memang tidak ada orang yang mau membantunya?", "hati",
               (360, 330), (420, 640))
        return dict(grade=COLD if l < 8 else WARM)
    if lt < 29.5:  # child close-up once more
        img.alpha_composite(blurred("sunset"))
        put(img, child_closeup("tired", t, 0.0), 360, 1010, scale=1.0 + 0.01 * (lt - 26.5))
        narr(img, lt, 26.6, 29.5, "Arka terdiam. Ia melihat anak itu sekali lagi...", y=860)
        return dict(grade=WARM)
    # pan to warung
    l = lt - 29.5
    st = STREETS["sunset"]
    cam = lerp(950, st["warung_x"] - W / 2, ease(l / 3.5))
    street_shot(img, t, "sunset", cam, people=2, cars=0.5)
    narr(img, l, 0.4, 3.5, "...lalu sebuah warung makan, tidak jauh dari tempat itu.")
    return dict(grade=WARM, zoom=1 + 0.08 * ease(l / 3.5), focus=(360, 640))


def sc_a4(img, lt, t):
    """Adegan 4 — 113..138"""
    if lt < 6:  # Arka walks to warung (side walk, pan)
        st = STREETS["sunset"]
        cam = st["warung_x"] - W / 2
        street_shot(img, t, "sunset", cam, people=2, cars=0.5)
        ph = t * 1.5
        put(img, side_walk(ph % 1), lerp(-100, 380, lt / 6), 1215 + walk_bob(ph), flip=True)
        chapter_card(img, lt, 0.2, 2.6, "ADEGAN 4", "Keputusan Kecil")
        narr(img, lt, 2.8, 6.0, "Arka melangkah menuju warung. Ia memesan dua bungkus makanan sederhana.")
        return dict(grade=WARM)
    if lt < 17:  # inside warung
        l = lt - 6
        img.alpha_composite(WARUNG_BG)
        put(img, ibu("smile" if l > 7.4 else "neutral", t), 360, 800 + 3 * math.sin(t * 1.5))
        img.alpha_composite(WARUNG_COUNTER)
        steam(img, t, 150, 640)
        steam(img, t + 1.3, 560, 640)
        if l > 4.5:
            k_ = min(1, (l - 4.5) / 0.4)
            put(img, BUNGKUS, 330, 762 - 30 * (1 - k_), alpha=k_)
            put(img, BUNGKUS, 420, 762 - 30 * (1 - k_), alpha=k_, flip=True)
        put(img, ARKA["back"], 150, 1420, scale=1.3)
        say(img, l, 0.4, 3.4, "Untuk siapa, Mas?", "PENJUAL WARUNG", (255, 190, 150))
        say(img, l, 3.5, 7.3, "Untuk anak kecil yang di pinggir jalan itu, Bu.", "ARKA", (150, 195, 255))
        narr(img, l, 7.5, 11, "Penjual itu hanya tersenyum kecil.", y=170)
        return dict(grade=WARM)
    # 17..25 Arka walks back toward the child with food, does NOT pass by
    l = lt - 17
    cam = 950
    cx = street_shot(img, t, "sunset", cam, people=2, cars=0.5)
    cxs = CHILD_X - cx
    shadow_ellipse(img, cxs, 1110, 150)
    put(img, child("sit_tired", "tired", t, k=0.58), cxs, 1112)
    p = l / 8
    x = lerp(-120, cxs - 170, ease(p))
    ph = t * 1.5
    moving = p < 0.97
    spr = side_walk(ph % 1) if moving else ARKA["side"]
    y = 1215 + (walk_bob(ph) if moving else 0)
    put(img, spr, x, y, flip=True)
    put(img, BUNGKUS, x + 22, y - 330 + (4 * math.sin(ph * 4 * math.pi) if moving else 0), scale=0.55)
    narr(img, l, 0.3, 4.0, "Untuk pertama kalinya, ia tidak berjalan melewatinya.")
    narr(img, l, 4.1, 8.0, "Ia justru berjalan mendekatinya.")
    return dict(grade=WARM)


def two_shot_bg(img):
    img.alpha_composite(blurred("sunset", 1000))
    ov = Image.new("RGBA", (W, H))
    ImageDraw.Draw(ov).rectangle((0, 980, W, H), fill=(120, 100, 90, 120))
    img.alpha_composite(ov.filter(ImageFilter.GaussianBlur(20)))


def sc_a5(img, lt, t):
    """Adegan 5 — 138..168"""
    if lt < 6:  # "Dek..." / "Iya, Om?"
        two_shot_bg(img)
        put(img, ARKA["front"], 210, 1300, scale=1.15)
        lu = ease((lt - 2.5) / 0.8)
        put(img, child("sit", "tired" if lt < 2.6 else "surprised", t, look_up=lu, k=1.0), 520, 1150)
        chapter_card(img, lt, 0.0, 1.6, "ADEGAN 5", "Memberi Tanpa Banyak Bertanya")
        say(img, lt, 1.4, 3.0, "Dek...", "ARKA", (150, 195, 255))
        say(img, lt, 3.1, 6.0, "Iya, Om?", "ANAK KECIL", (255, 210, 140))
        return dict(grade=WARM)
    if lt < 11:  # Arka crouches, offers food
        l = lt - 6
        two_shot_bg(img)
        put(img, ARKA["sit"], 230, 1190, scale=1.85)
        put(img, child("sit", "surprised", t, look_up=0.6, k=1.0), 530, 1150)
        reach = ease(l / 1.2)
        put(img, BUNGKUS, lerp(330, 390, reach), lerp(930, 900, reach), "center", scale=0.9)
        say(img, l, 0.3, 5.0, "Ini buat kamu. Makan ya.", "ARKA", (150, 195, 255))
        return dict(grade=WARM)
    if lt < 15:  # child unsure
        img.alpha_composite(blurred("sunset"))
        put(img, child_closeup("surprised", t, 0.3), 360, 1010)
        say(img, lt, 11.2, 15, "Ini... buat aku, Om?", "ANAK KECIL", (255, 210, 140))
        return dict(grade=WARM)
    if lt < 18:
        img.alpha_composite(blurred("sunset", 300))
        put(img, FACE["soft"], 360, 560, "center", scale=1 + 0.02 * (lt - 15))
        say(img, lt, 15.2, 18, "Iya. Buat kamu.", "ARKA", (150, 195, 255))
        return dict(grade=WARM)
    if lt < 22:
        img.alpha_composite(blurred("golden"))
        put(img, child_closeup("smile", t, 0.4, pose="hold"), 360, 1010)
        put(img, BUNGKUS, 360, 800, "center", scale=1.6)
        say(img, lt, 18.2, 22, "Makasih banyak, Om...", "ANAK KECIL", (255, 210, 140))
        return dict(grade=GOLD)
    l = lt - 22  # warm two-shot, narration
    two_shot_bg(img)
    motes(img, t)
    put(img, ARKA["sit"], 230, 1190, scale=1.85)
    put(img, child("sit_hold", "happy", t, k=1.0), 530, 1150)
    put(img, BUNGKUS, 530, 960, "center", scale=0.9)
    put(img, FACE["smile"], 360, 300, "center", alpha=fade(l, 0.3, 8, 0.6), scale=0.55)
    narr(img, l, 0.3, 4.0, "Tidak ada percakapan panjang. Tidak ada pertanyaan tentang siapa anak itu.", y=470)
    narr(img, l, 4.1, 8.0, "Hanya sebungkus makanan sederhana, untuk seseorang yang mungkin sedang membutuhkannya.", y=470)
    return dict(grade=GOLD)


def sc_a6(img, lt, t):
    """Adegan 6 — 168..196"""
    if lt < 9:
        cam = 300
        street_shot(img, t, "golden", cam, people=4, cars=0.6, lights=0)
        motes(img, t)
        p = lt / 9
        put(img, gif_frame(t), 360, lerp(1060, 1230, p), scale=lerp(0.85, 1.3, ease(p)))
        chapter_card(img, lt, 0.1, 2.4, "ADEGAN 6", "Percakapan di Dalam Diri")
        narr(img, lt, 2.5, 5.0, "Langkahnya kali ini terasa berbeda.")
        bubble(img, lt, 5.1, 7.0, "Bagaimana kalau memang dia penipu?", "pikiran", (360, 300), (360, 520))
        bubble(img, lt, 7.0, 9.0, "Bagaimana kalau dia benar-benar lapar?", "hati", (360, 300), (360, 520))
        return dict(grade=GOLD)
    l = lt - 9
    img.alpha_composite(blurred("golden", 300))
    motes(img, t)
    face = FACE["calm"] if l < 14 else FACE["smile"]
    put(img, face, 360, 640, "center", scale=1 + 0.01 * l)
    lines = [(0.2, 4.0, "Selama ini aku terlalu sibuk memikirkan kemungkinan buruk..."),
             (4.1, 9.0, "Aku sampai lupa bahwa ada kemungkinan seseorang benar-benar membutuhkan pertolongan."),
             (9.1, 13.8, "Memang aku tidak tahu siapa dia. Memang aku tidak tahu seluruh ceritanya."),
             (13.9, 19.0, "Tapi setidaknya, hari ini aku tidak memilih untuk diam.")]
    for a, b, s in lines:
        say(img, l, a, b, s, "ARKA (dalam hati)", (150, 195, 255))
    return dict(grade=GOLD)


def sc_a7(img, lt, t):
    """Adegan 7 — 196..236"""
    cam = lerp(700, 1050, lt / 40)
    cx = street_shot(img, t, "golden", cam, people=8, cars=0.8, lights=0)
    motes(img, t)
    cxs = CHILD_X - cx
    shadow_ellipse(img, cxs, 1110, 150)
    put(img, child("sit_hold", "happy", t, k=0.58), cxs, 1112)
    put(img, BUNGKUS, cxs, 1112 - 140, "center", scale=0.45)
    if lt > 20:  # Arka walks away into the distance
        p = (lt - 20) / 20
        ph = t * 1.6
        put(img, ARKA["back"], lerp(cxs - 260, 200, p), lerp(1230, 1000, p) + walk_bob(ph),
            scale=lerp(0.9, 0.45, p), rot=1.2 * math.sin(ph * 2 * math.pi))
    if lt < 2.6:
        chapter_card(img, lt, 0.0, 2.6, "ADEGAN 7", "Sebuah Pelajaran Kehidupan")
    L = [(2.7, 6.2, "Setiap hari, kita bertemu dengan banyak orang."),
         (6.3, 11.3, "Ada yang meminta bantuan. Ada yang membutuhkan perhatian. Ada pula yang diam-diam sedang berjuang melawan kehidupannya."),
         (11.4, 14.6, "Namun terkadang, kita terlalu takut untuk membantu."),
         (14.7, 19.4, "Kita takut ditipu. Kita takut dimanfaatkan. Kita takut melakukan sesuatu yang salah."),
         (19.5, 25.6, "Sampai kita lupa, di balik semua kemungkinan buruk itu, mungkin ada seseorang yang benar-benar membutuhkan kebaikan kita."),
         (25.7, 29.8, "Arka tidak pernah tahu apakah anak itu benar-benar membutuhkan pertolongan."),
         (29.9, 34.8, "Tetapi hari itu, ia belajar: membantu seseorang tidak selalu harus dimulai dengan uang yang banyak."),
         (34.9, 40.0, "Terkadang, cukup dengan sebungkus makanan dan sedikit kepedulian.")]
    for a, b, s in L:
        narr(img, lt, a, b, s)
    return dict(grade=GOLD)


def sc_end(img, lt, t):
    img.alpha_composite(blurred("golden"))
    img.alpha_composite(Image.new("RGBA", (W, H), (30, 15, 25, 170)))
    motes(img, t, (255, 210, 200))
    a = fade(lt, 0.4, 14, 0.8)
    g = Image.new("RGBA", (W, H))
    ImageDraw.Draw(g).ellipse((W / 2 - 220, 300, W / 2 + 220, 740), fill=(255, 140, 160, int(60 * a)))
    img.alpha_composite(g.filter(ImageFilter.GaussianBlur(70)))
    img.alpha_composite(text_layer(["SUARA HATI"], F_TITLE, (255, 210, 215), 400, a))
    img.alpha_composite(text_layer(["“Pikiran memberimu alasan", "untuk takut. Hati memberimu",
                                    "alasan untuk berani.”"], F_QUOTE, (255, 244, 236), 590,
                                   fade(lt, 1.6, 14, 0.8), spacing=12))
    img.alpha_composite(text_layer(wrap("Jangan biarkan ketakutanmu membuatmu kehilangan kesempatan untuk berbuat baik.",
                                        F_EP, 560), F_EP, (230, 215, 230), 780, fade(lt, 3.5, 14, 0.8)))
    img.alpha_composite(text_layer(["Episode 1 — Mereka yang Selalu Kita Abaikan"], F_NAME, (255, 205, 170), 900,
                                   fade(lt, 5, 14, 0.8)))
    return dict(grade=WARM)


SCENES = [(0, 7, lambda img, lt, t: sc_title(img, t)),
          (7, 48, sc_a1), (48, 80, sc_a2), (80, 113, sc_a3), (113, 138, sc_a4),
          (138, 168, sc_a5), (168, 196, sc_a6), (196, 236, sc_a7), (236, 250, sc_end)]
DUR = SCENES[-1][1]
def frame(t):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    info = {}
    for a, b, fn in SCENES:
        if a <= t < b:
            info = fn(img, t - a, t) or {}
            k = min(ease((t - a) / 0.5), ease((b - t) / 0.5))
            break
    else:
        k = 1
    img.alpha_composite(VIG)
    z = info.get("zoom", 1.0)
    if z > 1.001:
        fx, fy = info.get("focus", (W / 2, H / 2))
        cw, ch = W / z, H / z
        x0 = min(max(0, fx - cw / 2), W - cw)
        y0 = min(max(0, fy - ch / 2), H - ch)
        img = img.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))).resize((W, H), Image.BILINEAR)
    arr = np.asarray(img.convert("RGB"), np.float32)
    g = np.array(info.get("grade", NEUTRAL), np.float32)
    arr = arr * g
    arr *= k
    return np.clip(arr, 0, 255).astype(np.uint8)


# ======================================================================
# audio
# ======================================================================
def audio(path, sr=44100):
    n = int(DUR * sr)
    t = np.arange(n) / sr
    out = np.zeros(n, np.float32)

    def note(f, start, length, vol, decay=2.0):
        s = int(start * sr)
        ln = min(int(length * sr), n - s)
        if ln <= 0:
            return
        tt = np.arange(ln) / sr
        env = np.exp(-tt * decay) * np.minimum(1, tt * 150)
        w_ = np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt) + 0.1 * np.sin(6 * np.pi * f * tt)
        out[s:s + ln] += (vol * env * w_).astype(np.float32)

    def pad(f, start, length, vol):
        s = int(start * sr)
        ln = min(int(length * sr), n - s)
        if ln <= 0:
            return
        tt = np.arange(ln) / sr
        env = np.minimum(1, tt / 1.2) * np.minimum(1, (length - tt) / 1.2)
        vib = 1 + 0.004 * np.sin(2 * np.pi * 5 * tt)
        w_ = sum(np.sin(2 * np.pi * f * m * vib * tt + m) / m for m in (1, 2, 3, 4)) * 0.5
        w_ += 0.5 * np.sin(2 * np.pi * f * 1.003 * tt)
        out[s:s + ln] += (vol * env * w_).astype(np.float32)

    hz = lambda m: 440 * 2 ** ((m - 69) / 12)
    sad = [[57, 60, 64], [53, 57, 60], [48, 55, 64], [52, 55, 59]]     # Am F C Em
    hope = [[48, 55, 64], [55, 59, 62], [57, 60, 64], [53, 57, 65]]    # C G Am F
    bar = 3.0
    tt = 0.0
    i = 0
    while tt < DUR - 1:
        prog = hope if tt >= 138 else sad
        ch = prog[i % 4]
        quiet = 92 <= tt < 106.5
        vol = 0.05 if quiet else 0.085
        note(hz(ch[0] - 12), tt, 3.2, vol * 1.3, 1.2)
        arp = ch + [ch[1] + 12, ch[2] + 12, ch[1] + 12]
        for j, m in enumerate(arp):
            note(hz(m), tt + j * bar / 6, 2.2, vol * 0.65)
        if tt >= 138:
            note(hz(ch[2] + 12), tt + bar * 0.5, 2.0, vol * 0.6)
            pad(hz(ch[0]), tt, bar + 0.6, 0.03)
            pad(hz(ch[2]), tt, bar + 0.6, 0.02)
        tt += bar
        i += 1

    rng = np.random.default_rng(0)
    noise = rng.standard_normal(n).astype(np.float32)
    city = np.convolve(noise, np.ones(40) / 40, mode="same")
    city_env = np.zeros(n, np.float32)
    for a, b, v in [(7, 48, 1), (48, 80, 1), (80, 113, 0.6), (113, 119, 0.6), (130, 138, 0.6), (168, 177, 0.7),
                    (196, 236, 0.8)]:
        m = (t >= a) & (t < b)
        city_env[m] = v * np.minimum(1, np.minimum(t[m] - a, b - t[m]) / 1.0)
    out += 0.25 * city * city_env
    # car swooshes
    for k in range(60):
        c0 = 7 + k * 3.7 + rng.uniform(0, 1.5)
        if c0 > 236:
            break
        s = int(c0 * sr)
        ln = int(1.6 * sr)
        if s + ln >= n:
            break
        e = np.sin(np.linspace(0, np.pi, ln)) ** 2
        sw = np.convolve(rng.standard_normal(ln), np.ones(12) / 12, mode="same")
        out[s:s + ln] += (0.05 * e * sw * city_env[s]).astype(np.float32)
    # rain (hari ke-3)
    rn = np.convolve(noise, np.ones(4) / 4, mode="same")
    m = (t >= 57) & (t < 64.5)
    out[m] += 0.06 * rn[m] * np.minimum(1, np.minimum(t[m] - 57, 64.5 - t[m]) / 0.6)
    # heartbeat 92..106.5
    k = 0
    while 92 + k / 1.15 < 106.5:
        for off, v in ((0, 1.0), (0.17, 0.7)):
            s = int((92 + k / 1.15 + off) * sr)
            ln = int(0.15 * sr)
            tt_ = np.arange(ln) / sr
            out[s:s + ln] += (0.32 * v * np.sin(2 * np.pi * 52 * tt_) * np.exp(-tt_ * 30)).astype(np.float32)
        k += 1
    out *= np.minimum(1, np.minimum(t / 2.0, (DUR - t) / 3.0))
    out = out / max(1e-9, np.abs(out).max()) * 0.85
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((out * 32767).astype(np.int16).tobytes())


# ======================================================================
# render
# ======================================================================
def render_chunk(args):
    idx, f0, f1, path = args
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                          "-crf", "20", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for i in range(f0, f1):
        p.stdin.write(frame(i / FPS).tobytes())
        if (i - f0) % 300 == 0:
            print(f"[chunk {idx}] {i - f0}/{f1 - f0}", flush=True)
    p.stdin.close()
    p.wait()
    return path


if __name__ == "__main__":
    out = sys.argv[1]
    if len(sys.argv) > 2:
        for s in sys.argv[2:]:
            Image.fromarray(frame(float(s))).save(f"{out}_{s}.png")
        sys.exit()
    tmp = out + "_parts"
    os.makedirs(tmp, exist_ok=True)
    wav = os.path.join(tmp, "audio.wav")
    audio(wav)
    total = int(DUR * FPS)
    nproc = os.cpu_count() or 4
    step = math.ceil(total / nproc)
    jobs = [(i, i * step, min(total, (i + 1) * step), os.path.join(tmp, f"p{i}.mp4")) for i in range(nproc)]
    with Pool(nproc) as pool:
        parts = pool.map(render_chunk, jobs)
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as f:
        for p_ in parts:
            f.write(f"file '{os.path.abspath(p_)}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out],
                   check=True)
    for p_ in parts + [wav, lst]:
        os.remove(p_)
    os.rmdir(tmp)
    print("done:", out)
