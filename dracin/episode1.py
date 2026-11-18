"""LIANG CHEN vs LONG TIAN — Episode 1: Kebangkitan Naga Kuno.

Animasi Dracin vertikal 720x1280 @60fps (~2:29), dengan pertarungan epik 60 detik.

Aset (folder assets/, dari paket proyek Dracin):
  Referensi_Desain_Karakter_1.png      — Long Tian (naga): potret, badan penuh, 30 gerakan
  Referensi_Desain_Karakter_2.png      — Liang Chen: potret, desain visual, 30 pose
  Desain_Arena_dan_Dunia_Fantasi.png   — Skill Berpedang Dewa (25 jurus)
  Adegan_Pertarungan_Fantasi.png       — arena Tian Shen Dian
  audio/                               — suara Liang Chen (l_*), Long Tian (d_*), efek suara
Efek (petir, tebasan, lingkaran sihir, pedang cahaya, partikel) dan musik digambar /
disintesis prosedural.

Pemakaian:
  python3 episode1.py out.mp4                 # render penuh
  python3 episode1.py out 12.5 70 115         # simpan still (out_12.5.png …)
"""
import math
import os
import random
import subprocess
import sys
import wave
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sprites import cutout, split_strip  # noqa: E402

W, H = 720, 1280
FPS = int(os.environ.get("FPS", 60))
HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "assets")

FD = "/usr/share/fonts/truetype/crosextra/"
CJK = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
F_TITLE = ImageFont.truetype(FD + "Caladea-Bold.ttf", 64)
F_SUBT = ImageFont.truetype(FD + "Caladea-Italic.ttf", 30)
F_TEXT = ImageFont.truetype(FD + "Carlito-Regular.ttf", 32)
F_NAME = ImageFont.truetype(FD + "Carlito-Bold.ttf", 25)
F_NARR = ImageFont.truetype(FD + "Caladea-Italic.ttf", 32)
F_CALL = ImageFont.truetype(FD + "Caladea-Bold.ttf", 46)
F_CJK_XL = ImageFont.truetype(CJK, 120)
F_CJK_L = ImageFont.truetype(CJK, 92)
F_CJK_S = ImageFont.truetype(CJK, 26)

SUB_BOTTOM = H - 290


# ======================================================================
# math helpers
# ======================================================================
def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def eio(x):  # ease in-out cubic
    x = min(1.0, max(0.0, x))
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def eout(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def lerp(a, b, x):
    return a + (b - a) * x


def fade(t, a, b, f=0.3):
    if t < a or t > b:
        return 0.0
    return ease((t - a) / f) * ease((b - t) / f)


def noise1(t, seed=0):
    """smooth 1-D value noise in [-1,1]"""
    i = math.floor(t)
    f = t - i
    r = lambda k: random.Random(k * 7919 + seed).uniform(-1, 1)
    u = f * f * (3 - 2 * f)
    return r(i) * (1 - u) + r(i + 1) * u


def shake(t, amp, seed=1, freq=18):
    return amp * noise1(t * freq, seed), amp * noise1(t * freq, seed + 99)


# ======================================================================
# image utilities
# ======================================================================
def to_f(im):
    return np.asarray(im.convert("RGB"), np.float32) / 255.0


def to_img(arr):
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def view(src, cx, cy, zoom, rot=0.0, size=(W, H)):
    """Subpixel camera: output pixel → source (cx,cy) centre, zoom = out px per src px."""
    w_, h_ = size
    c, s = math.cos(rot), math.sin(rot)
    a, b = c / zoom, s / zoom
    d, e = -s / zoom, c / zoom
    coeffs = (a, b, cx - a * w_ / 2 - b * h_ / 2, d, e, cy - d * w_ / 2 - e * h_ / 2)
    return src.transform(size, Image.AFFINE, coeffs, resample=Image.BICUBIC)


def upscale(im, k):
    out = im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)
    return out.filter(ImageFilter.UnsharpMask(2.0, 70, 2)) if k > 1.2 else out


def pushpull(im, w):
    h, wd = w.shape
    if min(h, wd) <= 4:
        m = (im * w[..., None]).sum((0, 1)) / max(w.sum(), 1e-6)
        return np.where(w[..., None] > 0, im, m)
    ph, pw = h % 2, wd % 2
    imp = np.pad(im, ((0, ph), (0, pw), (0, 0)), mode="edge")
    wp = np.pad(w, ((0, ph), (0, pw)))
    h2, w2 = imp.shape[0] // 2, imp.shape[1] // 2
    s_iw = (imp * wp[..., None]).reshape(h2, 2, w2, 2, 3).sum((1, 3))
    s_w = wp.reshape(h2, 2, w2, 2).sum((1, 3))
    low = pushpull(s_iw / np.maximum(s_w, 1e-6)[..., None], np.minimum(s_w, 1))
    up = np.repeat(np.repeat(low, 2, 0), 2, 1)[:h, :wd]
    return w[..., None] * im + (1 - w[..., None]) * up


def inpaint(im, boxes, pad=6):
    a = to_f(im)
    m = np.zeros(a.shape[:2], bool)
    for x0, y0, x1, y1 in boxes:
        m[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = True
    filled = pushpull(a, (~m).astype(np.float32))
    sm = to_f(to_img(filled).filter(ImageFilter.GaussianBlur(10)))
    rng = np.random.default_rng(3)
    tex = to_f(Image.fromarray((rng.random((a.shape[0] // 8, a.shape[1] // 8)) * 255).astype(np.uint8))
               .resize((a.shape[1], a.shape[0]), Image.BICUBIC).convert("RGB"))
    sm = sm * (0.85 + 0.3 * tex)
    out = np.where(m[..., None], sm, a)
    # feather seam
    mk = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5))
    k = np.asarray(mk, np.float32)[..., None] / 255
    return to_img(a * (1 - k) + out * k)


def luma_key(im, lo=0.30, hi=0.55):
    a = to_f(im)
    lum = a.max(2) * 0.6 + a.mean(2) * 0.4
    al = np.clip((lum - lo) / (hi - lo), 0, 1)
    out = im.convert("RGBA")
    out.putalpha(Image.fromarray((al * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)))
    return out


# ======================================================================
# assets
# ======================================================================
print("menyiapkan aset…", flush=True)
REF_D = Image.open(os.path.join(A, "Referensi_Desain_Karakter_1.png")).convert("RGB")
REF_L = Image.open(os.path.join(A, "Referensi_Desain_Karakter_2.png")).convert("RGB")
SKILL = Image.open(os.path.join(A, "Desain_Arena_dan_Dunia_Fantasi.png")).convert("RGB")
ARENA0 = Image.open(os.path.join(A, "Adegan_Pertarungan_Fantasi.png")).convert("RGB")
ARENA = inpaint(ARENA0, [(25, 15, 322, 207), (22, 393, 332, 562), (1288, 22, 1532, 218),
                         (1262, 443, 1532, 642), (18, 773, 352, 987), (1272, 828, 1532, 987),
                         (540, 905, 1075, 1012)])

# Long Tian
DRAGON_FULL = upscale(REF_D.crop((418, 0, 832, 632)), 2.0)
DRAGON_HEAD = upscale(REF_D.crop((96, 0, 440, 462)), 2.2)
_dc = [(10, 245), (250, 440), (445, 705), (710, 905), (910, 1060), (1065, 1245)]
_dr = [(657, 770), (777, 885), (893, 1000), (1005, 1115), (1120, 1235)]


def dtile(r, c):
    x0, x1 = _dc[c]
    y0, y1 = _dr[r]
    return REF_D.crop((x0 + 30, y0 + 6, x1 - 6, y1 - 5))


_sc = [(8, 255), (262, 505), (510, 755), (760, 1005), (1010, 1250)]
_sr = [(388, 540), (545, 705), (710, 870), (875, 1030), (1035, 1195)]


def stile(r, c):
    x0, x1 = _sc[c]
    y0, y1 = _sr[r]
    return SKILL.crop((x0 + 6, y0 + 30, x1 - 6, y1 - 6))


HERO = SKILL.crop((0, 0, 432, 386))
LIANG_PORTRAIT = upscale(REF_L.crop((0, 0, 386, 646)), 2.0)

# Liang cut-outs (white background)
_kw = dict(bright=228, sat=14, keep_frac=0.25)
_r1 = split_strip(REF_L, (5, 680, 1250, 880), min_gap=3, **_kw)
_r2 = split_strip(REF_L, (5, 882, 1250, 1042), min_gap=3, **_kw)
_r3 = split_strip(REF_L, (5, 1044, 1250, 1252), min_gap=3, **_kw)
LP = {"walk0": _r1[0], "walk1": _r1[1], "walk2": _r1[2], "run": _r1[3], "lookup": _r2[0],
      "reach": _r2[1], "kneel": _r3[1], "hurt": _r3[2], "raise": _r3[6], "stand": _r3[8],
      "back": _r3[9]}
LP["front"] = cutout(REF_L, (378, 214, 503, 640), **_kw)

# stage: castle/sky from the arena + procedural formation floor
STW, STH = 1000, 1500


def build_stage():
    top = ARENA.crop((330, 0, 1262, 560)).resize((STW, int(560 * STW / 932)), Image.LANCZOS)
    st = Image.new("RGB", (STW, STH), (10, 12, 30))
    st.paste(top, (0, 0))
    floor_y = top.height - 40
    g = np.linspace(0, 1, STH - floor_y)[:, None, None]
    base = (np.array([18, 22, 48]) * (1 - g) + np.array([6, 8, 20]) * g)
    fl = Image.fromarray(np.repeat(base, STW, 1).astype(np.uint8))
    # sky reflection
    refl = top.transpose(Image.FLIP_TOP_BOTTOM).filter(ImageFilter.GaussianBlur(6))
    refl = refl.resize((STW, STH - floor_y))
    fl = Image.blend(fl, refl, 0.22)
    d = ImageDraw.Draw(fl, "RGBA")
    cx, cy = STW / 2, 520
    for i, r in enumerate([90, 170, 260, 360, 470, 590]):
        col = (255, 205, 110, 150) if i % 2 == 0 else (120, 170, 255, 130)
        d.ellipse((cx - r * 1.6, cy - r * 0.42, cx + r * 1.6, cy + r * 0.42), outline=col, width=3)
    for k in range(24):
        ang = k / 24 * 2 * math.pi
        d.line((cx + math.cos(ang) * 150, cy + math.sin(ang) * 40, cx + math.cos(ang) * 950,
                cy + math.sin(ang) * 250), fill=(200, 170, 110, 60), width=2)
    for y in range(20, STH - floor_y, 70):  # stone joints
        d.line((0, y, STW, y), fill=(60, 70, 110, 40), width=1)
    glow = fl.filter(ImageFilter.GaussianBlur(8))
    fl = Image.fromarray(np.clip(np.asarray(fl, np.float32) + np.asarray(glow, np.float32) * 0.6, 0, 255)
                         .astype(np.uint8))
    st.paste(fl, (0, floor_y))
    # fog over the seam
    fog = Image.new("RGBA", (STW, STH))
    ImageDraw.Draw(fog).rectangle((0, floor_y - 60, STW, floor_y + 70), fill=(120, 140, 210, 120))
    st = st.convert("RGBA")
    st.alpha_composite(fog.filter(ImageFilter.GaussianBlur(40)))
    return st.convert("RGB"), floor_y


STAGE, FLOOR_Y = build_stage()
DRAGON_KEY = luma_key(REF_D.crop((418, 0, 832, 632)).resize((828, 1264), Image.LANCZOS))
_panel_cache = {}


def panel_assets(key, src, band_h, over=1.18):
    """Precompute big foreground + blurred cover background for a landscape source."""
    if key in _panel_cache:
        return _panel_cache[key]
    z = band_h * over / src.height
    big = upscale(src, max(z, 1.0))
    cov = src.resize((int(src.width * H / src.height * 1.15), int(H * 1.15)), Image.BILINEAR)
    cov = cov.filter(ImageFilter.GaussianBlur(26))
    cov = Image.fromarray((np.asarray(cov, np.float32) * 0.55).astype(np.uint8))
    m = Image.new("L", (W, band_h), 0)
    ImageDraw.Draw(m).rectangle((0, 26, W, band_h - 26), fill=255)
    m = m.filter(ImageFilter.GaussianBlur(14))
    _panel_cache[key] = (big, cov, m, z)
    return _panel_cache[key]


def panel(key, src, lt, dur, band_h=560, cy=600, pan=(0.0, 0.0), zoom=(1.0, 1.08), sh=(0, 0), rot=0.0):
    big, cov, m, z = panel_assets(key, src, band_h)
    p = eio(lt / dur)
    zz = lerp(*zoom, p)
    bgc = view(cov, cov.width / 2 + pan[0] * 20 * p, cov.height / 2, 1.0 + 0.04 * p, rot * 0.5)
    fx_ = big.width / 2 + lerp(-pan[0], pan[0], p) * big.width * 0.18 + sh[0]
    fy_ = big.height / 2 + lerp(-pan[1], pan[1], p) * big.height * 0.1 + sh[1]
    fg = view(big, fx_, fy_, zz, rot, size=(W, band_h))
    bgc.paste(fg, (0, int(cy - band_h / 2)), m)
    return bgc


def full(src, lt, dur, c0, c1, z0, z1, sh=(0, 0), rot=0.0):
    p = eio(lt / dur)
    z = lerp(z0, z1, p)
    z = max(z, W / src.width * 1.01, H / src.height * 1.01)
    hw, hh = W / (2 * z), H / (2 * z)
    cx = min(max(lerp(c0[0], c1[0], p) + sh[0], hw), src.width - hw)
    cy = min(max(lerp(c0[1], c1[1], p) + sh[1], hh), src.height - hh)
    return view(src, cx, cy, z, rot)


# ======================================================================
# FX (drawn at half resolution, additive)
# ======================================================================
FW, FH = W // 2, H // 2


class FX:
    def __init__(self):
        self.core = Image.new("RGB", (FW, FH))
        self.glow = Image.new("RGB", (FW, FH))
        self.dc = ImageDraw.Draw(self.core)
        self.dg = ImageDraw.Draw(self.glow)

    def line(self, pts, col, w=2, gw=10, gk=1.0):
        p = [(x / 2, y / 2) for x, y in pts]
        self.dg.line(p, fill=tuple(int(c * gk) for c in col), width=gw, joint="curve")
        self.dc.line(p, fill=(255, 255, 255) if w > 0 else col, width=max(1, w), joint="curve")

    def ellipse(self, box, col, w=3, gw=12, fill=None):
        b = [v / 2 for v in box]
        if fill:
            self.dg.ellipse(b, fill=fill)
        self.dg.ellipse(b, outline=col, width=gw)
        self.dc.ellipse(b, outline=col, width=w)

    def poly(self, pts, col, glow_col=None):
        p = [(x / 2, y / 2) for x, y in pts]
        self.dc.polygon(p, fill=col)
        self.dg.polygon(p, fill=glow_col or col)

    def dot(self, x, y, r, col):
        self.dg.ellipse((x / 2 - r, y / 2 - r, x / 2 + r, y / 2 + r), fill=col)

    def apply(self, frame, k=1.0):
        g = self.glow.filter(ImageFilter.GaussianBlur(9))
        g2 = self.glow.filter(ImageFilter.GaussianBlur(3))
        acc = (np.asarray(g, np.float32) * 1.4 + np.asarray(g2, np.float32) * 0.8
               + np.asarray(self.core, np.float32)) / 255.0
        acc = np.asarray(Image.fromarray(np.clip(acc * 255, 0, 255).astype(np.uint8)).resize((W, H),
                                                                                          Image.BILINEAR),
                         np.float32) / 255.0
        return frame + acc * k


def bolt(seed, p0, p1, disp=0.22, depth=6):
    rnd = random.Random(seed)
    pts = [p0, p1]
    d = disp * math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    for _ in range(depth):
        new = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            nx, ny = -(b[1] - a[1]), b[0] - a[0]
            ln = math.hypot(nx, ny) or 1
            o = rnd.uniform(-d, d)
            new += [(mx + nx / ln * o, my + ny / ln * o), b]
        pts = new
        d *= 0.55
    return pts


def lightning(fx, t, seed, p0, p1, col=(130, 150, 255), life=0.18, every=0.55, branches=2):
    """flickering bolt that re-strikes every `every` seconds"""
    k = int(t / every)
    ph = (t % every) / life
    if ph > 1:
        return 0.0
    s = seed * 1000 + k
    pts = bolt(s, p0, p1)
    inten = (1 - ph) ** 1.5
    c = tuple(int(v * inten) for v in col)
    fx.line(pts, c, w=2 if inten > 0.3 else 1, gw=int(8 + 10 * inten))
    rnd = random.Random(s + 5)
    for _ in range(branches):
        i = rnd.randrange(len(pts) // 4, len(pts) * 3 // 4)
        a = pts[i]
        ang = rnd.uniform(0, 6.28)
        L = rnd.uniform(80, 200)
        fx.line(bolt(s + i, a, (a[0] + math.cos(ang) * L, a[1] + abs(math.sin(ang)) * L)), c, w=1, gw=6)
    return inten


def slash(fx, cx, cy, rx, ry, a0, a1, prog, fade_, col=(150, 200, 255), width=26, rot=0.0):
    """crescent sword trail along an ellipse arc, revealed by prog, dimmed by fade_"""
    if fade_ <= 0 or prog <= 0:
        return
    n = 40
    end = lerp(a0, a1, min(1, prog))
    head = max(a0, end - (a1 - a0) * 0.55)
    cr, sr = math.cos(rot), math.sin(rot)
    pts_o, pts_i = [], []
    for i in range(n + 1):
        a = lerp(head, end, i / n)
        w_ = width * (i / n) ** 1.3
        for r_, lst in ((1, pts_o), (-1, pts_i)):
            x = math.cos(a) * (rx + r_ * w_ / 2)
            y = math.sin(a) * (ry + r_ * w_ / 2)
            lst.append((cx + x * cr - y * sr, cy + x * sr + y * cr))
    poly = pts_o + pts_i[::-1]
    c = tuple(int(v * fade_) for v in col)
    fx.poly(poly, tuple(int(255 * fade_) for _ in range(3)), c)
    fx.line(pts_o, c, w=0, gw=16)


def ring(fx, cx, cy, r, col, w=4, sq=0.35, gw=14):
    if r <= 0:
        return
    fx.ellipse((cx - r, cy - r * sq, cx + r, cy + r * sq), col, w=w, gw=gw)


def magic_circle(fx, cx, cy, r, t, col=(255, 200, 110), sq=0.38, k=1.0):
    c = tuple(int(v * k) for v in col)
    for rr in (r, r * 0.82, r * 0.5):
        ring(fx, cx, cy, rr, c, w=2, sq=sq, gw=8)
    for j, (rr, n, spd) in enumerate(((r * 0.91, 36, 0.6), (r * 0.66, 8, -0.9))):
        for i in range(n):
            a = i / n * 2 * math.pi + t * spd
            x0, y0 = cx + math.cos(a) * rr, cy + math.sin(a) * rr * sq
            x1 = cx + math.cos(a) * (rr + (12 if j == 0 else 40))
            y1 = cy + math.sin(a) * (rr + (12 if j == 0 else 40)) * sq
            fx.line([(x0, y0), (x1, y1)], c, w=1, gw=4)
    pts = [(cx + math.cos(i * 4 * math.pi / 5 + t * 0.4) * r * 0.5,
            cy + math.sin(i * 4 * math.pi / 5 + t * 0.4) * r * 0.5 * sq) for i in range(6)]
    fx.line(pts, c, w=1, gw=6)


def light_sword(fx, x, y, h, col=(170, 210, 255), k=1.0, ang=0.0):
    c = tuple(int(v * k) for v in col)
    ca, sa = math.cos(ang), math.sin(ang)

    def P(px, py):
        return (x + px * ca - py * sa, y + px * sa + py * ca)
    blade = [P(-h * 0.035, -h * 0.25), P(h * 0.035, -h * 0.25), P(0, h * 0.75)]
    fx.poly(blade, tuple(int(255 * k) for _ in range(3)), c)
    fx.line([P(-h * 0.12, -h * 0.25), P(h * 0.12, -h * 0.25)], c, w=2, gw=10)
    fx.line([P(0, -h * 0.25), P(0, -h * 0.42)], c, w=3, gw=8)


RNG = np.random.default_rng(7)
EMB = RNG.random((260, 5))   # x, y, speed, phase, size


def embers(fx, t, col=(255, 190, 110), n=160, k=1.0, up=True, speed=1.0):
    for i in range(n):
        x, y, s, ph, sz = EMB[i]
        yy = (y * H - (1 if up else -1) * (40 + 140 * s) * speed * t) % H
        xx = x * W + 25 * math.sin(t * (0.6 + s) + ph * 6)
        a = 0.5 + 0.5 * math.sin(t * 3 + ph * 9)
        c = tuple(int(v * a * k) for v in col)
        fx.dot(xx, yy, 0.6 + sz * 1.6, c)


def speed_lines(fx, t, k=1.0, cx=W / 2, cy=H / 2, n=46, col=(200, 215, 255)):
    rnd = random.Random(int(t * 24))
    for _ in range(n):
        a = rnd.uniform(0, 2 * math.pi)
        r0 = rnd.uniform(260, 420)
        r1 = r0 + rnd.uniform(200, 600)
        c = tuple(int(v * k * rnd.uniform(0.3, 0.8)) for v in col)
        fx.line([(cx + math.cos(a) * r0, cy + math.sin(a) * r0), (cx + math.cos(a) * r1, cy + math.sin(a) * r1)],
                c, w=1, gw=3)


def hex_shield(fx, cx, cy, r, t, k):
    c = (int(180 * k), int(150 * k), int(255 * k))
    s = 46
    for i in range(-8, 9):
        for j in range(-8, 9):
            hx = cx + (i + (j % 2) * 0.5) * s * 1.73
            hy = cy + j * s * 1.5
            if math.hypot(hx - cx, hy - cy) > r:
                continue
            pul = 0.5 + 0.5 * math.sin(t * 5 - math.hypot(hx - cx, hy - cy) / 60)
            cc = tuple(int(v * pul) for v in c)
            pts = [(hx + math.cos(a) * s, hy + math.sin(a) * s) for a in
                   [math.pi / 6 + m * math.pi / 3 for m in range(7)]]
            fx.line(pts, cc, w=1, gw=4)
    ring(fx, cx, cy, r, c, w=3, sq=1.0, gw=18)


# ======================================================================
# post / text
# ======================================================================
def bloom(f, k=0.42, thr=0.66):
    small = to_img(f).resize((W // 4, H // 4), Image.BILINEAR)
    s = np.asarray(small, np.float32) / 255.0
    b = np.clip(s - thr, 0, 1) / (1 - thr)
    b = to_img(b).filter(ImageFilter.GaussianBlur(7)).resize((W, H), Image.BILINEAR)
    return f + np.asarray(b, np.float32) / 255.0 * k


GRAIN = (np.random.default_rng(1).standard_normal((H // 2, W // 2)).astype(np.float32))


def grain(f, t, k=0.022):
    ox, oy = int(t * 997) % (W // 2), int(t * 571) % (H // 2)
    g = np.roll(np.roll(GRAIN, ox, 1), oy, 0)
    g = np.repeat(np.repeat(g, 2, 0), 2, 1)
    return f + g[..., None] * k


def _vig():
    y, x = np.mgrid[0:H, 0:W]
    d = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2)
    return (1 - np.clip((d - 0.65) * 0.8, 0, 0.55))[..., None].astype(np.float32)


VIG = _vig()


def text_rgba(lines, font, col, cy, alpha, cx=W // 2, spacing=10, glow=None, stroke=2):
    ov = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(ov)
    lh = font.size + spacing
    y = cy - lh * len(lines) / 2
    pos = []
    for ln in lines:
        x = cx - d.textlength(ln, font=font) / 2
        pos.append((x, y, ln))
        y += lh
    if glow:
        gl = Image.new("RGBA", (W, H))
        gd = ImageDraw.Draw(gl)
        for x, y, ln in pos:
            gd.text((x, y), ln, font=font, fill=glow + (int(255 * alpha),))
        ov.alpha_composite(gl.filter(ImageFilter.GaussianBlur(10)))
        ov.alpha_composite(gl.filter(ImageFilter.GaussianBlur(4)))
    for x, y, ln in pos:
        d.text((x, y), ln, font=font, fill=col + (int(255 * alpha),), stroke_width=stroke,
               stroke_fill=(0, 0, 0, int(170 * alpha)))
    return ov


def wrap(text, font, maxw):
    out, cur = [], ""
    for w_ in text.split():
        t_ = (cur + " " + w_).strip()
        if font.getlength(t_) <= maxw:
            cur = t_
        else:
            out.append(cur)
            cur = w_
    out.append(cur)
    return out


def comp(f, ov):
    a = np.asarray(ov, np.float32) / 255.0
    return f * (1 - a[..., 3:]) + a[..., :3] * a[..., 3:]


SPEAKER = {"LIANG CHEN": ("梁辰", (150, 200, 255)), "LONG TIAN": ("龙天", (255, 120, 110)),
           "NARATOR": ("", (230, 210, 160))}


def subtitle(f, t, a, b, text, who):
    al = fade(t, a, b, 0.22)
    if al <= 0:
        return f
    cjk, col = SPEAKER[who]
    lines = wrap(text, F_TEXT if who != "NARATOR" else F_NARR, 580)
    font = F_TEXT if who != "NARATOR" else F_NARR
    lh = font.size + 10
    bh = lh * len(lines) + 56
    ov = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(ov)
    top = SUB_BOTTOM - bh
    d.rounded_rectangle((34, top, W - 34, SUB_BOTTOM), 18, fill=(6, 8, 20, int(175 * al)),
                        outline=col + (int(120 * al),), width=2)
    d.text((60, top + 12), who, font=F_NAME, fill=col + (int(255 * al),))
    if cjk:
        d.text((60 + d.textlength(who, font=F_NAME) + 10, top + 11), cjk, font=F_CJK_S,
               fill=col + (int(220 * al),))
    f = comp(f, ov)
    return comp(f, text_rgba(lines, font, (250, 246, 236), top + 44 + lh * len(lines) / 2 - 2, al, spacing=10))


def callout(f, t, a, b, cjk, latin, col=(255, 214, 140), glow=(255, 150, 60), y=300):
    if not (a <= t <= b):
        return f
    al = fade(t, a, b, 0.15)
    p = eout((t - a) / 0.35)
    sc = lerp(1.6, 1.0, p)
    ov = text_rgba([cjk], F_CJK_XL, col, y, al, glow=glow, stroke=3)
    ov.alpha_composite(text_rgba([latin], F_CALL, (255, 250, 240), y + 110, al, glow=glow))
    if sc > 1.001:
        big = ov.resize((int(W * sc), int(H * sc)), Image.BILINEAR)
        ox, oy = (big.width - W) // 2, int(y * sc - y)
        ov = big.crop((ox, oy, ox + W, oy + H))
    return comp(f, ov)


# ======================================================================
# shots — each returns float frame; t = global time, lt = local
# ======================================================================
def sh_open(lt, t):
    d = 9.0
    sx, sy = shake(t, 1.5)
    img = full(ARENA, lt, d, (820, 230), (900, 470), 1.55, 1.30, (sx, sy))
    f = to_f(img)
    fx = FX()
    for k_, (sd, x0, x1) in enumerate(((1, 120, 300), (2, 520, 640), (3, 300, 80))):
        if lightning(fx, t + k_ * 0.31, sd, (x0, -20), (x1, 520), every=1.7 + k_ * 0.4) > 0.6:
            f = f * 1.12
    embers(fx, t, (150, 170, 255), n=90, k=0.6)
    f = fx.apply(f)
    al = fade(lt, 1.2, 8.8, 0.8)
    f = comp(f, text_rgba(["天神殿"], F_CJK_L, (255, 225, 160), 300, al, glow=(255, 160, 60), stroke=3))
    f = comp(f, text_rgba(["TIAN SHEN DIAN"], F_CALL, (255, 245, 230), 400, al, glow=(120, 140, 255)))
    f = comp(f, text_rgba(["Istana Dewa di Atas Awan"], F_SUBT, (220, 220, 245), 455, fade(lt, 2.0, 8.8, 0.8)))
    return f


def sh_title(lt, t):
    img = panel("hero", HERO, lt, 7.0, band_h=640, cy=620, pan=(0.25, 0.0), zoom=(1.0, 1.1))
    f = to_f(img)
    fx = FX()
    lightning(fx, t, 11, (60, 420), (700, 760), col=(140, 170, 255), every=1.1)
    embers(fx, t, (160, 190, 255), n=120, k=0.7)
    p = eout(lt / 1.2)
    fx.line([(lerp(-200, 60, p), 900), (lerp(-200, 760, p), 880)], (120, 170, 255), w=2, gw=18)
    f = fx.apply(f)
    al = fade(lt, 0.6, 6.8, 0.6)
    f = comp(f, text_rgba(["梁辰  ·  龙天"], F_CJK_S, (255, 220, 150), 170, al))
    f = comp(f, text_rgba(["LIANG CHEN", "vs LONG TIAN"], F_TITLE, (255, 240, 220), 250, al,
                          glow=(90, 120, 255), spacing=0))
    f = comp(f, text_rgba(["Episode 1 — Kebangkitan Naga Kuno"], F_SUBT, (255, 214, 150), 1000,
                          fade(lt, 1.5, 6.8, 0.6), glow=(255, 140, 60)))
    return f


def sh_legend(lt, t):
    img = panel("moon", dtile(4, 5), lt, 5.0, band_h=520, cy=640, pan=(0.6, 0.0), zoom=(1.05, 1.15))
    f = to_f(img)
    fx = FX()
    embers(fx, t, (200, 210, 255), n=70, k=0.5)
    f = fx.apply(f)
    f = subtitle(f, lt, 0.3, 4.9, "Sepuluh ribu tahun lalu, naga kuno Long Tian disegel di bawah Istana Dewa.",
                 "NARATOR")
    return f


def stage_view(lt, t, dur, c0, c1, z0, z1, sh=(0, 0)):
    return to_f(full(STAGE, lt, dur, c0, c1, z0, z1, sh))


def put_sprite(f, sp, x, y, h, alpha=1.0, flip=False, tint=None):
    s = h / sp.height
    im = sp.resize((max(1, int(sp.width * s)), int(h)), Image.LANCZOS)
    if s > 1.3:
        im = im.filter(ImageFilter.UnsharpMask(1.5, 60, 2))
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    a = np.asarray(im, np.float32) / 255.0
    rgb, al = a[..., :3], a[..., 3:] * alpha
    if tint is not None:
        rgb = rgb * np.array(tint, np.float32)
    return _blit(f, rgb, al, int(x - im.width / 2), int(y - im.height))
    reg = f[max(0, y0):y1, max(0, x0):x1]
    sa = al[ys:ys + reg.shape[0], xs:xs + reg.shape[1]]
    sr = rgb[ys:ys + reg.shape[0], xs:xs + reg.shape[1]]
    f[max(0, y0):y1, max(0, x0):x1] = reg * (1 - sa) + sr * sa
    return f


def _blit(f, rgb, al, x0, y0, mode="over"):
    h_, w_ = al.shape[:2]
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(W, x0 + w_), min(H, y0 + h_)
    if dx1 <= dx0 or dy1 <= dy0:
        return f
    sx0, sy0 = dx0 - x0, dy0 - y0
    sa = al[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]
    sr = rgb[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]
    reg = f[dy0:dy1, dx0:dx1]
    top = np.maximum(reg, sr) if mode == "lighten" else sr
    f[dy0:dy1, dx0:dx1] = reg * (1 - sa) + top * sa
    return f


def put_dragon(f, x, y, h, alpha=1.0):
    s = h / DRAGON_KEY.height
    im = DRAGON_KEY.resize((int(DRAGON_KEY.width * s), int(h)), Image.BILINEAR)
    a = np.asarray(im, np.float32) / 255.0
    return _blit(f, a[..., :3], a[..., 3:] * alpha, int(x - im.width / 2), int(y - im.height / 2), "lighten")


def sh_seal(lt, t):
    """segel retak — floor cracks with light"""
    f = stage_view(lt, t, 4.0, (500, 700), (500, 860), 0.95, 1.1, shake(t, 3 + 6 * ease(lt / 4)))
    fx = FX()
    cy = 1050
    for i in range(7):
        a = i / 7 * 2 * math.pi + 0.3
        L = 420 * ease((lt - 0.4 * i / 7) / 2.5)
        lightning(fx, t, 30 + i, (360, cy), (360 + math.cos(a) * L, cy + math.sin(a) * L * 0.35),
                  col=(255, 180, 90), every=0.3, life=0.3, branches=1)
    magic_circle(fx, 360, cy, 300, t, k=0.4 + 0.6 * ease(lt / 3))
    embers(fx, t, n=120, k=0.8)
    f = fx.apply(f)
    f = subtitle(f, lt, 0.3, 3.9, "Malam ini, segel itu mulai retak.", "NARATOR")
    return f


WALK = ["walk0", "walk1", "walk2", "walk1"]


def sh_arrive(lt, t):
    d = 4.5
    f = stage_view(lt, t, d, (430, 820), (560, 850), 1.0, 1.08)
    fx = FX()
    magic_circle(fx, 360, 1060, 300, t, k=0.35)
    embers(fx, t, (150, 180, 255), n=80, k=0.5)
    f = fx.apply(f)
    p = lt / d
    x = lerp(180, 420, eio(p))
    pose = LP[WALK[int(lt * 3.2) % 4]] if p < 0.9 else LP["front"]
    bob = -abs(math.sin(lt * 3.2 * math.pi / 2)) * 6 if p < 0.9 else 0
    f = put_sprite(f, pose, x, 1150 + bob, 520)
    f = subtitle(f, lt, 0.3, 4.4, "Liang Chen, Penguasa Pedang Dewa, datang seorang diri.", "NARATOR")
    return f


def sh_awaken(lt, t):
    d = 2.5
    k = ease(lt / 0.6)
    sx, sy = shake(t, 4 + 12 * fade(lt, 0.3, 2.5, 0.3))
    img = full(DRAGON_FULL, lt, d, (414, 900), (414, 380), 1.05, 1.15, (sx, sy))
    f = to_f(img) * (0.25 + 0.75 * k)
    fx = FX()
    for i in range(3):
        lightning(fx, t + i * 0.2, 40 + i, (100 + i * 260, -10), (300 + i * 80, 700), every=0.7 + i * 0.2)
    f = fx.apply(f)
    return f


def sh_dialog_dragon(src_kind, lt, t, dur, text, a=0.1, b=None, close=False):
    sx, sy = shake(t, 2)
    if src_kind == "head":
        img = full(DRAGON_HEAD, lt, dur, (380, 560), (420, 520), 1.0, 1.12, (sx, sy))
    elif src_kind == "full":
        img = full(DRAGON_FULL, lt, dur, (430, 500), (400, 420), 1.15, 1.3, (sx, sy))
    else:
        img = panel(src_kind, dtile(*src_kind), lt, dur, band_h=560, cy=560, pan=(0.3, 0), zoom=(1.0, 1.12),
                    sh=(sx, sy))
    f = to_f(img)
    fx = FX()
    lightning(fx, t, 77, (60, -10), (220, 600), every=1.3)
    lightning(fx, t + 0.6, 78, (680, -10), (520, 520), every=1.6)
    embers(fx, t, (180, 160, 255), n=80, k=0.5)
    f = fx.apply(f)
    return subtitle(f, lt, a, b or dur - 0.1, text, "LONG TIAN")


def sh_dialog_liang(kind, lt, t, dur, text, aura=0.0):
    if kind == "portrait":
        img = full(LIANG_PORTRAIT, lt, dur, (420, 520), (430, 470), 1.0, 1.1)
    else:
        img = panel("hero2", HERO, lt, dur, band_h=640, cy=560, pan=(-0.25, 0), zoom=(1.05, 1.15))
    f = to_f(img)
    fx = FX()
    embers(fx, t, (150, 190, 255), n=100, k=0.5 + aura)
    if aura > 0:
        for i in range(3):
            lightning(fx, t + i * 0.13, 90 + i, (100 + i * 250, 1100), (200 + i * 180, 200), col=(110, 160, 255),
                      every=0.45, life=0.2)
    f = fx.apply(f)
    return subtitle(f, lt, 0.1, dur - 0.1, text, "LIANG CHEN")


# ---------------------------------------------------------------- battle beats
def b_tile(key, src, lt, dur, fxfn=None, band=520, cy=620, pan=(0.4, 0), zoom=(1.0, 1.08), shk=4, rot=0.0):
    sx, sy = shake(lt + hash(str(key)) % 7, shk)
    img = panel(key, src, lt, dur, band_h=band, cy=cy, pan=pan, zoom=zoom, sh=(sx, sy), rot=rot)
    f = to_f(img)
    if fxfn:
        fx = FX()
        fxfn(fx, lt)
        f = fx.apply(f)
    return f


def battle(lt, t):
    """60-second fight, lt in [0,60)"""
    B = BEATS
    for i, (a, b, fn) in enumerate(B):
        if a <= lt < b:
            return fn(lt - a, t, b - a)
    return np.zeros((H, W, 3), np.float32)


def bt_draw(l, t, d):   # Tian Shen Jian!
    def fx_(fx, l):
        embers(fx, t, (150, 190, 255), n=160, k=1.0, speed=2.0)
        lightning(fx, t, 201, (40, 700), (680, 560), col=(130, 170, 255), every=0.4)
        ring(fx, 360, 640, 600 * eout(l / 0.8), (120, 170, 255), sq=1.0, gw=20)
    f = b_tile("s00", stile(0, 0), l, d, fx_, pan=(0.3, 0), zoom=(1.12, 1.0))
    return callout(f, l, 0.25, d, "天神剑", "TIAN SHEN JIAN!", col=(200, 225, 255), glow=(80, 130, 255))


def bt_roar(l, t, d):
    def fx_(fx, l):
        for i in range(4):
            lightning(fx, t + i * 0.11, 210 + i, (360, 600), (lerp(-100, 820, i / 3), lerp(1300, -100, i % 2)),
                      col=(190, 140, 255), every=0.35)
        ring(fx, 520, 520, 900 * ((l * 0.9) % 1), (200, 150, 255), sq=1.0, gw=10)
    return b_tile("d05", dtile(0, 5), l, d, fx_, pan=(-0.2, 0), zoom=(1.0, 1.1), shk=14)


def bt_dive(l, t, d):
    def fx_(fx, l):
        speed_lines(fx, t, 1.0)
        embers(fx, t, (255, 200, 120), n=120, k=1.0, speed=4)
    return b_tile("d12", dtile(1, 2), l, d, fx_, pan=(1.0, 0), zoom=(1.0, 1.1), shk=8)


def bt_dodge(l, t, d):
    f = stage_view(l, t, d, (300, 830), (700, 830), 1.05, 1.05, shake(t, 4))
    fx = FX()
    speed_lines(fx, t, 0.7, cx=360, cy=900)
    # dragon swoops overhead
    f = put_dragon(f, lerp(900, -300, eio(l / d)), lerp(280, 360, l / d), 900, alpha=0.9)
    x = lerp(80, 640, eio(l / d))
    for k in range(5, 0, -1):  # afterimages
        xk = lerp(80, 640, eio(max(0, l - k * 0.05) / d))
        f = put_sprite(f, LP["run"], xk, 1160, 440, alpha=0.12 * (6 - k), tint=(0.6, 0.75, 1.2))
    f = put_sprite(f, LP["run"], x, 1160, 440)
    slash(fx, x + 40, 900, 260, 140, -2.6, 0.4, (l - 1.6) / 0.35, fade(l, 1.6, d, 0.2))
    f = fx.apply(f)
    return f


def bt_clash(l, t, d):
    sx, sy = shake(t, 10 * fade(l, 0.0, 1.0, 0.1) + 3)
    img = full(ARENA, l, d, (700, 700), (760, 650), 1.5, 1.75, (sx, sy))
    f = to_f(img)
    fx = FX()
    slash(fx, 470, 760, 320, 160, -2.8, 0.6, l / 0.3, fade(l, 0, 1.6, 0.2))
    slash(fx, 420, 700, 300, 200, 0.4, 3.4, (l - 1.0) / 0.3, fade(l, 1.0, 2.6, 0.2), col=(255, 210, 140))
    for k in range(40):  # sparks
        rnd = random.Random(k)
        a = rnd.uniform(0, 6.28)
        r = (l % 1.0) * rnd.uniform(200, 600)
        fx.dot(520 + math.cos(a) * r, 700 + math.sin(a) * r, 2, (255, 220, 160))
    return fx.apply(f)


def bt_quick(l, t, d):
    def fx_(fx, l):
        for k in range(3):
            st = 0.25 + k * 0.7
            slash(fx, 360, 600 + k * 40, 330, 120, -2.9, 0.5, (l - st) / 0.25, fade(l, st, st + 1.0, 0.15),
                  rot=(-0.4 + k * 0.4))
    f = b_tile("s01", stile(0, 1), l, d, fx_, pan=(-0.6, 0), zoom=(1.0, 1.1), shk=6)
    return callout(f, l, 0.1, d, "疾风斩", "TEBASAN CEPAT", col=(200, 225, 255), glow=(80, 130, 255))


def bt_breath(l, t, d):
    def fx_(fx, l):
        for i in range(7):
            lightning(fx, t + i * 0.07, 300 + i, (680, 500), (lerp(-80, 300, i / 6), lerp(300, 1300, i / 6)),
                      col=(200, 150, 255), every=0.25, life=0.22)
        fx.line([(720, 520), (lerp(720, -100, eout(l / 0.6)), 620)], (220, 180, 255), w=10, gw=60)
    f = b_tile("d25", dtile(2, 5), l, d, fx_, pan=(0.3, 0), zoom=(1.0, 1.1), shk=12)
    return callout(f, l, 0.2, d, "灭天雷", "PETIR PENGHANCUR LANGIT", col=(230, 200, 255), glow=(150, 60, 255), y=260)


def bt_aura(l, t, d):
    def fx_(fx, l):
        magic_circle(fx, 360, 860, 330, t * 1.5, k=0.7 * ease(l / 0.6))
        magic_circle(fx, 360, 520, 200, -t, k=ease(l / 0.8) * 0.45, sq=1.0)
        embers(fx, t, n=200, k=1.0, speed=3)
    f = b_tile("s11", stile(1, 1), l, d, fx_, pan=(0, 0.2), zoom=(1.0, 1.1))
    return callout(f, l, 0.1, d, "神之气", "AURA DEWA!", glow=(255, 150, 40))


def bt_spiral(l, t, d):
    def fx_(fx, l):
        for k in range(4):
            a0 = t * 7 + k * 1.57
            slash(fx, 360, 640, 260 + k * 40, 120 + k * 20, a0, a0 + 2.2, 1, 0.9, width=18)
    f = b_tile("s02", stile(0, 2), l, d, fx_, zoom=(1.0, 1.1), rot=0.0)
    return callout(f, l, 0.1, d, "螺旋剑", "PEDANG SPIRAL", col=(200, 225, 255), glow=(80, 130, 255))


def bt_orb(l, t, d):
    def fx_(fx, l):
        r = 30 + 170 * ease(l / (d - 0.6))
        fx.ellipse((360 - r, 600 - r, 360 + r, 600 + r), (255, 210, 120), w=3, gw=18,
                   fill=(int(90 * min(1, l / 2)), int(65 * min(1, l / 2)), 20))
        for i in range(5):
            lightning(fx, t + i * 0.1, 400 + i, (360, 600), (360 + math.cos(i * 1.3) * r * 2.2,
                                                           600 + math.sin(i * 1.3) * r * 2.2),
                      col=(255, 200, 120), every=0.2, life=0.18)
    f = b_tile("d24", dtile(2, 4), l, d, fx_, zoom=(1.0, 1.15), shk=4 + 10 * l / d)
    f = callout(f, l, 0.1, d - 0.5, "金龙珠", "MUTIARA NAGA EMAS", glow=(255, 160, 40), y=250)
    ex = max(0, (l - (d - 0.5)) / 0.5)
    return f * (1 - ex) + ex * 1.6


def bt_hurt(l, t, d):
    f = stage_view(l, t, d, (520, 900), (500, 920), 1.15, 1.2, shake(t, 5 * (1 - l / d)))
    fx = FX()
    for k in range(60):  # debris
        rnd = random.Random(k + 900)
        x = rnd.uniform(0, W)
        y = (rnd.uniform(0, H) + 300 * l * rnd.uniform(0.3, 1)) % H
        fx.dot(x, y, rnd.uniform(1, 3), (120, 110, 100))
    embers(fx, t, (255, 170, 90), n=100, k=0.7)
    f = fx.apply(f)
    f = put_sprite(f, LP["kneel"], 380, 1160, 430)
    return subtitle(f, l, 0.1, d - 0.05, "Hanya itu kekuatanmu, manusia?", "LONG TIAN")


def bt_rise(l, t, d):
    f = stage_view(l, t, d, (500, 760), (500, 700), 1.1, 1.3)
    fx = FX()
    k = ease(l / 1.0)
    for i in range(5):
        lightning(fx, t + i * 0.09, 500 + i, (360 + (i - 2) * 50, 1180), (360 + (i - 2) * 120, 300),
                  col=(110, 170, 255), every=0.3, life=0.2)
    ring(fx, 360, 1150, 320 * k, (120, 180, 255), gw=24)
    embers(fx, t, (150, 200, 255), n=200, k=1.2, speed=3)
    f = fx.apply(f, 0.6 + 0.4 * k)
    f = put_sprite(f, LP["lookup"], 360, 1170, 560)
    return subtitle(f, l, 0.05, d - 0.05, "Aku... belum kalah!", "LIANG CHEN")


def bt_illusion(l, t, d):
    def fx_(fx, l):
        for k in range(26):
            rnd = random.Random(k + 600)
            st = rnd.uniform(0, d - 0.6)
            p = (l - st) / 0.45
            if p < 0:
                continue
            x = rnd.uniform(40, 680)
            y = lerp(-300, rnd.uniform(500, 1000), eout(min(1, p)))
            kk = 1.0 if p < 1 else max(0, 1 - (p - 1) * 0.6)
            light_sword(fx, x, y, rnd.uniform(220, 380), k=kk, ang=rnd.uniform(-0.15, 0.15))
            if 1 <= p < 1.3:
                ring(fx, x, y + 200, 120 * (p - 1) / 0.3, (170, 210, 255), gw=10)
    f = b_tile("s30", stile(3, 0), l, d, fx_, zoom=(1.0, 1.1), shk=5)
    return callout(f, l, 0.1, d, "幻影剑", "PEDANG ILUSI!", col=(200, 225, 255), glow=(80, 130, 255))


def bt_shield(l, t, d):
    def fx_(fx, l):
        hex_shield(fx, 380, 600, 330, t, ease(l / 0.4))
        for k in range(8):
            rnd = random.Random(k + 700)
            st = 0.3 + k * 0.33
            p = (l - st) / 0.25
            if 0 <= p <= 1.5:
                a = rnd.uniform(0, 6.28)
                x = 380 + math.cos(a) * lerp(700, 330, min(1, p))
                y = 600 + math.sin(a) * lerp(700, 330, min(1, p))
                light_sword(fx, x, y, 200, k=1.0 if p < 1 else 1.5 - p, ang=a + math.pi / 2)
                if p >= 1:
                    ring(fx, x, y, 90 * (p - 1) * 2, (255, 230, 200), sq=1.0)
    f = b_tile("d23", dtile(2, 3), l, d, fx_, zoom=(1.0, 1.15), shk=7)
    return callout(f, l, 0.1, d, "永恒鳞盾", "PERISAI SISIK ABADI", col=(230, 200, 255), glow=(150, 60, 255), y=250)


def bt_skydragon(l, t, d):
    def fx_(fx, l):
        embers(fx, t, n=220, k=1.3, speed=4)
        for k in range(3):
            a0 = t * 4 + k * 2.1
            slash(fx, 360, 620, 330, 200, a0, a0 + 2.5, 1, 0.8, col=(255, 200, 110), width=20)
    f = b_tile("s10", stile(1, 0), l, d, fx_, pan=(0.5, 0), zoom=(1.0, 1.12), shk=6)
    return callout(f, l, 0.1, d, "天龙剑", "PEDANG NAGA LANGIT!", glow=(255, 150, 40))


def bt_collide(l, t, d):
    def fx_(fx, l):
        speed_lines(fx, t, 1.2, col=(255, 210, 150))
        r = 900 * eout(max(0, l - 1.6) / 0.8)
        ring(fx, 360, 640, r, (255, 230, 180), sq=1.0, gw=40, w=8)
    f = b_tile("d35", dtile(3, 5), l, d, fx_, pan=(-1.0, 0), zoom=(1.0, 1.15), shk=10 + 12 * fade(l, 1.6, d, 0.1))
    fl = 0.0 if l < 1.6 else math.exp(-(l - 1.6) * 5)
    return f * (1 - fl) + fl * 1.4


def bt_final_charge(l, t, d):
    def fx_(fx, l):
        k = ease(l / d)
        magic_circle(fx, 360, 1050, 340, t * 2, k=0.6 + 0.4 * k)
        for i in range(70):  # converging particles
            rnd = random.Random(i + 800)
            a = rnd.uniform(0, 6.28)
            ph = (l * rnd.uniform(0.6, 1.4) + rnd.random()) % 1
            r = (1 - ph) * rnd.uniform(300, 700)
            fx.dot(360 + math.cos(a) * r, 380 + math.sin(a) * r, 2.5, (255, 220, 150))
        lightning(fx, t, 820, (360, -20), (360, 700), col=(255, 220, 160), every=0.3, life=0.25)
    f = b_tile("s43", stile(4, 3), l, d, fx_, pan=(0, 0.5), zoom=(1.0, 1.15), shk=3 + 6 * l / d)
    f = callout(f, l, 0.2, d, "神剑天诛", "PEDANG DEWA AGUNG!", glow=(255, 150, 40), y=230)
    return subtitle(f, l, 0.6, d - 0.05, "Pedang Dewa Agung... hancurkan langit!", "LIANG CHEN")


def bt_final_strike(l, t, d):
    sx, sy = shake(t, 4 + 26 * fade(l, 0.3, 3.0, 0.1))
    img = full(DRAGON_FULL, l, d, (414, 600), (414, 560), 1.2, 1.45, (sx, sy))
    f = to_f(img)
    fx = FX()
    p = eout(l / 0.45)
    bw = 140 * fade(l, 0.0, d - 0.6, 0.3)
    if bw > 0:
        fx.poly([(360 - bw / 2, -50), (360 + bw / 2, -50), (360 + bw / 2, lerp(-50, 1300, p)),
                 (360 - bw / 2, lerp(-50, 1300, p))], (255, 250, 230), (255, 210, 130))
    light_sword(fx, 360, lerp(-900, 500, p), 900, col=(255, 220, 150))
    for k in range(3):
        r = 1000 * eout(max(0, l - 0.45 - k * 0.25) / 1.0)
        ring(fx, 360, 640, r, (255, 235, 200), sq=1.0, gw=30, w=6)
    for i in range(6):
        lightning(fx, t + i * 0.05, 900 + i, (360, 640), (360 + math.cos(i) * 700, 640 + math.sin(i) * 900),
                  col=(255, 220, 160), every=0.2, life=0.18)
    f = fx.apply(f)
    wf = 0.0 if l < 0.42 else (ease((l - 0.42) / 0.1) if l < 0.52 else math.exp(-(l - 0.52) * 3.2))
    return f * (1 - wf) + wf * 1.5


BEATS = []


def _build_beats():
    seq = [(3.0, bt_draw), (3.0, bt_roar), (3.0, bt_dive), (3.5, bt_dodge), (3.0, bt_clash), (3.0, bt_quick),
           (3.5, bt_breath), (3.0, bt_aura), (3.0, bt_spiral), (3.5, bt_orb), (3.0, bt_hurt), (3.0, bt_rise),
           (3.5, bt_illusion), (3.0, bt_shield), (3.5, bt_skydragon), (2.5, bt_collide), (5.0, bt_final_charge),
           (5.0, bt_final_strike)]
    a = 0.0
    for d, fn in seq:
        BEATS.append((a, a + d, fn))
        a += d
    assert abs(a - 60.0) < 1e-6, a


_build_beats()


def sh_retreat(lt, t):
    f = b_tile("d45", dtile(4, 5), lt, 4.8, None, pan=(0.8, 0), zoom=(1.0, 1.2), shk=2)
    return subtitle(f, lt, 0.1, 4.7, "Mustahil... pedang manusia melukai sisikku?!", "LONG TIAN")


def sh_echo(lt, t):
    img = full(ARENA, lt, 7.4, (760, 260), (760, 330), 1.4, 1.55)
    f = to_f(img) * 0.9
    fx = FX()
    lightning(fx, t, 950, (800, -10), (600, 500), every=2.1)
    embers(fx, t, (200, 200, 255), n=90, k=0.6)
    f = fx.apply(f)
    return subtitle(f, lt, 0.1, 7.3, "Ingat namaku, Liang Chen... Long Tian akan kembali, dan langit akan runtuh "
                                     "bersamaku.", "LONG TIAN")


def sh_after(lt, t):
    d = 5.6
    f = stage_view(lt, t, d, (500, 820), (500, 760), 1.05, 1.15)
    fx = FX()
    magic_circle(fx, 360, 1060, 300, t * 0.4, k=0.3)
    embers(fx, t, n=140, k=0.8, speed=0.6)
    f = fx.apply(f)
    f = put_sprite(f, LP["back"], 360, 1180, 600)
    return subtitle(f, lt, 0.3, 5.4, "Naga kuno telah bangkit. Dan pertarungan ini... baru permulaan.", "NARATOR")


def sh_end(lt, t):
    img = panel("hero_end", HERO, lt, 8.0, band_h=640, cy=600, pan=(0.2, 0), zoom=(1.12, 1.0))
    f = to_f(img) * 0.75
    fx = FX()
    embers(fx, t, n=160, k=0.8)
    f = fx.apply(f)
    al = fade(lt, 0.4, 7.9, 0.7)
    f = comp(f, text_rgba(["天神剑"], F_CJK_L, (255, 225, 160), 210, al, glow=(255, 150, 50), stroke=3))
    f = comp(f, text_rgba(["LIANG CHEN vs LONG TIAN"], F_CALL, (255, 245, 230), 310, al, glow=(90, 120, 255)))
    f = comp(f, text_rgba(wrap("“Pertarungan antara Dewa Pedang dan Naga Kuno akan mengguncang langit!”",
                               F_SUBT, 600), F_SUBT, (255, 236, 210), 960, fade(lt, 1.2, 7.9, 0.7)))
    f = comp(f, text_rgba(["Bersambung…"], F_CALL, (255, 214, 150), 1060, fade(lt, 2.4, 7.9, 0.7),
                          glow=(255, 140, 60)))
    return f


# timeline (start, end, fn) — dialogue starts 32.0, battle 63.3–123.3
DIALOG = [  # (start, dur, who, clip, text, kind)
    (32.0, 4.6, "D", "d_00", "Sepuluh ribu tahun aku tertidur... dan manusia berani menginjak istanaku?", "head"),
    (36.6, 4.7, "L", "l_00", "Tian Shen Dian bukan milikmu, Long Tian. Langit ini milik semua makhluk.", "portrait"),
    (41.3, 5.9, "D", "d_01", "Kau hanya debu di bawah sisikku. Pedangmu tak akan mampu melukaiku.", (2, 1)),
    (47.2, 3.9, "L", "l_01", "Kalau begitu, biarkan pedang ini yang menjawab.", "hero"),
    (51.1, 8.3, "D", "d_02", "Bodoh! Rasakan petir penghancur langit... akan kuhapus kerajaanmu dari dunia ini!",
     "full"),
    (59.4, 3.9, "L", "l_02", "Selama aku berdiri, tak satu pun rakyatku akan kau sentuh.", "portrait"),
]


def make_dialog(start, dur, who, text, kind):
    def fn(lt, t):
        if who == "D":
            return sh_dialog_dragon(kind, lt, t, dur, text)
        return sh_dialog_liang(kind, lt, t, dur, text, aura=0.6 if start > 59 else 0.0)
    return (start, start + dur, fn)


SHOTS = [(0.0, 9.0, sh_open), (9.0, 16.0, sh_title), (16.0, 21.0, sh_legend), (21.0, 25.0, sh_seal),
         (25.0, 29.5, sh_arrive), (29.5, 32.0, sh_awaken)]
SHOTS += [make_dialog(s, d, w, x, k) for s, d, w, c, x, k in DIALOG]
SHOTS += [(63.3, 123.3, lambda lt, t: battle(lt, t)), (123.3, 128.1, sh_retreat), (128.1, 135.5, sh_echo),
          (135.5, 141.1, sh_after), (141.1, 149.1, sh_end)]
DUR = SHOTS[-1][1]
DISSOLVE = {9.0, 16.0, 21.0, 25.0, 123.3, 128.1, 135.5, 141.1}


def render_shot(t):
    for a, b, fn in SHOTS:
        if a <= t < b:
            return fn(t - a, t), a, b
    return np.zeros((H, W, 3), np.float32), 0, DUR


def frame(t):
    f, a, b = render_shot(t)
    # dissolves on calm cuts
    if a in DISSOLVE and t - a < 0.35 and a > 0:
        prev, _, _ = render_shot(a - 1e-3)
        k = ease((t - a) / 0.35)
        f = prev * (1 - k) + f * k
    f = bloom(f)
    f = f * VIG
    f = grain(f, t)
    # battle flash on beat changes
    if 63.3 <= t < 123.3:
        lt = t - 63.3
        for s, e, _ in BEATS:
            if 0 <= lt - s < 0.12 and s > 0:
                f = f + (1 - (lt - s) / 0.12) * 0.55
    k = min(ease(t / 1.2), ease((DUR - t) / 1.5))
    return (np.clip(f * k, 0, 1) * 255).astype(np.uint8)


# ======================================================================
# audio
# ======================================================================
SR = 44100


def load(name):
    with wave.open(os.path.join(A, "audio", name + ".wav")) as wf:
        sr, n, ch, sw = wf.getframerate(), wf.getnframes(), wf.getnchannels(), wf.getsampwidth()
        x = np.frombuffer(wf.readframes(n), dtype=np.int16 if sw == 2 else np.uint8).astype(np.float32)
    if sw == 1:
        x = (x - 128) * 256
    x = x.reshape(-1, ch).mean(1) / 32768.0
    if sr != SR:
        x = np.interp(np.arange(int(len(x) * SR / sr)) * sr / SR, np.arange(len(x)), x)
    return x.astype(np.float32)


def reverb(x, taps=((0.045, 0.5), (0.09, 0.35), (0.17, 0.25), (0.31, 0.15))):
    out = x.copy()
    for d, g in taps:
        k = int(d * SR)
        out[k:] += x[:-k] * g
    return out


def ks(freq, dur, vol=0.2, damp=0.996):
    """Karplus-Strong pluck (guzheng-like)"""
    n = int(dur * SR)
    p = max(2, int(SR / freq))
    buf = np.random.default_rng(int(freq)).uniform(-1, 1, p).astype(np.float32)
    out = np.empty(n, np.float32)
    for i in range(n):
        out[i] = buf[i % p]
        buf[i % p] = damp * 0.5 * (buf[i % p] + buf[(i + 1) % p])
    return out * vol


_ks_cache = {}


def pluck(freq, dur, vol):
    key = (round(freq, 2), dur)
    if key not in _ks_cache:
        _ks_cache[key] = ks(freq, dur, 1.0)
    return _ks_cache[key] * vol


def taiko(vol=0.5, low=55):
    n = int(0.6 * SR)
    tt = np.arange(n) / SR
    f = low + 90 * np.exp(-tt * 25)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 7)
    s += np.random.default_rng(3).standard_normal(n) * np.exp(-tt * 60) * 0.3
    return (s * vol).astype(np.float32)


def thunder(dur=3.0, vol=0.4, seed=0):
    n = int(dur * SR)
    x = np.random.default_rng(seed).standard_normal(n).astype(np.float32)
    x = np.convolve(x, np.ones(60) / 60, mode="same")
    tt = np.arange(n) / SR
    env = np.exp(-tt * 1.6) * np.minimum(1, tt * 20)
    return x * env * vol * 6


def audio(path):
    n = int((DUR + 0.5) * SR)
    music = np.zeros(n, np.float32)
    sfx = np.zeros(n, np.float32)
    voice = np.zeros(n, np.float32)
    duck = np.ones(n, np.float32)

    def add(buf, x, t, g=1.0):
        s = int(t * SR)
        e = min(n, s + len(x))
        if e > s:
            buf[s:e] += x[:e - s] * g

    hz = lambda m: 440 * 2 ** ((m - 69) / 12)
    pent = [62, 65, 67, 69, 72, 74, 77, 79, 81]          # D minor pentatonic
    # drone pads (whole piece)
    tt = np.arange(n) / SR
    for m, g in ((38, 0.05), (45, 0.035), (50, 0.03)):
        f0 = hz(m)
        music += (g * (np.sin(2 * np.pi * f0 * tt) + 0.4 * np.sin(2 * np.pi * f0 * 2.003 * tt)
                       + 0.2 * np.sin(2 * np.pi * f0 * 3.01 * tt))).astype(np.float32)
    # guzheng phrases in calm sections
    rnd = random.Random(5)
    t0 = 0.5
    while t0 < 63:
        for j in range(6):
            m = pent[rnd.randrange(len(pent))]
            add(music, pluck(hz(m), 2.5, 0.16), t0 + j * 0.28 + rnd.uniform(0, 0.04))
        t0 += 3.2
    t0 = 124
    while t0 < DUR - 3:
        for j in range(5):
            add(music, pluck(hz(pent[(j * 2 + int(t0)) % len(pent)]), 2.5, 0.15), t0 + j * 0.34)
        t0 += 3.4
    # battle drums 63.3–123.3, 140 bpm
    beat = 60 / 140
    k = 0
    tb = 63.3
    tk = taiko(0.5)
    tk_hi = taiko(0.25, low=110)
    while tb < 123.3:
        add(music, tk, tb)
        if k % 2 == 1:
            add(music, tk_hi, tb + beat / 2)
        if k % 8 == 7:
            for r in range(4):
                add(music, tk_hi, tb + r * beat / 4)
        # power strings: pentatonic ostinato
        add(music, pluck(hz([50, 53, 55, 57][k % 4]), 1.2, 0.18), tb)
        tb += beat
        k += 1
    for tb in (63.3, 78.8, 94.8, 113.3, 118.3):
        add(music, taiko(0.9, low=40), tb)
    # sfx
    S = {k_: load(k_) for k_ in ("sword_swish", "sword_clash", "impact_boom", "divine_energy", "dragon_roar_effect")}
    for i, tt_ in enumerate((0.8, 3.5, 6.2, 21.2, 23.0, 29.5, 52.0, 55.0, 129.0)):
        add(sfx, thunder(3.0, 0.25, i), tt_)
    add(sfx, S["divine_energy"], 9.2, 0.8)
    add(sfx, S["dragon_roar_effect"], 29.7, 1.0)
    add(sfx, S["dragon_roar_effect"], 51.3, 0.7)
    add(sfx, S["divine_energy"], 59.6, 0.6)
    B0 = 63.3
    ev = [(0.05, "divine_energy", 1.0), (0.3, "sword_swish", 0.9), (3.0, "dragon_roar_effect", 1.0),
          (6.0, "sword_swish", 0.8), (6.4, "impact_boom", 0.5), (9.0, "sword_swish", 0.9), (10.6, "sword_swish", 0.9),
          (12.5, "sword_clash", 1.0), (13.5, "sword_clash", 0.9), (15.75, "sword_swish", 0.9),
          (16.45, "sword_swish", 0.9), (17.15, "sword_swish", 0.9), (18.5, "impact_boom", 0.9),
          (22.0, "divine_energy", 1.0), (25.0, "sword_swish", 0.8), (25.8, "sword_swish", 0.8),
          (28.0, "divine_energy", 0.7), (30.9, "impact_boom", 1.0), (31.2, "impact_boom", 0.7),
          (37.0, "divine_energy", 1.0), (40.0, "sword_clash", 0.7), (40.3, "sword_clash", 0.7)]
    for k_ in range(26):
        rr = random.Random(k_ + 600)
        st = rr.uniform(0, 2.9)
        ev.append((37.0 + st + 0.45, "sword_clash", 0.25))
    for k_ in range(8):
        ev.append((40.0 + 0.3 + k_ * 0.33 + 0.25, "sword_clash", 0.6))
    ev += [(43.0, "divine_energy", 0.9), (43.1, "dragon_roar_effect", 0.6), (44.0, "sword_swish", 0.9),
           (46.5, "sword_swish", 1.0), (48.1, "impact_boom", 1.0), (49.0, "divine_energy", 1.0),
           (51.5, "divine_energy", 0.8), (54.0, "sword_swish", 1.0), (54.4, "impact_boom", 1.0),
           (54.7, "impact_boom", 0.8), (55.0, "dragon_roar_effect", 1.0), (56.0, "impact_boom", 0.6)]
    for tt_, nm, g in ev:
        add(sfx, S[nm], B0 + tt_, g)
    for i, tt_ in enumerate((3.0, 18.5, 30.9)):
        add(sfx, thunder(2.5, 0.35, 50 + i), B0 + tt_)
    # voices
    clips = [(s, c) for s, d, w, c, x, k in DIALOG] + [
        (B0 + 0.15, "l_03"), (B0 + 22.05, "l_04"), (B0 + 37.05, "l_05"), (B0 + 34.0, "l_06"),
        (B0 + 43.1, "l_07"), (B0 + 49.6, "l_08"), (B0 + 31.2, "d_03"), (123.4, "d_04"), (128.2, "d_05")]
    for s, c in clips:
        x = load(c)
        if c.startswith("d_"):
            x = reverb(x, ((0.06, 0.45), (0.13, 0.35), (0.26, 0.25), (0.45, 0.15)))
            g = 0.9
        else:
            x = reverb(x, ((0.04, 0.2), (0.09, 0.12)))
            g = 1.0
        add(voice, x, s, g)
        si, ei = int(s * SR), min(n, int((s + len(x) / SR) * SR))
        duck[si:ei] = np.minimum(duck[si:ei], 0.45)
    duck = np.convolve(duck, np.ones(4410) / 4410, mode="same").astype(np.float32)
    voice /= max(1e-6, np.abs(voice).max())
    mix = music * duck * 0.8 + sfx * 0.7 * (0.6 + 0.4 * duck) + voice * 0.95
    env = np.minimum(1, np.minimum(tt / 1.0, (DUR + 0.5 - tt) / 2.5))
    mix *= env
    mix = np.tanh(mix * 1.2) / np.tanh(1.2)
    mix /= max(1e-6, np.abs(mix).max()) / 0.92
    # stereo: slight width via delayed copy
    left = mix
    right = np.concatenate([np.zeros(220, np.float32), mix[:-220]]) * 0.3 + mix * 0.7
    st = np.stack([left, right], 1)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((st * 32767).astype(np.int16).tobytes())


# ======================================================================
# render
# ======================================================================
def render_chunk(args):
    idx, f0, f1, path = args
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                          "-crf", "19", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for i in range(f0, f1):
        p.stdin.write(frame(i / FPS).tobytes())
        if (i - f0) % 600 == 0:
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
    nchunk = nproc * 3
    step = math.ceil(total / nchunk)
    jobs = [(i, i * step, min(total, (i + 1) * step), os.path.join(tmp, f"p{i:02d}.mp4")) for i in range(nchunk)]
    with Pool(nproc) as pool:
        parts = pool.map(render_chunk, jobs)
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as f_:
        for p_ in parts:
            f_.write(f"file '{os.path.abspath(p_)}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out],
                   check=True)
    for p_ in parts + [wav, lst]:
        os.remove(p_)
    os.rmdir(tmp)
    print("done:", out)
