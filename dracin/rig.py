"""Rig animasi 2.5D (gaya Live2D) untuk ilustrasi diam.

Setiap ilustrasi dianimasikan dengan *mesh warp* (PIL Image.MESH) yang digerakkan
parameter waktu nyata — tanpa menggambar ulang karakternya:

  * napas            dada & bahu naik-turun (siklus ~3,8 detik)
  * angin rambut     helai rambut/surai bergoyang, makin bebas makin jauh dari akar
  * kepala           sedikit mengangguk/miring, lebih aktif saat berbicara
  * bibir            membuka-menutup mengikuti amplitudo suara dialog (lip-sync)
  * kedip            mata berkedip acak 2,5–5 detik sekali
  * ekspresi         alis, senyum, dan sipit berubah bertahap (di-ease)
  * jari             genggaman mengencang/mengendur pelan
  * bahu berjalan    tubuh atas berayun berlawanan dengan langkah (untuk sprite)

Koordinat penanda (landmark) ditulis dalam piksel gambar asli, lalu dikali `scale`
gambar yang dirig.
"""
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ----------------------------------------------------------------------------- utils
def gauss(X, Y, cx, cy, rx, ry):
    return np.exp(-(((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2))


def smooth_box(X, Y, cx, cy, rx, ry, p=4):
    """~1 inside an ellipse, smooth falloff outside"""
    d = (((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2) ** 0.5
    return 1 / (1 + d ** (2 * p))


def mesh_warp(img, move_fn, box=None, cell=20):
    """Warp `img` inside `box` by a content-motion field move_fn(X, Y) -> (MX, MY)."""
    x0, y0, x1, y1 = box or (0, 0, img.width, img.height)
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(img.width, int(x1)), min(img.height, int(y1))
    xs = np.unique(np.append(np.arange(x0, x1, cell), x1)).astype(np.float64)
    ys = np.unique(np.append(np.arange(y0, y1, cell), y1)).astype(np.float64)
    X, Y = np.meshgrid(xs, ys)
    MX, MY = move_fn(X, Y)
    # zero motion on the box border so the patch stitches seamlessly
    if box is not None:
        MX[[0, -1], :] = 0
        MX[:, [0, -1]] = 0
        MY[[0, -1], :] = 0
        MY[:, [0, -1]] = 0
    SX, SY = X - MX, Y - MY
    mesh = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            bb = (int(xs[i] - x0), int(ys[j] - y0), int(xs[i + 1] - x0), int(ys[j + 1] - y0))
            mesh.append((bb, (SX[j, i], SY[j, i], SX[j + 1, i], SY[j + 1, i],
                              SX[j + 1, i + 1], SY[j + 1, i + 1], SX[j, i + 1], SY[j, i + 1])))
    patch = img.transform((x1 - x0, y1 - y0), Image.MESH, mesh, Image.BICUBIC)
    if box is None:
        return patch
    out = img.copy()
    out.paste(patch, (x0, y0))
    return out


def noise1(t, seed=0):
    i = math.floor(t)
    f = t - i
    r = lambda k: random.Random(k * 7919 + seed).uniform(-1, 1)
    u = f * f * (3 - 2 * f)
    return r(i) * (1 - u) + r(i + 1) * u


def blink_amount(t, seed=0):
    """0 = open, 1 = closed. Random blinks every 2.5–5 s, 0.18 s long."""
    rnd = random.Random(seed)
    tt = rnd.uniform(0.5, 2.0)
    while tt < t - 0.3:
        tt += rnd.uniform(2.5, 5.0)
    d = t - tt
    if 0 <= d < 0.18:
        return math.sin(d / 0.18 * math.pi) ** 0.7
    return 0.0


# ------------------------------------------------------------------------- lip sync
class LipSync:
    """Mouth openness + width from a voice clip's amplitude/brightness, sampled at fps."""

    def __init__(self, samples, sr, fps=60):
        hop = sr // fps
        n = len(samples) // hop
        fr = samples[:n * hop].reshape(n, hop)
        rms = np.sqrt((fr ** 2).mean(1) + 1e-12)
        # brightness: share of energy in the high band (first-difference ≈ high-pass)
        hi = np.sqrt((np.diff(fr, axis=1) ** 2).mean(1) + 1e-12)
        bright = np.clip(hi / (rms * 1.2 + 1e-9), 0, 1)
        env = np.zeros(n)
        e = 0.0
        ref = np.percentile(rms, 95) + 1e-9
        for i in range(n):
            target = min(1.0, rms[i] / ref)
            e += (target - e) * (0.55 if target > e else 0.22)   # fast attack, softer release
            env[i] = e
        env[env < 0.08] = 0
        self.open = np.clip(env * 1.1, 0, 1) ** 0.8
        self.wide = np.convolve(bright, np.ones(5) / 5, mode="same")
        self.fps = fps
        self.dur = n / fps

    def at(self, t):
        if t < 0 or t >= self.dur:
            return 0.0, 0.5
        i = int(t * self.fps)
        return float(self.open[i]), float(self.wide[i])


# ------------------------------------------------------------------------------ rig
class Rig:
    """spec keys (original-pixel coords, multiplied by scale):
        eyes   [(cx, cy, rx, ry), …]          brows [(cx, cy, rx, ry), …]
        mouth  (cx, cy, half_w, half_h)       chin  (cx, cy, rx, ry)
        head   dict(pivot=(x, y), cx, cy, rx, ry)
        chest  (cx, cy, rx, ry)               shoulders [(cx, cy, rx, ry), …]
        hair   [dict(c=(cx, cy), r=(rx, ry), root=(x, y), dir=(dx, dy), len=L, amp=A), …]
        jaw    dict(hinge=(x, y), c=(cx, cy), r=(rx, ry), max=rad)   (creatures)
        fingers dict(pivot=(x, y), c=(cx, cy), r=(rx, ry))
        sway   amplitude (px) of a slow whole-body undulation (serpentine bodies)
        face_box (x0, y0, x1, y1) region for the fine face pass
    """

    def __init__(self, img, spec, scale=1.0, seed=0):
        self.img = img
        self.s = scale
        self.spec = spec
        self.seed = seed

    def S(self, *v):
        return [x * self.s for x in v]

    # fine pass: eyes, brows, mouth
    def _face(self, img, t, o, wide, expr):
        sp = self.spec
        if "face_box" not in sp:
            return img
        bx = self.S(*sp["face_box"])
        blink = max(blink_amount(t, self.seed), 0.3 * expr.get("squint", 0))
        brow = expr.get("brow", 0.0)
        smile = expr.get("smile", 0.0)

        def move(X, Y):
            MX = np.zeros_like(X)
            MY = np.zeros_like(Y)
            for cx, cy, rx, ry in sp.get("eyes", []):
                cx, cy, rx, ry = self.S(cx, cy, rx, ry)
                w = smooth_box(X, Y, cx, cy, rx * 1.25, ry * 2.2)
                # squash toward the lower lid: content above moves down
                MY += w * blink * np.clip(cy + ry * 0.4 - Y, -ry * 0.5, ry * 2.2) * 0.85
            for cx, cy, rx, ry in sp.get("brows", []):
                cx, cy, rx, ry = self.S(cx, cy, rx, ry)
                w = gauss(X, Y, cx, cy, rx * 1.3, ry * 2.0)
                MY += w * (-brow * 6.0 * self.s + blink * 1.5 * self.s)
            if "mouth" in sp:
                cx, cy, hw, hh = self.S(*sp["mouth"])
                M = hh * 1.5 * o
                wl = gauss(X, Y, cx, cy + hh * 1.6, hw * 1.5, hh * 2.6) * (Y > cy - hh * 0.3)
                wu = gauss(X, Y, cx, cy - hh * 0.8, hw * 1.3, hh * 1.4) * (Y < cy)
                MY += wl * M - wu * M * 0.25
                # width: wide vowels stretch, round vowels pinch
                wc = gauss(X, Y, cx, cy, hw * 1.6, hh * 2.2)
                MX += wc * (X - cx) / hw * (wide - 0.5) * 0.35 * hw * o
                # smile: corners up & out
                for side in (-1, 1):
                    wk = gauss(X, Y, cx + side * hw * 0.95, cy, hw * 0.45, hh * 1.3)
                    MY += -wk * smile * hh * 0.7
                    MX += wk * side * smile * hw * 0.12
            if "chin" in sp and "mouth" in sp:
                cx, cy, rx, ry = self.S(*sp["chin"])
                MY += gauss(X, Y, cx, cy, rx, ry) * sp["mouth"][3] * self.s * 1.1 * o
            return MX, MY

        out = mesh_warp(img, move, bx, cell=max(4, int(4 * self.s)))
        # mouth cavity + teeth hint
        if "mouth" in sp and o > 0.06:
            cx, cy, hw, hh = self.S(*sp["mouth"])
            gap = hh * 1.5 * o
            ww = hw * (0.48 + 0.18 * (wide - 0.5))
            ov = Image.new("RGBA", out.size)
            d = ImageDraw.Draw(ov)
            d.ellipse((cx - ww, cy - gap * 0.35, cx + ww, cy + gap * 0.95), fill=(52, 18, 22, int(235 * min(1, o * 1.6))))
            if o > 0.35:
                d.ellipse((cx - ww * 0.7, cy - gap * 0.3, cx + ww * 0.7, cy + gap * 0.12),
                          fill=(215, 205, 200, int(150 * (o - 0.35) / 0.65)))
            ov = ov.filter(ImageFilter.GaussianBlur(max(0.8, 0.7 * self.s)))
            out = out.convert("RGBA")
            out.alpha_composite(ov)
            out = out.convert(img.mode)
        return out

    # coarse pass: breath, head, hair, jaw, fingers, sway
    def _body(self, img, t, o, talk, wind):
        sp = self.spec
        br = math.sin(2 * math.pi * t / 3.8 + self.seed)
        hd = sp.get("head")
        ang = 0.0
        nod = 0.0
        if hd:
            ang = 0.010 * math.sin(0.55 * t + self.seed) + 0.012 * talk * noise1(t * 2.2, self.seed + 3)
            nod = (2.0 + 4.0 * talk * o) * self.s * noise1(t * 3.0, self.seed + 7) * (0.4 + talk)

        def move(X, Y):
            MX = np.zeros_like(X)
            MY = np.zeros_like(Y)
            if "chest" in sp:
                cx, cy, rx, ry = self.S(*sp["chest"])
                w = gauss(X, Y, cx, cy, rx, ry)
                MY += -br * 3.2 * self.s * w
                MX += br * 1.0 * self.s * w * (X - cx) / rx
            for cx, cy, rx, ry in sp.get("shoulders", []):
                cx, cy, rx, ry = self.S(cx, cy, rx, ry)
                MY += -br * 2.4 * self.s * gauss(X, Y, cx, cy, rx, ry)
            if hd:
                px, py = self.S(*hd["pivot"])
                cx, cy, rx, ry = self.S(hd["cx"], hd["cy"], hd["rx"], hd["ry"])
                w = smooth_box(X, Y, cx, cy, rx, ry, p=3)
                dx, dy = X - px, Y - py
                MX += w * (dx * (math.cos(ang) - 1) - dy * math.sin(ang))
                MY += w * (dx * math.sin(ang) + dy * (math.cos(ang) - 1) + nod)
                MY += -w * br * 1.6 * self.s
            if "jaw" in sp:
                j = sp["jaw"]
                hx, hy = self.S(*j["hinge"])
                cx, cy = self.S(*j["c"])
                rx, ry = self.S(*j["r"])
                a = j["max"] * (o - 0.25)
                w = smooth_box(X, Y, cx, cy, rx, ry, p=3)
                dx, dy = X - hx, Y - hy
                MX += w * (dx * (math.cos(a) - 1) - dy * math.sin(a))
                MY += w * (dx * math.sin(a) + dy * (math.cos(a) - 1))
            if "fingers" in sp:
                f = sp["fingers"]
                px, py = self.S(*f["pivot"])
                cx, cy = self.S(*f["c"])
                rx, ry = self.S(*f["r"])
                a = 0.035 * math.sin(t * 1.3 + 1) + 0.02 * noise1(t * 1.7, 11)
                w = gauss(X, Y, cx, cy, rx, ry)
                dx, dy = X - px, Y - py
                MX += w * (dx * (math.cos(a) - 1) - dy * math.sin(a))
                MY += w * (dx * math.sin(a) + dy * (math.cos(a) - 1))
            for h in sp.get("hair", []):
                cx, cy = self.S(*h["c"])
                rx, ry = self.S(*h["r"])
                rx_, ry_ = self.S(*h["root"])
                ddx, ddy = h["dir"]
                n = math.hypot(ddx, ddy)
                ddx, ddy = ddx / n, ddy / n
                L = h["len"] * self.s
                w = smooth_box(X, Y, cx, cy, rx, ry, p=2)
                along = ((X - rx_) * ddx + (Y - ry_) * ddy) / L
                stiff = np.clip(along, 0, 1) ** 1.4
                ph = along * 5.5 - t * 3.1 * wind - self.seed
                sway = (np.sin(ph) + 0.45 * np.sin(ph * 2.3 + 1.7)) * h["amp"] * self.s * wind
                gust = 0.6 + 0.4 * noise1(t * 0.7, 21)
                MX += w * stiff * sway * gust * (-ddy)
                MY += w * stiff * sway * gust * ddx
            if sp.get("sway"):
                a = sp["sway"] * self.s
                MX += a * np.sin(Y / (160 * self.s) - t * 1.1) * (1 - Y / (img.height + 1)) * wind
                MY += a * 0.4 * np.sin(X / (190 * self.s) - t * 0.9)
            return MX, MY

        return mesh_warp(img, move, None, cell=max(12, int(10 * self.s)))

    def frame(self, t, mouth=(0.0, 0.5), talk=0.0, expr=None, wind=1.0):
        o, wide = mouth
        img = self._face(self.img, t, o, wide, expr or {})
        return self._body(img, t, o, talk, wind)


# ----------------------------------------------------------------------- sprites
def sprite_motion(sp, t, step_phase=None, breath=1.0, hair=None, seed=0):
    """Cut-out sprite (RGBA, full body): breathing + shoulder swing while walking + hair."""
    w, h = sp.size
    br = math.sin(2 * math.pi * t / 3.6 + seed) * breath
    waist = h * 0.46
    ang = 0.0 if step_phase is None else 0.03 * math.sin(step_phase * 2 * math.pi)

    def move(X, Y):
        MX = np.zeros_like(X)
        MY = np.zeros_like(Y)
        chest = gauss(X, Y, w * 0.5, h * 0.27, w * 0.35, h * 0.09)
        MY += -br * 0.006 * h * chest
        for side in (-1, 1):
            MY += -br * 0.004 * h * gauss(X, Y, w * (0.5 + side * 0.3), h * 0.2, w * 0.18, h * 0.05)
        if step_phase is not None:
            up = np.clip((waist - Y) / waist, 0, 1)
            dx, dy = X - w / 2, Y - waist
            MX += up * (dx * (math.cos(ang) - 1) - dy * math.sin(ang))
            MY += up * (dx * math.sin(ang) + dy * (math.cos(ang) - 1))
            for side in (-1, 1):   # shoulders bob alternately
                MY += side * math.sin(step_phase * 2 * math.pi) * 0.006 * h * gauss(
                    X, Y, w * (0.5 + side * 0.3), h * 0.2, w * 0.2, h * 0.06)
        if hair:
            cx, cy, rx, ry, root = hair
            wgt = smooth_box(X, Y, cx * w, cy * h, rx * w, ry * h, p=2)
            stiff = np.clip((Y - root * h) / (h * 0.35), 0, 1) ** 1.3
            ph = Y / (h * 0.06) - t * 3.0 - seed
            MX += wgt * stiff * (np.sin(ph) + 0.4 * np.sin(ph * 2.1 + 1)) * 0.012 * w * 4
        return MX, MY

    pad = int(w * 0.08)
    big = Image.new("RGBA", (w + 2 * pad, h + pad))
    big.paste(sp, (pad, pad))

    def move_p(X, Y):
        mx, my = move(X - pad, Y - pad)
        return mx, my

    out = mesh_warp(big, move_p, None, cell=max(8, w // 14))
    bb = out.getbbox()
    return out.crop(bb) if bb else out
