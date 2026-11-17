"""SUARA HATI — video pendek 60 detik.

Merender video 1280x720 @30fps dari dua lembar karakter (Arka & Nara),
lengkap dengan musik piano sintetis, suara hujan, dan detak jantung.

Pemakaian:
    python3 render.py <nara_sheet.png> <arka_sheet.png> <output.mp4>
"""
import math
import random
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS, DUR = 1280, 720, 30, 60.0
NARA_SHEET, ARKA_SHEET, OUT = sys.argv[1], sys.argv[2], sys.argv[3]

FONT_DIR = "/usr/share/fonts/truetype/crosextra/"
F_TITLE = ImageFont.truetype(FONT_DIR + "Caladea-Bold.ttf", 96)
F_SUB = ImageFont.truetype(FONT_DIR + "Caladea-Italic.ttf", 30)
F_CHAP = ImageFont.truetype(FONT_DIR + "Caladea-Bold.ttf", 26)
F_TEXT = ImageFont.truetype(FONT_DIR + "Carlito-Regular.ttf", 32)
F_NAME = ImageFont.truetype(FONT_DIR + "Carlito-Bold.ttf", 26)
F_QUOTE = ImageFont.truetype(FONT_DIR + "Caladea-Italic.ttf", 40)

# ---------------------------------------------------------------- sprites
SHEETS = {
    # path, faces x0, x1, y0, y1, full-body box
    "nara": (NARA_SHEET, 305, 1224, 0, 1285, (30, 8, 303, 1280)),
    "arka": (ARKA_SHEET, 370, 1212, 0, 1296, (5, 4, 366, 1175)),
}
_sheet_cache = {}


def sheet(who):
    if who not in _sheet_cache:
        _sheet_cache[who] = Image.open(SHEETS[who][0]).convert("RGBA")
    return _sheet_cache[who]


def body(who, height):
    im = sheet(who).crop(SHEETS[who][5])
    w = int(im.width * height / im.height)
    return im.resize((w, height), Image.LANCZOS)


def rounded_mask(size, radius, feather):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle(
        (feather, feather, size[0] - feather, size[1] - feather), radius, fill=255)
    return m.filter(ImageFilter.GaussianBlur(feather / 2))


def face(who, r, c, size=(300, 340), panel=(255, 240, 225)):
    """Ekspresi wajah (baris r, kolom c) dalam kartu bersudut membulat."""
    _, x0, x1, y0, y1, _ = SHEETS[who]
    tw, th = (x1 - x0) / 5, (y1 - y0) / 6
    inset = 9
    t = sheet(who).crop((int(x0 + c * tw + inset), int(y0 + r * th + inset),
                         int(x0 + (c + 1) * tw - inset), int(y0 + (r + 1) * th - inset - 6)))
    # cover-fit
    s = max(size[0] / t.width, size[1] / t.height)
    t = t.resize((int(t.width * s), int(t.height * s)), Image.LANCZOS)
    card = Image.new("RGBA", size, panel + (235,))
    # soft radial glow on panel
    g = Image.new("L", size, 0)
    ImageDraw.Draw(g).ellipse((-60, -40, size[0] + 60, size[1] + 80), fill=90)
    g = g.filter(ImageFilter.GaussianBlur(40))
    card = Image.composite(Image.new("RGBA", size, (255, 255, 255, 235)), card, g)
    card.alpha_composite(t, ((size[0] - t.width) // 2, size[1] - t.height))
    mask = rounded_mask(size, 28, 6)
    card.putalpha(Image.fromarray(np.minimum(np.array(card.getchannel("A")), np.array(mask))))
    # border + shadow
    out = Image.new("RGBA", (size[0] + 40, size[1] + 40), (0, 0, 0, 0))
    sh = Image.new("L", out.size, 0)
    ImageDraw.Draw(sh).rounded_rectangle((22, 26, size[0] + 22, size[1] + 26), 28, fill=150)
    out.putalpha(sh.filter(ImageFilter.GaussianBlur(12)))
    out.alpha_composite(card, (20, 20))
    ImageDraw.Draw(out).rounded_rectangle((26, 26, size[0] + 14, size[1] + 14), 26,
                                          outline=(255, 255, 255, 140), width=2)
    return out


# ------------------------------------------------------------ backgrounds
def grad(top, bottom, h=H, w=W):
    t = np.linspace(0, 1, h)[:, None, None]
    a = np.array(top, float)[None, None, :]
    b = np.array(bottom, float)[None, None, :]
    return np.repeat((a * (1 - t) + b * t), w, axis=1)


def to_img(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGBA")


def city(img, color, win, seed, base=H, maxh=260):
    rnd = random.Random(seed)
    d = ImageDraw.Draw(img)
    x = -20
    while x < W:
        bw, bh = rnd.randint(50, 130), rnd.randint(80, maxh)
        d.rectangle((x, base - bh, x + bw, base), fill=color)
        if win:
            for wy in range(base - bh + 12, base - 10, 18):
                for wx in range(x + 8, x + bw - 8, 16):
                    if rnd.random() < 0.28:
                        d.rectangle((wx, wy, wx + 6, wy + 9), fill=win)
        x += bw + rnd.randint(-10, 6)
    return img


def stars(img, n, seed):
    rnd = random.Random(seed)
    d = ImageDraw.Draw(img)
    for _ in range(n):
        x, y, r = rnd.randint(0, W), rnd.randint(0, int(H * 0.7)), rnd.choice([1, 1, 1, 2])
        a = rnd.randint(80, 230)
        d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, a))
    return img


def bg_title():
    im = to_img(grad((8, 10, 30), (40, 22, 60)))
    return stars(im, 220, 1)


def bg_rain():
    im = to_img(grad((14, 20, 38), (38, 46, 70)))
    far = city(Image.new("RGBA", (W, H)), (28, 34, 56, 255), (90, 100, 130, 255), 2, maxh=330)
    near = city(Image.new("RGBA", (W, H)), (12, 15, 28, 255), (240, 190, 110, 255), 3, maxh=220)
    im.alpha_composite(far.filter(ImageFilter.GaussianBlur(2)))
    im.alpha_composite(near)
    # wet street reflection glow
    glow = Image.new("RGBA", (W, H))
    ImageDraw.Draw(glow).rectangle((0, H - 90, W, H), fill=(240, 180, 110, 50))
    im.alpha_composite(glow.filter(ImageFilter.GaussianBlur(25)))
    return im


def bg_sunset():
    im = to_img(grad((255, 150, 110), (90, 60, 120)))
    sun = Image.new("RGBA", (W, H))
    ImageDraw.Draw(sun).ellipse((W // 2 - 110, 330, W // 2 + 110, 550), fill=(255, 225, 160, 255))
    im.alpha_composite(sun.filter(ImageFilter.GaussianBlur(18)))
    im.alpha_composite(city(Image.new("RGBA", (W, H)), (70, 40, 80, 255), None, 5, maxh=200))
    # cherry petals layer baked lightly
    return im


def bg_dark_red():
    im = to_img(grad((30, 6, 12), (8, 4, 10)))
    return im


def bg_dawn():
    im = to_img(grad((150, 190, 240), (255, 210, 190)))
    rays = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(rays)
    cx, cy = W // 2, -60
    for i in range(12):
        a = math.radians(40 + i * 9)
        d.polygon([(cx, cy), (cx + 1600 * math.cos(a - .03), cy + 1600 * math.sin(a - .03)),
                   (cx + 1600 * math.cos(a + .03), cy + 1600 * math.sin(a + .03))],
                  fill=(255, 255, 240, 40))
    im.alpha_composite(rays.filter(ImageFilter.GaussianBlur(20)))
    im.alpha_composite(city(Image.new("RGBA", (W, H)), (120, 130, 170, 200), None, 9, maxh=170))
    return im


# ---------------------------------------------------------------- effects
RND = random.Random(42)
DROPS = [(RND.uniform(0, W + 200), RND.uniform(0, H), RND.uniform(700, 1100), RND.uniform(14, 30))
         for _ in range(260)]
PETALS = [(RND.uniform(0, W), RND.uniform(0, H), RND.uniform(30, 70), RND.uniform(0, 6.28),
           RND.uniform(3, 6)) for _ in range(45)]
MOTES = [(RND.uniform(0, W), RND.uniform(0, H), RND.uniform(8, 25), RND.uniform(0, 6.28),
          RND.uniform(1.5, 4)) for _ in range(70)]


def rain(img, t, alpha=120):
    ov = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(ov)
    for x, y, v, ln in DROPS:
        yy = (y + v * t) % (H + 40) - 40
        xx = (x - 0.25 * v * t) % (W + 200) - 100
        d.line((xx, yy, xx - ln * 0.25, yy + ln), fill=(190, 210, 255, alpha), width=1)
    img.alpha_composite(ov)


def petals(img, t):
    ov = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(ov)
    for x, y, v, ph, r in PETALS:
        yy = (y + v * t) % (H + 20) - 10
        xx = (x + 40 * math.sin(t * 1.3 + ph) - 25 * t) % W
        d.ellipse((xx - r, yy - r * 0.6, xx + r, yy + r * 0.6), fill=(255, 200, 215, 200))
    img.alpha_composite(ov)


def motes(img, t, color=(255, 240, 200)):
    ov = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(ov)
    for x, y, v, ph, r in MOTES:
        yy = (y - v * t) % H
        xx = x + 15 * math.sin(t + ph)
        a = int(120 + 100 * math.sin(t * 2 + ph))
        d.ellipse((xx - r, yy - r, xx + r, yy + r), fill=color + (max(a, 0),))
    img.alpha_composite(ov.filter(ImageFilter.GaussianBlur(1.2)))


def vignette():
    y, x = np.mgrid[0:H, 0:W]
    d = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2)
    a = np.clip((d - 0.65) * 255, 0, 200).astype(np.uint8)
    v = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    v.putalpha(Image.fromarray(a))
    return v


VIG = vignette()


# ------------------------------------------------------------------ text
def ease(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def fade(t, a, b, f=0.5):
    """Opacity 0..1 for an element visible between a and b."""
    if t < a or t > b:
        return 0.0
    return ease((t - a) / f) * ease((b - t) / f)


def wrap(text, font, maxw):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        test = (cur + " " + w_).strip()
        if font.getlength(test) <= maxw:
            cur = test
        else:
            lines.append(cur)
            cur = w_
    lines.append(cur)
    return lines


def text_layer(lines, font, color, center_y, alpha, shadow=True, x_center=W // 2, spacing=10):
    ov = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(ov)
    lh = font.size + spacing
    y = center_y - lh * len(lines) / 2
    for ln in lines:
        w_ = font.getlength(ln)
        x = x_center - w_ / 2
        if shadow:
            d.text((x + 2, y + 3), ln, font=font, fill=(0, 0, 0, int(200 * alpha)))
        d.text((x, y), ln, font=font, fill=color + (int(255 * alpha),))
        y += lh
    return ov


def subtitle(img, t, a, b, text, name=None, name_color=(255, 200, 120)):
    al = fade(t, a, b, 0.35)
    if al <= 0:
        return
    lines = wrap(text, F_TEXT, 980)
    box_h = 46 * len(lines) + (40 if name else 24)
    ov = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(ov)
    top = H - 40 - box_h
    d.rounded_rectangle((110, top, W - 110, H - 40), 18, fill=(10, 10, 20, int(170 * al)))
    if name:
        d.text((140, top + 10), name, font=F_NAME, fill=name_color + (int(255 * al),))
    img.alpha_composite(ov)
    cy = top + (40 if name else 14) + 46 * len(lines) / 2 - 4
    img.alpha_composite(text_layer(lines, F_TEXT, (250, 248, 240), cy, al))


def chapter(img, t, a, b, label):
    al = fade(t, a, b, 0.5)
    if al <= 0:
        return
    ov = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(ov)
    w_ = F_CHAP.getlength(label)
    d.rounded_rectangle((40, 34, 40 + w_ + 40, 82), 24, fill=(0, 0, 0, int(120 * al)))
    d.text((60, 42), label, font=F_CHAP, fill=(255, 235, 200, int(255 * al)))
    img.alpha_composite(ov)


def paste(img, sprite, x, y, alpha=1.0, scale=1.0):
    if alpha <= 0:
        return
    s = sprite
    if scale != 1.0:
        s = s.resize((max(1, int(s.width * scale)), max(1, int(s.height * scale))), Image.BILINEAR)
    if alpha < 1:
        a = np.array(s.getchannel("A"), float) * alpha
        s = s.copy()
        s.putalpha(Image.fromarray(a.astype(np.uint8)))
    img.alpha_composite(s, (int(x - s.width / 2), int(y - s.height / 2)))


# ---------------------------------------------------------------- assets
print("preparing assets…", flush=True)
BG = {"title": bg_title(), "rain": bg_rain(), "sunset": bg_sunset(),
      "red": bg_dark_red(), "dawn": bg_dawn()}
ARKA_BODY = body("arka", 600)
NARA_BODY = body("nara", 620)
ARKA_BODY_DARK = Image.fromarray(
    (np.array(ARKA_BODY) * np.array([0.75, 0.8, 0.95, 1])).astype(np.uint8))

FA = {k: face("arka", *rc, panel=(205, 215, 235)) for k, rc in {
    "neutral": (0, 0), "think": (3, 0), "troubled": (1, 4), "shock": (1, 2),
    "pain": (4, 4), "grit": (4, 3), "sad": (2, 4), "soft": (5, 1), "smile": (2, 2),
    "closed": (3, 3), "worry": (1, 3)}.items()}
FN = {k: face("nara", *rc) for k, rc in {
    "smile": (0, 0), "curious": (2, 4), "gentle": (2, 3), "warm": (0, 1),
    "happy": (0, 2), "wink": (3, 3), "calm": (3, 2)}.items()}

# --------------------------------------------------------------- scenes
# (start, end) in seconds
def scene_title(img, t):
    img.alpha_composite(BG["title"])
    motes(img, t, (200, 210, 255))
    a = fade(t, 0.6, 5.2, 1.0)
    # heart glow
    glow = Image.new("RGBA", (W, H))
    r = 140 + 10 * math.sin(t * 4)
    ImageDraw.Draw(glow).ellipse((W / 2 - r, 300 - r, W / 2 + r, 300 + r),
                                 fill=(255, 120, 140, int(70 * a)))
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(50)))
    img.alpha_composite(text_layer(["SUARA HATI"], F_TITLE, (255, 245, 235), 300, a))
    img.alpha_composite(text_layer(["Ketika Logika Tidak Lagi Menjadi Jawaban"], F_SUB,
                                   (230, 210, 230), 390, fade(t, 1.6, 5.2, 1.0)))


def scene_bab1(img, t):  # 5 - 13
    lt = t - 5
    img.alpha_composite(BG["rain"])
    paste(img, ARKA_BODY_DARK, 330 + lt * 6, 410, 1.0)
    rain(img, t)
    paste(img, FA["neutral"], 900, 300, fade(lt, 0.3, 4.3))
    paste(img, FA["think"], 900, 300, fade(lt, 4.0, 8.0))
    chapter(img, lt, 0.2, 8, "Bab 1 — Kehidupan yang Selalu Diperhitungkan")
    subtitle(img, lt, 0.4, 4.0, "Bagi Arka, setiap keputusan harus punya alasan. "
                                 "Setiap risiko harus dihitung berkali-kali.", "NARASI")
    subtitle(img, lt, 4.2, 7.8, "Ia selalu memilih jalan paling aman… "
                                 "bahkan ketika itu membuat orang lain terluka.", "NARASI")


def scene_bab2(img, t):  # 13 - 22
    lt = t - 13
    img.alpha_composite(BG["sunset"])
    paste(img, ARKA_BODY, 250, 420)
    paste(img, NARA_BODY, 1040, 410)
    petals(img, t)
    paste(img, FN["curious"], 760, 260, fade(lt, 0.3, 5.0), 0.85)
    paste(img, FA["troubled"], 520, 260, fade(lt, 4.8, 9.0), 0.85)
    chapter(img, lt, 0.2, 9, "Bab 2 — Sesuatu yang Tidak Bisa Dijelaskan")
    subtitle(img, lt, 0.4, 5.0, "Kalau semua hal harus masuk akal, lalu bagaimana kamu "
                                 "menjelaskan perasaanmu sendiri?", "NARA", (255, 170, 140))
    subtitle(img, lt, 5.2, 8.8, "…", "ARKA", (150, 190, 255))


def scene_bab3(img, t):  # 22 - 32
    lt = t - 22
    beat = max(0, math.sin(lt * math.pi * 2 * 1.2)) ** 8   # heartbeat pulse
    img.alpha_composite(BG["rain"])
    red = Image.new("RGBA", (W, H), (120, 0, 20, int(60 + 60 * beat)))
    img.alpha_composite(red)
    rain(img, t, 150)
    shake = 4 * beat
    paste(img, FA["shock"], 640 + shake, 330, fade(lt, 0.2, 3.5), 1.1)
    paste(img, FA["pain"], 640 - shake, 330, fade(lt, 3.3, 7.0), 1.1 + 0.03 * beat)
    paste(img, FA["grit"], 640, 330, fade(lt, 6.8, 10.0), 1.1 + 0.03 * beat)
    chapter(img, lt, 0.2, 10, "Bab 3 — Keputusan yang Tidak Masuk Akal")
    # pikiran vs hati
    al_p = fade(lt, 3.4, 6.9, 0.3)
    al_h = fade(lt, 4.6, 6.9, 0.3)
    img.alpha_composite(text_layer(["PIKIRAN:", "“Jangan ikut campur.”"],
                                   F_NAME, (170, 200, 255), 300, al_p, x_center=230))
    img.alpha_composite(text_layer(["HATI:", "“Tolong dia.”"],
                                   F_NAME, (255, 150, 160), 300, al_h, x_center=1050))
    subtitle(img, lt, 0.4, 3.3, "Seseorang membutuhkan pertolongan.", "NARASI")
    subtitle(img, lt, 7.0, 9.8, "Untuk pertama kalinya, Arka melangkah… "
                                 "tanpa tahu bagaimana akhirnya.", "NARASI")


def scene_climax(img, t):  # 32 - 49
    lt = t - 32
    img.alpha_composite(BG["sunset"])
    dim = Image.new("RGBA", (W, H), (40, 20, 60, 90))
    img.alpha_composite(dim)
    petals(img, t)
    chapter(img, lt, 0.2, 17, "Bab 4 — Keberanian yang Lahir dari Hati")
    paste(img, FA["worry"], 330, 300, fade(lt, 0.2, 4.4))
    paste(img, FA["sad"], 330, 300, fade(lt, 4.2, 17))
    paste(img, FN["gentle"], 950, 300, fade(lt, 0.2, 8.8))
    paste(img, FN["warm"], 950, 300, fade(lt, 8.6, 17))
    subtitle(img, lt, 0.4, 4.2, "Bagaimana kalau aku gagal? Bagaimana kalau semuanya "
                                 "jadi lebih buruk?", "ARKA", (150, 190, 255))
    subtitle(img, lt, 4.4, 8.6, "Setidaknya kamu sudah mencoba. Lebih menyakitkan tahu "
                                 "kamu bisa, tapi memilih diam.", "NARA", (255, 170, 140))
    subtitle(img, lt, 8.8, 11.6, "Jadi… aku harus berhenti berpikir?", "ARKA", (150, 190, 255))
    subtitle(img, lt, 11.8, 16.8, "Bukan. Gunakan pikiranmu untuk memahami dunia. "
                                   "Tapi jangan biarkan pikiranmu membungkam hatimu.",
             "NARA", (255, 170, 140))


def scene_bab5(img, t):  # 49 - 55
    lt = t - 49
    img.alpha_composite(BG["dawn"])
    motes(img, t)
    paste(img, ARKA_BODY, 470 + 10 * ease(lt / 6), 420)
    paste(img, NARA_BODY, 810 - 10 * ease(lt / 6), 410)
    paste(img, FA["smile"], 190, 290, fade(lt, 0.5, 6), 0.75)
    paste(img, FN["happy"], 1090, 290, fade(lt, 0.8, 6), 0.75)
    chapter(img, lt, 0.2, 6, "Bab 5 — Arti Menjadi Manusia")
    subtitle(img, lt, 0.4, 5.8, "Pikiran membantunya menemukan jalan, "
                                 "tetapi hatilah yang memberinya alasan untuk melangkah.",
             "NARASI")


def scene_end(img, t):  # 55 - 60
    lt = t - 55
    img.alpha_composite(BG["title"])
    motes(img, t, (255, 200, 210))
    a = fade(lt, 0.3, 5.4, 0.8)
    img.alpha_composite(text_layer(
        ["“Pikiran memberimu alasan untuk takut.", "Hati memberimu alasan untuk berani.”"],
        F_QUOTE, (255, 240, 235), 270, a))
    img.alpha_composite(text_layer(["SUARA HATI"], F_TITLE, (255, 205, 210), 450,
                                   fade(lt, 1.6, 5.4, 0.8)))
    img.alpha_composite(text_layer(
        ["Jangan selalu gunakan pikiranmu. Sesekali, gunakanlah hatimu."],
        F_SUB, (220, 210, 230), 545, fade(lt, 2.4, 5.4, 0.8)))


SCENES = [(0, 5, scene_title), (5, 13, scene_bab1), (13, 22, scene_bab2),
          (22, 32, scene_bab3), (32, 49, scene_climax), (49, 55, scene_bab5),
          (55, 60, scene_end)]


def frame(t):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    for a, b, fn in SCENES:
        if a <= t < b:
            fn(img, t)
            img.alpha_composite(VIG)
            # dip-to-black transitions
            k = min(ease((t - a) / 0.4), ease((b - t) / 0.4))
            if k < 1:
                img.alpha_composite(Image.new("RGBA", (W, H), (0, 0, 0, int(255 * (1 - k)))))
            break
    return img.convert("RGB")


# ----------------------------------------------------------------- audio
def audio(path, sr=44100):
    n = int(DUR * sr)
    t = np.arange(n) / sr
    out = np.zeros(n)

    def note(freq, start, length, vol):
        s = int(start * sr)
        ln = min(int(length * sr), n - s)
        if ln <= 0:
            return
        tt = np.arange(ln) / sr
        env = np.exp(-tt * 2.2) * np.minimum(1, tt * 200)
        w_ = (np.sin(2 * np.pi * freq * tt) + 0.35 * np.sin(4 * np.pi * freq * tt)
              + 0.12 * np.sin(6 * np.pi * freq * tt))
        out[s:s + ln] += vol * env * w_

    def hz(m):
        return 440 * 2 ** ((m - 69) / 12)

    # progression: Am - F - C - G (melancholic), then brighter C - G - Am - F at the end
    prog_sad = [[57, 60, 64], [53, 57, 60], [48, 55, 64], [55, 59, 62]]
    prog_hope = [[48, 55, 64], [55, 59, 62], [57, 60, 64], [53, 57, 65]]
    bar = 2.5
    tt = 0.0
    i = 0
    while tt < DUR - 0.5:
        prog = prog_hope if tt >= 46 else prog_sad
        chord = prog[i % 4]
        vol = 0.06 if 22 <= tt < 32 else 0.09
        note(hz(chord[0] - 12), tt, 3.0, vol * 1.2)
        for k, m in enumerate(chord + [chord[1] + 12, chord[2] + 12, chord[1] + 12]):
            note(hz(m), tt + k * bar / 6, 2.0, vol * 0.7)
        if tt >= 32:  # melody line during climax / ending
            note(hz(chord[2] + 12), tt + bar * 0.5, 1.8, vol * 0.6)
        tt += bar
        i += 1

    # rain noise (5-13, 22-32)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(n)
    noise = np.convolve(noise, np.ones(6) / 6, mode="same")
    rain_env = np.zeros(n)
    for a, b in [(5, 13), (22, 32)]:
        m = (t >= a) & (t < b)
        rain_env[m] = np.minimum(1, np.minimum(t[m] - a, b - t[m]) / 0.8)
    out += 0.05 * noise * rain_env

    # heartbeat during bab 3
    for k in range(int(10 * 1.2)):
        for off, v in [(0, 1.0), (0.18, 0.7)]:
            s0 = 22 + k / 1.2 + off
            s = int(s0 * sr)
            ln = int(0.15 * sr)
            tt_ = np.arange(ln) / sr
            out[s:s + ln] += 0.35 * v * np.sin(2 * np.pi * 50 * tt_) * np.exp(-tt_ * 30)

    # master fade
    out *= np.minimum(1, np.minimum(t / 1.5, (DUR - t) / 2.5))
    out = out / max(1e-9, np.abs(out).max()) * 0.8
    pcm = (out * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


if __name__ == "__main__":
    wav = OUT + ".wav"
    audio(wav)
    if len(sys.argv) > 4:  # preview stills: python3 render.py a b out t1 t2 ...
        for s in sys.argv[4:]:
            frame(float(s)).save(f"{OUT}_{s}.png")
        sys.exit()
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", wav,
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", OUT],
        stdin=subprocess.PIPE)
    total = int(DUR * FPS)
    for i in range(total):
        p.stdin.write(frame(i / FPS).tobytes())
        if i % 150 == 0:
            print(f"frame {i}/{total}", flush=True)
    p.stdin.close()
    p.wait()
    print("done:", OUT)
