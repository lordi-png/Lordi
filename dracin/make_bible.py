"""Character bible: turnaround, viseme (bentuk mulut), transisi ekspresi bertahap, palet.

python3 make_bible.py   ->  bible/bible_liang_chen.png, bible/bible_long_tian.png
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.argv = [sys.argv[0], "_"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import episode1 as E  # noqa: E402
from sprites import split_by_body  # noqa: E402

OUT = os.path.join(E.HERE, "bible")
os.makedirs(OUT, exist_ok=True)
FD = "/usr/share/fonts/truetype/crosextra/"
FT = ImageFont.truetype(FD + "Caladea-Bold.ttf", 54)
FH = ImageFont.truetype(FD + "Caladea-Bold.ttf", 32)
FS = ImageFont.truetype(FD + "Carlito-Regular.ttf", 22)
FC = ImageFont.truetype(E.CJK, 40)
BG, INK, GOLD = (14, 16, 30), (235, 230, 220), (230, 190, 110)
VIS = [("Diam", (0.0, .5)), ("M/B/P", (0.04, .5)), ("A", (1.0, .6)), ("I", (.35, 1.0)),
       ("U", (.4, 0.0)), ("E", (.55, .85)), ("O", (.75, .15))]
STEPS = [("Netral", {}), ("", {"brow": -.3}), ("", {"brow": -.6, "squint": .3}),
         ("Marah", {"brow": -.9, "squint": .55}), ("", {"brow": -.2, "smile": .2}),
         ("", {"smile": .5, "brow": .1}), ("Tersenyum", {"smile": .9, "brow": .2})]


def fit(im, w, h, bg=(26, 28, 46)):
    c = Image.new("RGB", (w, h), bg)
    s = min(w / im.width, h / im.height)
    r = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    if r.mode == "RGBA":
        c.paste(r, ((w - r.width) // 2, h - r.height), r)
    else:
        c.paste(r, ((w - r.width) // 2, (h - r.height) // 2))
    return c


def palette(im, k=6):
    a = np.asarray(im.convert("RGB").resize((80, 80))).reshape(-1, 3)
    q = Image.fromarray(a.reshape(80, 80, 3)).quantize(k, method=Image.Quantize.MEDIANCUT)
    p = q.getpalette()[:k * 3]
    return [tuple(p[i:i + 3]) for i in range(0, k * 3, 3)]


def sheet(name, cjk, role, turn, turn_lbl, rig, crop, vis_fn, steps_fn, notes, pal_src, steps=None):
    steps = steps or STEPS
    Wd = 1800
    c = Image.new("RGB", (Wd, 2300), BG)
    d = ImageDraw.Draw(c)
    d.text((60, 40), name, font=FT, fill=GOLD)
    d.text((60 + d.textlength(name, font=FT) + 20, 50), cjk, font=FC, fill=GOLD)
    d.text((62, 110), role, font=FS, fill=INK)
    y = 170
    d.text((60, y), "1. TURNAROUND — tampak depan · samping · belakang (kepala sampai kaki)", font=FH, fill=INK)
    y += 50
    tw = (Wd - 120 - 20 * (len(turn) - 1)) // len(turn)
    for i, (im, lb) in enumerate(zip(turn, turn_lbl)):
        c.paste(fit(im, tw, 620), (60 + i * (tw + 20), y))
        d.text((60 + i * (tw + 20) + 10, y + 628), lb, font=FS, fill=GOLD)
    y += 680
    d.text((60, y), "2. BENTUK MULUT UNTUK DIALOG (viseme, dari rig lip-sync)", font=FH, fill=INK)
    y += 50
    vw = (Wd - 120 - 10 * 6) // 7
    for i, (lb, m) in enumerate(VIS):
        c.paste(fit(vis_fn(m).crop(crop), vw, 230), (60 + i * (vw + 10), y))
        d.text((60 + i * (vw + 10) + 8, y + 236), lb, font=FS, fill=GOLD)
    y += 290
    d.text((60, y), "3. EKSPRESI BERUBAH BERTAHAP (interpolasi parameter, bukan potongan)", font=FH, fill=INK)
    y += 50
    for i, (lb, ex) in enumerate(steps):
        c.paste(fit(steps_fn(ex).crop(crop), vw, 230), (60 + i * (vw + 10), y))
        d.text((60 + i * (vw + 10) + 8, y + 236), lb or "→", font=FS, fill=GOLD)
    y += 290
    d.text((60, y), "4. PALET & STANDAR KONSISTENSI", font=FH, fill=INK)
    y += 50
    for i, col in enumerate(palette(pal_src)):
        d.rounded_rectangle((60 + i * 110, y, 160 + i * 110, y + 70), 10, fill=col)
        d.text((60 + i * 110, y + 76), "#%02X%02X%02X" % col, font=FS, fill=INK)
    yy = y
    for ln in notes:
        d.text((760, yy), "• " + ln, font=FS, fill=INK)
        yy += 32
    return c.crop((0, 0, Wd, max(y + 130, yy + 30)))


L = E.REF_L
from sprites import cutout  # noqa: E402
turn_L = [cutout(L, b, bright=228, sat=14, keep_frac=0.25) for b in
          ((376, 214, 506, 640), (500, 214, 600, 640), (596, 214, 732, 640))]
LR = E.LIANG_RIG
sheet_L = sheet(
    "LIANG CHEN", "梁辰", "Pangeran / Raja Muda · 28 th · 185 cm · Penguasa Pedang Dewa (Tian Shen Jian)",
    turn_L, ["Depan", "Samping", "Belakang"], LR, (300, 420, 700, 760),
    lambda m: LR.frame(1.0, m, 0.0, {}), lambda ex: LR.frame(1.0, (0.0, .5), 0.0, ex),
    ["Wajah: rahang tegas, alis tebal lurus, mata hitam tajam; tidak pernah kartunis.",
     "Rambut: hitam panjang sampai pinggang, sanggul atas + mahkota perak-emas.",
     "Pakaian: jubah biru tua berlapis, sulaman naga emas, sabuk hitam-emas.",
     "Proporsi: 8 kepala; bahu lebar; postur tegak & tenang.",
     "Animasi wajib: napas, rambut berangin, kepala ikut bicara, kedip 2,5–5 dtk.",
     "Ekspresi selalu ditransisikan 0,6–1,0 dtk (tidak meloncat)."],
    E.LIANG_PORTRAIT)
sheet_L.save(os.path.join(OUT, "bible_liang_chen.png"))

DR = E.DHEAD_RIG
turn_D = [E.REF_D.crop((418, 0, 832, 632)), E.dtile(3, 3), E.dtile(3, 0), E.dtile(4, 5)]
sheet_D = sheet(
    "LONG TIAN", "龙天", "Naga Kuno Penguasa Langit · ±10.000 th · ±200 m · Petir biru-ungu",
    turn_D, ["Badan penuh (3/4)", "Depan", "Samping", "Siluet terbang"], DR, (300, 400, 756, 900),
    lambda m: DR.frame(1.0, m, 0.0, {}), lambda ex: DR.frame(1.0, (ex.get("o", 0), .5), 0.0, {}),
    ["Sisik putih-perak dengan pinggiran emas; perut berlapis biru gelap.",
     "Tanduk & cakar emas, surai putih panjang, mata merah menyala.",
     "Gerak tubuh meliuk seperti ular (sway rig), surai selalu tertiup.",
     "Bicara: rahang membuka mengikuti amplitudo suara + gema.",
     "Skala: kepala ≥ 3× tinggi Liang Chen di setiap adegan bersama.",
     "Aura petir hanya biru-ungu; emas khusus Mutiara Naga."],
    E.DRAGON_HEAD,
    [("Tenang", {"o": 0}), ("", {"o": .15}), ("", {"o": .3}), ("Menggeram", {"o": .5}), ("", {"o": .7}),
     ("", {"o": .85}), ("Mengaum", {"o": 1.0})])
sheet_D.save(os.path.join(OUT, "bible_long_tian.png"))
print("ok")
