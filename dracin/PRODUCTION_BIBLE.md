# LIANG CHEN vs LONG TIAN — Bible Produksi

Serial animasi Dracin vertikal (9:16, 720×1280, 60 fps). Gaya semi-realistis dan sinematik,
seperti adegan drama Tiongkok.

## 1. Konsep seri

| | |
|---|---|
| Judul seri | **Tian Shen Jian — Pedang Dewa Langit** |
| Logline | Sepuluh ribu tahun lalu, naga kuno Long Tian disegel di bawah Istana Dewa. Saat segel retak, raja muda Liang Chen harus naik dari pendekar pedang menjadi Penguasa Pedang Tertinggi sebelum langit runtuh. |
| Tema | Tanggung jawab seorang pemimpin dibandingkan kekuatan yang tak terbatas. |
| Arena utama | Tian Shen Dian: istana melayang di atas awan, air terjun, gerbang emas, formasi kuno di lantai, langit badai, dan bulan besar. |
| Format | Episode 2–3 menit, masing-masing dengan satu pertarungan 60 detik. |

**Busur 5 episode (tingkatan pedang Liang Chen):**
1. *Kebangkitan Naga Kuno*: Pendekar Pedang. Liang Chen menahan Long Tian, lalu naga itu mundur. **Sudah dibuat.**
2. *Gerbang Emas Runtuh*: Ahli Pedang. Long Tian menyerang kota, dan Liang Chen kalah untuk pertama kalinya.
3. *Pedang yang Mendengar Langit*: Dewa Pedang. Liang Chen berlatih di pulau batu melayang.
4. *Badai Sepuluh Ribu Tahun*: Dewa Pedang Langit. Long Tian mengendalikan badai (Storm Control).
5. *Penguasa Pedang Tertinggi*: Liang Chen menyegel ulang Long Tian tanpa membunuhnya.

## 2. Standar karakter

Setiap karakter wajib memiliki:

| Standar | Liang Chen | Long Tian |
|---|---|---|
| Tampak depan, samping, belakang | ✅ (`bible/bible_liang_chen.png`) | ⚠️ ada depan, samping, dan 3/4. **Belum ada tampak belakang** di sumber |
| Full body kepala sampai kaki | ✅ | ✅ (pose melingkar, 3/4) |
| 30 ekspresi wajah | ✅ lembar sumber | ✅ lembar sumber |
| 30 gerakan tubuh | ✅ lembar sumber | ✅ lembar sumber |
| Gerakan mulut untuk dialog | ✅ 7 viseme dari rig (Diam, M/B/P, A, I, U, E, O) | ✅ rahang mengikuti suara |
| Konsistensi wajah, rambut, pakaian, proporsi | ✅ semua adegan memakai lembar yang sama, ditambah aturan di bible | ✅ |

Aturan konsistensi tiap karakter tercantum di lembar bible (bagian 4).

## 3. Standar animasi (diterapkan oleh `rig.py`)

| Gerakan | Implementasi |
|---|---|
| Dada naik-turun saat bernapas | Dada dan bahu dilenturkan dengan siklus 3,6–3,8 detik. Saat berlutut kelelahan, napas dibuat 2,8× lebih dalam. |
| Rambut bergerak mengikuti angin | Gelombang menjalar dari akar ke ujung rambut. Ujung paling bebas, ditambah hembusan acak. Surai naga memakai angin 1,6–2,5×. |
| Kepala sedikit bergerak ketika berbicara | Kepala berputar pelan di sekitar leher, dengan anggukan yang mengikuti suara dialog. |
| Bibir mengikuti dialog | Analisis suara per frame (60 fps): keras-lemahnya suara menentukan bukaan mulut, tinggi-rendahnya bunyi menentukan lebar mulut. Rongga mulut dan gigi digambar di bawahnya. |
| Jari tangan bergerak | Genggaman pedang di potret pahlawan mengencang dan mengendur pelan. |
| Bahu bergerak saat berjalan | Badan atas berayun berlawanan dengan langkah, dan bahu naik-turun bergantian. |
| Ekspresi berubah bertahap | Alis, senyum, dan mata menyipit diinterpolasi selama 0,9 detik antar-ketukan cerita. |
| Kedip | Mata berkedip acak setiap 2,5–5 detik, selama 0,18 detik. |

Kamera bergerak subpiksel dengan percepatan *ease-in-out*. Guncangan kamera memakai *noise* halus, bukan acak per frame.

## 4. Pipeline

```
assets/                  lembar referensi + audio paket
sprites.py               pemisah latar → sprite RGBA
rig.py                   rig 2.5D: napas, rambut, kepala, bibir, kedip, ekspresi, jari, bahu
episode1.py              shot list, FX, kamera, audio mix → MP4
make_bible.py            lembar bible (turnaround, viseme, transisi ekspresi, palet)
```

Render: `python3 episode1.py out.mp4`, dikerjakan paralel di semua core CPU.
Pratinjau still frame: `python3 episode1.py out 12.5 70`.

## 5. Kekurangan yang tersisa

Hal berikut **tidak bisa** dibuat dari aset yang ada. Bagian ini butuh ilustrator atau aset baru:

1. **Tampak belakang Long Tian.** Lembar sumber tidak menyediakannya.
2. **Animasi frame-demi-frame sungguhan.** Contohnya lari, menebas, atau naga meliuk dengan pose antara yang digambar.
   Saat ini pose berganti dengan potongan gambar dan dilenturkan oleh rig.
   Untuk gerak penuh, setiap gerakan butuh 8–12 gambar kunci per detik.
3. **Lip-sync fonem yang tepat.** Rig membaca keras-lemah dan tinggi-rendah suara, bukan teks.
   Bentuk mulut sudah mengikuti suku kata, tetapi belum mengenali fonem per huruf.
4. **Suara dialog.** Suara di paket hasil sintesis dasar dan tidak membawa naskah,
   sehingga teks di layar adalah naskah baru. Rekaman pengisi suara atau naskah asli akan menyamakannya.
5. **Resolusi panel jurus.** Panel di lembar referensi berukuran ±245×160 px dan buram setelah diperbesar.
   Versi resolusi tinggi tiap jurus akan langsung meningkatkan ketajaman.
