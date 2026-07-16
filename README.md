# 🚘 Multifinance Loan Process Intelligence Dashboard

Dashboard Streamlit untuk memetakan alur proses bisnis pembiayaan kendaraan
(multifinance) secara end-to-end, mendeteksi bottleneck operasional, dan
merekomendasikan perbaikan yang terukur per departemen — dilengkapi model
Machine Learning untuk memvalidasi temuan secara statistik.

Dibangun dari log operasional **20.000 aplikasi pembiayaan** (Jan–Mar 2026)
yang mencatat 6 titik waktu proses: `T1 Pengajuan → T2 Verifikasi Dokumen →
T3 Analisis Kredit → T4 Survei Lapangan → T5 Persetujuan Akhir → T6 Pencairan Dana`.

## ✨ Fitur

| Tab | Isi |
|---|---|
| 📊 **Ringkasan** | KPI utama, mini "process rail" (visual signature), tren volume & tingkat persetujuan harian, distribusi status & segmen |
| 🔄 **Peta Alur Proses** | Diagram alur proses T1–T6 dengan durasi & departemen pemilik, Sankey diagram volume aplikasi, tabel ringkasan per tahap, box plot sebaran durasi |
| 🔍 **Analisis Bottleneck** | Identifikasi otomatis tahap paling lambat, heatmap wilayah × jenis aset, breakdown segmen, analisis kepatuhan SLA |
| 🤖 **Insight ML** | Model RandomForest untuk mendiagnosis penyebab keterlambatan (feature importance), uji prediktabilitas keputusan akhir, deteksi anomali berbasis z-score per segmen |
| 🎯 **Rekomendasi** | Rencana aksi per departemen dengan estimasi dampak (jam/hari dihemat) dan matriks prioritas dampak vs kompleksitas |
| 🗂️ **Data Explorer** | Tabel data terfilter + fitur durasi turunan, dapat diunduh sebagai CSV |

Semua tab bereaksi terhadap filter global di sidebar (rentang tanggal, wilayah,
jenis aset, jenis nasabah, status aplikasi).

## 🧠 Temuan Kunci dari Data

1. **Bottleneck tunggal yang dominan**: ruas *Analisis Kredit → Survei Lapangan*
   berlangsung rata-rata **~19 jam** — 5–10x lebih lama dari ruas proses lainnya
   (~2–8 jam).
2. **Dua pola independen** menyebabkan hal ini:
   - **Mobil bekas** butuh waktu appraisal ~36 jam di **semua wilayah** (vs ~7,5 jam
     untuk mobil baru/truk) → isu SOP/appraisal nasional.
   - **Wilayah Jawa Timur** lambat untuk **semua jenis aset** (~36 jam) → indikasi
     keterbatasan kapasitas tim survei regional, bukan kompleksitas aset.
3. Model ML mengonfirmasi kedua pola ini sebagai fitur paling berpengaruh
   (feature importance) terhadap durasi tahap tersebut.
4. Keputusan akhir (disetujui/ditolak/dibatalkan) **tidak dapat diprediksi**
   dari segmen atau kecepatan proses — akurasi model setara baseline menebak
   kelas mayoritas, mengindikasikan keputusan ditentukan faktor kelayakan
   kredit individual di luar log operasional ini.

## 🗂️ Struktur Proyek

```
.
├── app.py                # Entry point Streamlit — layout & seluruh tab UI
├── data_pipeline.py       # Loading data, feature engineering, filter, formatting
├── ml_engine.py           # Model diagnosis bottleneck, model keputusan, deteksi anomali
├── ui_kit.py              # Design system: CSS, tema Plotly, komponen "process rail"
├── requirements.txt
├── .streamlit/
│   └── config.toml        # Tema warna Streamlit
└── data/
    └── multifinance_loan_process_logs.csv
```

## 🚀 Menjalankan Secara Lokal

```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501` di browser.

## ☁️ Deploy ke Streamlit Community Cloud

1. Push folder ini ke repo GitHub (lihat langkah di bawah).
2. Buka [share.streamlit.io](https://share.streamlit.io), hubungkan ke repo Anda.
3. Set **Main file path** ke `app.py`.
4. Deploy — Streamlit Cloud otomatis membaca `requirements.txt` dan `.streamlit/config.toml`.

## 📤 Push ke GitHub

```bash
cd multifinance-dashboard
git init
git add .
git commit -m "Initial commit: Multifinance Loan Process Intelligence Dashboard"
git branch -M main
git remote add origin <URL_REPO_GITHUB_ANDA>
git push -u origin main
```

## 📝 Catatan Data & Asumsi

- Ambang SLA siklus penuh (24 jam) dan skor kompleksitas implementasi pada
  matriks prioritas bersifat **asumsi analitis** untuk mendukung diskusi —
  sesuaikan dengan kebijakan internal perusahaan yang sebenarnya.
- Dataset bersifat operasional (bukan finansial/kredit), sehingga model ML
  di sini fokus pada efisiensi proses, bukan penilaian kelayakan kredit.

## 🛠️ Tech Stack

Streamlit · Pandas · NumPy · Plotly · scikit-learn
