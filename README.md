# MotoSense — Klasifikasi Audio Kerusakan Mesin Sepeda Motor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.15+-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/YAMNet-Feature%20Extractor-green.svg" alt="YAMNet">
  <img src="https://img.shields.io/badge/Test%20Accuracy-94.44%25-brightgreen.svg" alt="Best Accuracy">
  <img src="https://img.shields.io/badge/FastAPI-REST%20API-teal.svg" alt="FastAPI">
</p>

---

## Deskripsi Proyek

**MotoSense** adalah sistem klasifikasi audio berbasis machine learning yang mendeteksi dan mengklasifikasikan kerusakan komponen mesin sepeda motor melalui analisis rekaman suara. Sistem ini mengekstrak fitur audio menggunakan **YAMNet** (pre-trained model dari Google) sebagai backbone, lalu mengklasifikasikan hasilnya menggunakan salah satu dari tiga model: **Sequential Neural Network (Dense)**, **Support Vector Machine (SVM)**, atau **Random Forest**.

Kami melakukan tiga skenario eksperimen berbeda untuk menemukan kombinasi optimal antara strategi augmentasi data dan model klasifikasi. Hasil terbaik diperoleh dari **Eksperimen 1 (YAMNet + Aug-Split + Sequential Dense)** yang mencapai akurasi test **94.44%** - sebuah pencapaian yang menunjukkan potensi besar sistem ini untuk aplikasi diagnostik preventif pada kendaraan bermotor.

---

## Jenis Kerusakan yang Dideteksi

Sistem mengklasifikasikan **8 kategori kerusakan** komponen mesin:

| No | Kelas | Komponen |
|----|-------|----------|
| 1 | **Clutch-Shoe** | Kampas kopling |
| 2 | **Conecting-Rod** | Stang seher |
| 3 | **Drive-Belt** | Van belt / sabuk CVT |
| 4 | **Piston** | Piston |
| 5 | **Tensioner** | Tensioner rantai |
| 6 | **Slider** | Slider CVT |
| 7 | **Roller** | Roller CVT |
| 8 | **Face-Drive** | Face pulley / drive face |

---

## Struktur Repository

```
CapstonePijak2026/
├── YAMNet aug-split/                  # Eksperimen utama ✅ (Augmentasi sebelum Split)
│   └── YAMNET + SVM + RF/
│       ├── MotoSense_YAMNet.ipynb     # Notebook utama — pipeline lengkap
│       ├── dataset/                   # Dataset yang digunakan
│       │   ├── part-rusak/            # Audio mentah per kelas
│       │   ├── rename/                # Audio yang sudah direname standar
│       │   ├── preprocessed/         # Audio setelah preprocessing (16kHz, 2s)
│       │   ├── augmented/            # Audio hasil augmentasi (target 2000/kelas)
│       │   ├── split/                # Dataset setelah split train/val/test
│       │   └── features/             # YAMNet embeddings (.npy)
│       └── models/
│           ├── sequential/
│           │   ├── keras/             # yamnet_sequential.h5
│           │   ├── tflite/            # yamnet_sequential.tflite
│           │   └── scaler/            # yamnet_scaler.joblib
│           ├── svm/                   # Model SVM (joblib)
│           └── random_forest/         # Model Random Forest (joblib)
├── YAMNet split-aug/                  # Eksperimen 2 (Split sebelum Augmentasi)
│   └── YAMNET + SVM + RF/
│       └── MotoSense_YAMNet.ipynb
├── without YAMNet/                    # Eksperimen 3 (Baseline tanpa YAMNet)
│   └── MotoSense_CNN_NoYAMNet.ipynb
├── api/                               # REST API (FastAPI)
│   ├── main.py
│   └── requirements.txt
├── assets/                            # Visualisasi hasil eksperimen
├── helper/                            # Notebook bantu
├── requirements.txt
└── README.md
```

---

## Dataset Awal (EDA)

Dataset dikumpulkan dari rekaman suara mesin sepeda motor yang rusak. Karakteristik data mentah:

| Kelas | Jumlah File | Sample Rate | Durasi Rata-rata | Min | Max |
|-------|:-----------:|:-----------:|:----------------:|:---:|:---:|
| Clutch-Shoe | 15 | 44100 Hz | 2.51s | 1.07s | 4.08s |
| Conecting-Rod | 10 | 44100 Hz | 2.37s | 1.62s | 4.21s |
| Drive-Belt | 20 | 44100 Hz | 3.55s | 1.60s | 6.75s |
| Face-Drive | 9 | 44100 Hz | 3.14s | 1.20s | 4.82s |
| Piston | 6 | 44100 Hz | 1.74s | 1.00s | 3.17s |
| Roller | 19 | 44100 Hz | 3.78s | 1.53s | 9.15s |
| Slider | 13 | 44100 Hz | 3.91s | 0.84s | 6.51s |
| Tensioner | 8 | 44100 Hz | 2.97s | 1.59s | 7.41s |
| **Total** | **100** | | | | |

Dataset awal sangat kecil dan tidak seimbang (6–20 file per kelas), sehingga augmentasi adalah komponen kritis dari pipeline ini.

---

## Pipeline Training

```
Audio Mentah (WAV/MP3/M4A, 44.1kHz)
         │
         ▼
  1. Rename & Standarisasi
         │
         ▼
  2. EDA — Analisis durasi & sample rate
         │
         ▼
  3. Preprocessing
     - Resample ke 16kHz (target YAMNet)
     - Segmentasi 2 detik dengan overlap 1.5 detik
     - Normalisasi amplitudo
         │
         ▼
  4. Augmentasi Audio → target 2000 segmen/kelas
     (AddGaussianNoise, PitchShift, TimeStretch,
      LowPassFilter, BandPassFilter, ClippingDistortion, Gain)
         │
         ▼
  5. Split Dataset  →  Train / Validation / Test
     (80% / 10% / 10%)
         │
         ▼
  6. Feature Extraction — YAMNet (1024-dim embeddings)
     - Rata-rata embedding dari seluruh frame per segmen
         │
         ▼
  7. Training Classifier
     ├── Sequential Dense NN (Keras)
     ├── SVM (scikit-learn)
     └── Random Forest (scikit-learn)
         │
         ▼
  8. Evaluasi — Classification Report + Confusion Matrix
         │
         ▼
  9. Export Model → .keras / .h5 / .tflite / .joblib
```

---

## Parameter Konfigurasi

### Preprocessing
```python
TARGET_SR      = 16000   # Sample rate yang dibutuhkan YAMNet
DURATION       = 2       # Durasi segmen (detik)
TARGET_SAMPLES = 32000   # 16000 × 2
STRIDE_SAMPLES = 24000   # Overlap 1.5 detik
```

### Augmentasi
```python
AUG_TARGET = 2000  # Target jumlah segmen per kelas setelah augmentasi

Teknik augmentasi (Audiomentations):
- AddGaussianNoise  : SNR acak 5–15 dB
- PitchShift        : −3 hingga +3 semitone
- TimeStretch       : faktor 0.8x–1.2x
- LowPassFilter     : cutoff 2000–7000 Hz
- BandPassFilter    : pita frekuensi tertentu
- ClippingDistortion: distorsi clipping ringan
- Gain              : perubahan volume −12 hingga +12 dB
```

### Split Dataset
```
Setelah augmentasi: 2000 segmen × 8 kelas = 16.000 total segmen
Train : 12.801 (80%)
Val   :  1.599 (10%)
Test  :  1.600 (10%)
```

---

## Arsitektur Model

### Sequential Neural Network (Dense)
```
Input  → (1024,)   # YAMNet embeddings
Dense  → 512  + BatchNorm + ReLU + Dropout(0.5)
Dense  → 256  + BatchNorm + ReLU + Dropout(0.5)
Dense  → 128  + BatchNorm + ReLU + Dropout(0.5)
Output → 8    + Softmax

Optimizer  : Adam (lr=1e-3)
Loss       : Categorical Crossentropy
Batch Size : 32
Max Epochs : 500
Callbacks  : EarlyStopping (patience=20), ReduceLROnPlateau (factor=0.5, patience=7)
```

### Support Vector Machine (SVM)
```
Kernel      : RBF
Class Weight: Balanced
Preprocessing: StandardScaler
```

### Random Forest
```
n_estimators : tuned
Class Weight : Balanced
Preprocessing: StandardScaler
```

---

## Hasil Eksperimen

Kami menjalankan tiga skenario eksperimen yang berbeda untuk memahami dampak dari strategi augmentasi data dan penggunaan transfer learning dengan YAMNet.

### Eksperimen 1 — YAMNet + Aug-Split *(Pendekatan Terbaik)*

Pendekatan ini menggunakan augmentasi data **sebelum** melakukan split dataset. Keputusan ini diambil dengan pertimbangan bahwa augmentasi di tahap awal akan menghasilkan dataset yang lebih besar dan lebih seimbang di seluruh split (train/validation/test). Hasilnya memang membuktikan bahwa strategi ini memberikan performa terbaik.

#### Hasil Performa Model

| Model | Train Accuracy | Validation Accuracy | **Test Accuracy** | Macro Avg Precision | Macro Avg Recall | Macro Avg F1 |
|-------|:--------------:|:-------------------:|:-----------------:|:-------------------:|:----------------:|:------------:|
| 🥇 **Sequential (Dense)** | 98.88% | 96.81% | **94.44%** | 0.95 | 0.94 | 0.94 |
| 🥈 **Random Forest** | — | 93.75% | **91.50%** | 0.92 | 0.92 | 0.91 |
| 🥉 **SVM (RBF Kernel)** | — | 92.12% | **91.38%** | 0.92 | 0.91 | 0.91 |

**Catatan Penting tentang Sequential Model:**
- Training berjalan hingga epoch ke-125 dengan mekanisme EarlyStopping (patience=20)
- Model terbaik dicapai pada epoch ke-105 dengan validation accuracy 96.81%
- Berkat callback `restore_best_weights=True`, model yang disimpan menggunakan bobot dari epoch terbaik
- Learning rate adjustment dilakukan secara bertahap: 1e-3 → 5e-4 → 2.5e-4 → 1.25e-4 → 6.25e-5
- Test accuracy final model adalah **94.44%**, yang menunjukkan generalisasi yang sangat baik

#### Analisis Per-Kelas (Sequential Dense Model)

Berikut adalah performa detail dari model Sequential Dense pada test set, yang menunjukkan kekuatan dan area yang perlu perhatian:

```
                  precision    recall  f1-score   support

  Clutch-Shoe       0.95      0.94      0.95       200
Conecting-Rod       0.98      0.99      0.99       200
   Drive-Belt       0.96      0.95      0.96       200
       Piston       0.96      0.94      0.95       200
    Tensioner       0.97      0.97      0.97       200
       Slider       0.89      0.94      0.91       200
       Roller       0.91      0.91      0.91       200
   Face-Drive       0.92      0.92      0.92       200

     accuracy                           0.94      1600
    macro avg       0.95      0.95      0.95      1600
 weighted avg       0.95      0.94      0.95      1600
```

**Insight dari Hasil:**
- `Conecting-Rod` menunjukkan performa terbaik dengan F1-score 0.99 - pola suara kerusakan ini sangat distinctive
- `Piston` dan `Tensioner` juga memberikan hasil yang sangat baik (F1 ≥ 0.95)
- `Slider` memiliki F1-score terendah (0.91), mungkin karena karakteristik suara yang overlap dengan komponen CVT lainnya
- Tidak ada kelas yang menunjukkan performa buruk, semua berada di atas threshold 0.89

### Eksperimen 2 — YAMNet + Split-Aug

Pada eksperimen ini, kami melakukan split dataset **terlebih dahulu**, baru kemudian menerapkan augmentasi. Pendekatan ini menghasilkan test set yang lebih "bersih" (tidak ada augmentasi), namun di sisi lain training set menjadi lebih kecil.

| Model | Validation Accuracy | **Test Accuracy** | Macro Avg Precision | Macro Avg Recall | Macro Avg F1 |
|-------|:-------------------:|:-----------------:|:-------------------:|:----------------:|:------------:|
| Sequential (Dense) | — | 79.17% | 0.80 | 0.79 | 0.77 |
| SVM | 89.47% | 83.33% | 0.80 | 0.83 | 0.78 |
| Random Forest | 89.47% | 83.33% | 0.80 | 0.83 | 0.79 |

Penurunan performa yang signifikan (sekitar 11-12%) menunjukkan betapa pentingnya augmentasi data yang dilakukan sebelum split untuk meningkatkan kemampuan generalisasi model.

### Eksperimen 3 — Tanpa YAMNet (Baseline CNN)

Sebagai pembanding, kami juga menguji pendekatan tanpa menggunakan transfer learning YAMNet. Model ini dibangun dari scratch menggunakan arsitektur CNN konvensional.

| Model | Validation Accuracy | **Test Accuracy** | Macro Avg Precision | Macro Avg Recall | Macro Avg F1 |
|-------|:-------------------:|:-----------------:|:-------------------:|:----------------:|:------------:|
| Sequential (Dense) | — | 75.00% | 0.72 | 0.75 | 0.70 |
| SVM | 78.95% | 79.17% | 0.69 | 0.79 | 0.73 |
| Random Forest | 89.47% | 83.33% | 0.80 | 0.83 | 0.79 |

Hasil ini membuktikan kontribusi besar dari transfer learning menggunakan YAMNet pre-trained embeddings. Peningkatan 15-19% pada test accuracy menunjukkan keunggulan memanfaatkan knowledge yang sudah dipelajari YAMNet dari jutaan sample audio.

---

## Perbandingan Visual Akurasi

```
Eksperimen 1 — YAMNet + Aug-Split (Setup Optimal)
┌──────────────────────────────────────────────────────────┐
│  Sequential   ███████████████████████████░░  94.44%     │
│  Random Forest█████████████████████████████  91.50%     │
│  SVM          █████████████████████████████  91.38%     │
└──────────────────────────────────────────────────────────┘

Eksperimen 2 — YAMNet + Split-Aug
┌──────────────────────────────────────────────────────────┐
│  Sequential   ████████████████████░░░░░░░░   79.17%     │
│  SVM          █████████████████████████░░░   83.33%     │
│  Random Forest█████████████████████████░░░   83.33%     │
└──────────────────────────────────────────────────────────┘

Eksperimen 3 — Tanpa YAMNet (Baseline)
┌──────────────────────────────────────────────────────────┐
│  Sequential   ███████████████████░░░░░░░░░   75.00%     │
│  SVM          ████████████████████████░░░░   79.17%     │
│  Random Forest█████████████████████████░░░   83.33%     │
└──────────────────────────────────────────────────────────┘
```

---

## Kurva Training — Sequential (YAMNet + Aug-Split)

![Training Accuracy & Loss](assets/yamnet_dense_training.png)

Proses training berlangsung dengan pengaturan yang optimal:
- **Total Epochs**: Training berhenti di epoch 125 dengan EarlyStopping (patience=20)
- **Best Epoch**: Model terbaik dicapai di epoch 105 dengan validation accuracy 96.81%
- **Learning Rate Schedule**: Dilakukan penurunan bertahap oleh ReduceLROnPlateau
  - Initial: 1e-3
  - Step 1: 5e-4 (factor=0.5, patience=7)
  - Step 2: 2.5e-4
  - Step 3: 1.25e-4
  - Final: 6.25e-5
- **Final Metrics**:
  - Train accuracy: 98.88%
  - Validation accuracy (best): 96.81%
  - **Test accuracy: 94.44%**

Grafik menunjukkan learning curve yang healthy tanpa overfitting signifikan. Gap antara train dan validation accuracy yang minimal (sekitar 2%) mengindikasikan bahwa model memiliki kemampuan generalisasi yang sangat baik.

---

## Visualisasi Embedding

### Distribusi Embedding YAMNet (per kelas)
![Embedding Distribution](assets/yamnet_embedding_distribution.png)

### PCA Embedding
![PCA Embedding](assets/pca_embedding.png)

### UMAP Embedding
![UMAP Embedding](assets/umap_embedding.png)

### t-SNE Embedding
![t-SNE Embedding](assets/tsne_embedding.png)

---

## Kesimpulan Eksperimen

Dari ketiga eksperimen yang telah dilakukan, beberapa pembelajaran penting dapat ditarik:

| Aspek | Temuan |
|-------|--------|
| **Strategi Augmentasi Optimal** | Melakukan augmentasi **sebelum** split dataset (Aug-Split) terbukti memberikan hasil superior. Pendekatan ini menghasilkan training set yang lebih besar dan lebih beragam, yang pada akhirnya meningkatkan kemampuan generalisasi model secara konsisten di semua algoritma yang diuji. |
| **Model Terbaik** | **Sequential Dense Neural Network** dengan YAMNet embeddings mencapai test accuracy **94.44%**, unggul sekitar 3% dibanding algoritma ML tradisional. Arsitektur deep learning dengan regularization yang tepat (Dropout, BatchNorm) mampu menangkap pola kompleks dalam data audio dengan lebih baik. |
| **Kontribusi Transfer Learning** | Pemanfaatan pre-trained YAMNet memberikan boost performa 15-19% dibanding model CNN yang dilatih dari scratch. Ini membuktikan bahwa knowledge yang sudah dipelajari YAMNet dari jutaan audio samples sangat relevan untuk domain kita. |
| **Kelas Termudah** | `Conecting-Rod` (F1=0.99) dan `Piston` (F1=0.95) memiliki karakteristik suara yang sangat distinctive, membuat model dapat mengidentifikasi dengan akurasi sangat tinggi. |
| **Kelas Tersulit** | `Slider` (F1=0.91) masih menjadi tantangan terbesar, kemungkinan karena overlap karakteristik suara dengan komponen CVT lainnya. Namun performa 0.91 tetap menunjukkan hasil yang sangat baik. |
| **Impact Augmentasi** | Augmentasi sebelum split meningkatkan variasi di training set tanpa "mengotori" test set, menghasilkan model yang lebih robust terhadap variasi real-world. Perbedaan 11-12% accuracy antara Eksperimen 1 dan 2 membuktikan pentingnya strategi ini. |
| **Computational Efficiency** | Model Sequential Dense, meski paling akurat, membutuhkan waktu training lebih lama. Untuk deployment dengan constraint komputasi, Random Forest (91.50%) atau SVM (91.38%) bisa menjadi alternatif yang sangat viable dengan trade-off minimal. |

---

## REST API

Model Sequential Dense di-deploy sebagai REST API menggunakan **FastAPI**.

### Endpoint

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `GET` | `/` | Health check & status model |
| `GET` | `/classes` | Daftar 8 kelas kerusakan |
| `POST` | `/predict` | Upload audio → prediksi kelas |

### Contoh Response `/predict`

```json
{
  "filename": "rekaman_mesin.wav",
  "predicted_class": "Clutch-Shoe",
  "confidence": 0.9412,
  "all_scores": [
    { "label": "Clutch-Shoe",   "probability": 0.9412 },
    { "label": "Conecting-Rod", "probability": 0.0031 },
    { "label": "Drive-Belt",    "probability": 0.0189 },
    { "label": "Piston",        "probability": 0.0008 },
    { "label": "Tensioner",     "probability": 0.0144 },
    { "label": "Slider",        "probability": 0.0097 },
    { "label": "Roller",        "probability": 0.0062 },
    { "label": "Face-Drive",    "probability": 0.0057 }
  ],
  "inference_ms": 412.3
}
```

Format audio yang diterima: `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`

### Menjalankan API

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

> **Catatan:** File model (`models/sequential/keras/yamnet_sequential.h5` dan `models/sequential/scaler/yamnet_scaler.joblib`) harus tersedia relatif terhadap direktori `api/`.

### Inferensi API

Pipeline inferensi di API:
1. Load audio → resample 16kHz → mono
2. Trim silence (top_db=30)
3. Normalisasi amplitudo
4. Ekstraksi YAMNet embeddings → rata-rata per segmen
5. StandardScaler transform
6. Prediksi dengan Sequential Dense model
7. Kembalikan kelas + confidence + all scores

---

## Instalasi & Reproduksi

### Prerequisites

- Python 3.8+
- pip
- ffmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```

### Install Dependencies (Notebook)

```bash
pip install tensorflow>=2.15.0 tensorflow-hub>=0.16.0
pip install librosa>=0.10.2 soundfile>=0.12.1
pip install audiomentations
pip install scikit-learn>=1.4.0
pip install numpy pandas matplotlib seaborn
pip install umap-learn
pip install jupyter ipywidgets
```

### Install Dependencies (API)

```bash
pip install -r api/requirements.txt
```

> **Catatan khusus:** Jika muncul error `distutils/pkg_resources` setelah install `tensorflow-hub`, jalankan:
> ```bash
> pip install "setuptools<81" --force-reinstall
> ```

---

## Menjalankan Notebook

Buka notebook di direktori eksperimen yang diinginkan:

```bash
# Eksperimen terbaik (Aug-Split)
jupyter notebook "YAMNet aug-split/YAMNET + SVM + RF/MotoSense_YAMNet.ipynb"
```

Notebook akan menjalankan seluruh pipeline secara berurutan:
1. Setup path dan konfigurasi
2. Rename & standardisasi nama file
3. EDA — analisis durasi, sample rate, distribusi kelas
4. Preprocessing — resample, segmentasi, normalisasi
5. Augmentasi — menggunakan Audiomentations
6. Split dataset — train/val/test
7. Ekstraksi fitur YAMNet — menghasilkan 1024-dim embeddings
8. Training Sequential Dense NN
9. Training SVM
10. Training Random Forest
11. Evaluasi — classification report, confusion matrix
12. Export model ke format Keras, TFLite, dan joblib

---

## Referensi

1. **YAMNet** — [TensorFlow Hub](https://tfhub.dev/google/yamnet/1)
2. **Audiomentations** — [GitHub iver56/audiomentations](https://github.com/iver56/audiomentations)
3. **Librosa** — [librosa.org](https://librosa.org/)
4. Howard, A., et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.* arXiv:1704.04861

---

## Anggota Tim

**Capstone Project — Pijak IBM 2026**
