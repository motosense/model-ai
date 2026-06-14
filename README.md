# MotoSense - Sistem Klasifikasi Audio Kerusakan Mesin Sepeda Motor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/YAMNet-Audio%20Classification-green.svg" alt="YAMNet">
</p>

## 📋 Deskripsi Proyek

**MotoSense** adalah sistem klasifikasi audio berbasis deep learning yang dirancang untuk mendeteksi dan mengklasifikasikan berbagai jenis kerusakan pada mesin sepeda motor melalui analisis suara. Proyek ini menggunakan **YAMNet** (Yet Another Mobile Network) dari TensorFlow Hub sebagai feature extractor, dikombinasikan dengan classifier seperti **Sequential Neural Network**, **Support Vector Machine (SVM)**, dan **Random Forest**.

### 🎯 Tujuan
- Mendeteksi kerusakan mesin sepeda motor secara otomatis melalui audio
- Mengklasifikasikan 8 jenis kerusakan komponen mesin
- Menyediakan model yang dapat diimplementasikan pada perangkat mobile (TFLite)

---

## 🔧 Jenis Kerusakan yang Dapat Dideteksi

Sistem ini dapat mengidentifikasi **8 kategori kerusakan** pada komponen mesin:

1. **Clutch-Shoe** - Kerusakan kampas kopling
2. **Connecting-Rod** - Kerusakan stang seher
3. **Drive-Belt** - Kerusakan van belt/sabuk CVT
4. **Piston** - Kerusakan piston
5. **Tensioner** - Kerusakan tensioner
6. **Slider** - Kerusakan slider CVT
7. **Roller** - Kerusakan roller CVT
8. **Face-Drive** - Kerusakan face pulley/drive face

---

## 🏗️ Arsitektur Sistem

### Model Pipeline
```
Audio Input (WAV/MP3/M4A)
    ↓
Preprocessing (16kHz, 2s duration)
    ↓
YAMNet Feature Extraction (1024-dim embeddings)
    ↓
Classifier (Sequential/SVM/Random Forest)
    ↓
Predicted Class + Confidence Score
```

### Teknologi Utama
- **YAMNet**: Pre-trained audio event detection model dari Google
- **TensorFlow/Keras**: Deep learning framework
- **Librosa**: Audio processing dan feature extraction
- **Audiomentations**: Data augmentation untuk audio
- **Scikit-learn**: Machine learning classifiers (SVM, Random Forest)

---

## 📦 Instalasi

### Prerequisites
- Python 3.8 atau lebih tinggi
- pip (Python package manager)
- ffmpeg (untuk audio processing)

### Langkah Instalasi

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/MotoSense-motorcycle-engine-audio-classifier.git
cd MotoSense-motorcycle-engine-audio-classifier
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

Atau install secara manual:
```bash
pip install tensorflow tensorflow-hub
pip install librosa soundfile
pip install audiomentations
pip install scikit-learn
pip install pandas numpy matplotlib seaborn
pip install jupyter ipywidgets
```

3. **Install FFmpeg** (jika belum terinstall)
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **Windows**: Download dari [ffmpeg.org](https://ffmpeg.org/download.html)

---

## 📂 Struktur Dataset

```
dataset/
├── part-rusak/              # Dataset audio mentah
│   ├── Clutch-Shoe/
│   ├── Conecting-Rod/
│   ├── Drive-Belt/
│   ├── Piston/
│   ├── Tensioner/
│   ├── Slider/
│   ├── Roller/
│   └── Face-Drive/
├── rename/                  # File audio yang sudah direname
├── preprocessed/            # Audio setelah preprocessing
├── augmented/               # Audio hasil augmentasi (target: 1000 file/kelas)
└── split/                   # Dataset split (train/validation/test)
    ├── train/
    ├── validation/
    └── test/
```

---

## 🚀 Cara Menggunakan

### 1. Persiapan Data

Letakkan file audio kerusakan mesin di folder `dataset/part-rusak/` sesuai dengan kategori masing-masing.

### 2. Training Model

Buka dan jalankan notebook sesuai kebutuhan:

#### a. **YAMNet + Sequential Neural Network**
```bash
jupyter notebook "YAMNet aug-split/YAMNET + SVM + RF/MotoSense_YAMNet.ipynb"
```

Atau gunakan helper notebook:
```bash
jupyter notebook "helper/kalsisfikasi_audio.ipynb"
```

### 3. Pipeline Training

Notebook akan menjalankan pipeline berikut:

1. **Rename Files**: Standarisasi penamaan file
2. **Exploratory Data Analysis (EDA)**: Analisis durasi, sample rate
3. **Preprocessing**: 
   - Resample ke 16kHz
   - Segmentasi audio (2 detik dengan overlap)
   - Normalisasi
4. **Data Augmentation**:
   - Gaussian Noise
   - Pitch Shift
   - Time Stretch
   - Low/Band Pass Filter
   - Clipping Distortion
5. **Feature Extraction**: YAMNet embeddings (1024 dimensi)
6. **Model Training**: Sequential NN / SVM / Random Forest
7. **Evaluation**: Classification report, confusion matrix
8. **Export**: Model TFLite untuk deployment mobile

### 4. Inferensi

Gunakan widget interaktif di notebook untuk melakukan prediksi:

```python
# Upload audio file melalui widget
# Model akan otomatis melakukan prediksi dan menampilkan hasil
```

---

## 📊 Preprocessing & Augmentasi

### Parameter Preprocessing
```python
TARGET_SR = 16000           # Sample rate YAMNet
DURATION = 2                # Durasi segmen (detik)
TARGET_SAMPLES = 32000      # 16000 * 2
STRIDE_SAMPLES = 24000      # Overlap 1.5 detik
```

### Teknik Augmentasi
- **AddGaussianNoise**: Menambah noise gaussian (SNR: 5-15 dB)
- **PitchShift**: Mengubah pitch (-3 hingga +3 semitones)
- **TimeStretch**: Mempercepat/memperlambat (0.8x - 1.2x)
- **LowPassFilter**: Filter frekuensi tinggi (2000-7000 Hz)
- **BandPassFilter**: Filter pita frekuensi
- **ClippingDistortion**: Menambah distorsi
- **Gain**: Mengubah volume (-12 hingga +12 dB)

**Target**: 1000 sampel per kelas

---

## 🎯 Model Performance

### Konfigurasi Model

#### Sequential Neural Network
```python
Model: Sequential
- Input: YAMNet Embeddings (1024 dim)
- Dense Layer 1: 512 units, ReLU, Dropout(0.3)
- Dense Layer 2: 256 units, ReLU, Dropout(0.3)
- Dense Layer 3: 128 units, ReLU, Dropout(0.2)
- Output: 8 units (softmax)

Optimizer: Adam
Loss: Categorical Crossentropy
Batch Size: 8
Epochs: 100 (with EarlyStopping)
```

#### Support Vector Machine (SVM)
```python
Kernel: RBF
Class Weight: Balanced
StandardScaler: Pre-processing
```

#### Random Forest
```python
n_estimators: 100-200
max_depth: Tuned via GridSearchCV
Class Weight: Balanced
```

---

## 📱 Export ke TensorFlow Lite

Model dapat dikonversi ke TFLite untuk deployment mobile:

```python
# Conversion sudah tersedia di notebook
# Output: yamnet_sequential.tflite
```

File model tersimpan di:
```
models/sequential/tflite/yamnet_sequential.tflite
```

---

## 📈 Evaluasi Model

Model dievaluasi menggunakan:
- **Confusion Matrix**: Visualisasi prediksi vs actual
- **Classification Report**: Precision, Recall, F1-Score per kelas
- **Accuracy**: Overall accuracy
- **ROC Curve**: (opsional)

---

## 🗂️ File Penting

```
├── YAMNet aug-split/
│   └── YAMNET + SVM + RF/
│       ├── MotoSense_YAMNet.ipynb      # Notebook utama
│       ├── dataset/                     # Dataset
│       └── models/                      # Model yang tersimpan
│           ├── sequential/              # Sequential NN
│           │   ├── keras/              # Format .keras
│           │   └── tflite/             # Format .tflite
│           ├── svm/                    # SVM model (joblib)
│           └── random_forest/          # Random Forest model
├── helper/
│   └── kalsisfikasi_audio.ipynb        # Notebook helper/alternatif
├── README.md                            # Dokumentasi ini
└── requirements.txt                     # Dependencies
```

---

## 🔬 Exploratory Data Analysis (EDA)

Dataset asli memiliki karakteristik:

| Kelas           | Jumlah File | Sample Rate | Durasi Rata-rata | Min   | Max   |
|----------------|-------------|-------------|------------------|-------|-------|
| Clutch-Shoe    | 15          | 44100 Hz    | 2.51s           | 1.07s | 4.08s |
| Connecting-Rod | 10          | 44100 Hz    | 2.37s           | 1.62s | 4.21s |
| Drive-Belt     | 20          | 44100 Hz    | 3.55s           | 1.60s | 6.75s |
| Piston         | 6           | 44100 Hz    | 1.74s           | 1.00s | 3.17s |
| Tensioner      | 8           | 44100 Hz    | 2.97s           | 1.59s | 7.41s |
| Slider         | 13          | 44100 Hz    | 3.91s           | 0.84s | 6.51s |
| Roller         | 19          | 44100 Hz    | 3.78s           | 1.53s | 9.15s |
| Face-Drive     | 9           | 44100 Hz    | 3.14s           | 1.20s | 4.82s |

Setelah augmentasi: **1000 sampel per kelas** (8000 total)

---

## 🛠️ Troubleshooting

### Issue: FFmpeg tidak ditemukan
**Solusi**: Install ffmpeg sesuai OS Anda (lihat bagian Instalasi)

### Issue: Out of Memory saat training
**Solusi**: 
- Kurangi `BATCH_SIZE`
- Gunakan GPU jika tersedia
- Reduce augmentation target

### Issue: Model overfitting
**Solusi**:
- Tambah Dropout layers
- Augmentasi lebih agresif
- Kurangi kompleksitas model

---

## 📚 Referensi

1. **YAMNet**: [TensorFlow Hub - YAMNet](https://tfhub.dev/google/yamnet/1)
2. **Audiomentations**: [GitHub - iver56/audiomentations](https://github.com/iver56/audiomentations)
3. **Librosa**: [librosa.org](https://librosa.org/)
4. Paper: *"Transfer Learning for Audio Classification using YAMNet"*

---

## 👥 Tim Pengembang

**Capstone Project - Pijak IBM 2026**

---

## 📄 Lisensi

[Tambahkan lisensi sesuai kebutuhan]

---

## 🙏 Acknowledgments

- Google Research untuk YAMNet pre-trained model
- TensorFlow Team
- IBM Capstone Program

---

## 📞 Kontak

Untuk pertanyaan atau saran, silakan buat issue di repository ini.

---

**Happy Coding! 🚀**
