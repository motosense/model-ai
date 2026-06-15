import io
import time
import logging
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("motosense")

KERAS_PATH  = "models/sequential/keras/yamnet_sequential.h5"
SCALER_PATH = "models/sequential/scaler/yamnet_scaler.joblib"

TARGET_SR = 16_000
CLASSES   = [
    "Clutch-Shoe",
    "Conecting-Rod",
    "Drive-Belt",
    "Piston",
    "Tensioner",
    "Slider",
    "Roller",
    "Face-Drive",
]

yamnet_model  = None
keras_model   = None
scaler        = None

app = FastAPI(
    title="MotoSense API",
    description=(
        "Klasifikasi kerusakan mesin motor dari rekaman audio.\n\n"
        "Upload file `.wav` / `.mp3` / `.m4a` → API mengembalikan "
        "prediksi kelas kerusakan beserta confidence score."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def load_models():
    global yamnet_model, keras_model, scaler

    logger.info("Memuat YAMNet dari TF-Hub …")
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    logger.info("YAMNet berhasil dimuat ✓")

    logger.info("Memuat Keras H5 model dari: %s", KERAS_PATH)
    if not Path(KERAS_PATH).exists():
        raise FileNotFoundError(f"Keras model tidak ditemukan: {KERAS_PATH}")
    keras_model = tf.keras.models.load_model(KERAS_PATH)
    logger.info(
        "Keras model dimuat ✓  | input: %s  output: %s",
        keras_model.input_shape,
        keras_model.output_shape,
    )

    logger.info("Memuat scaler dari: %s", SCALER_PATH)
    if not Path(SCALER_PATH).exists():
        raise FileNotFoundError(f"Scaler tidak ditemukan: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    logger.info("Scaler berhasil dimuat ✓")


class ClassScore(BaseModel):
    label: str
    probability: float


class PredictionResponse(BaseModel):
    filename:        str
    predicted_class: str
    confidence:      float
    all_scores:      list[ClassScore]
    inference_ms:    float


class HealthResponse(BaseModel):
    status:  str
    classes: list[str]
    model:   str


def preprocess_audio(audio_bytes: bytes) -> np.ndarray:
    wav, _ = librosa.load(io.BytesIO(audio_bytes), sr=TARGET_SR, mono=True)
    wav, _ = librosa.effects.trim(wav, top_db=30)
    wav_norm = wav / (np.max(np.abs(wav)) + 1e-8)
    _, embeddings, _ = yamnet_model(wav_norm)
    emb_mean = np.mean(embeddings.numpy(), axis=0)
    return emb_mean


def run_inference(embedding: np.ndarray) -> tuple[str, float, list[ClassScore]]:
    scaled = scaler.transform(embedding.reshape(1, -1)).astype(np.float32)
    preds  = keras_model.predict(scaled, verbose=0)[0]
    idx    = int(np.argmax(preds))
    predicted_class = CLASSES[idx]
    confidence      = float(preds[idx])
    all_scores = [
        ClassScore(label=CLASSES[i], probability=float(np.clip(p, 0.0, 1.0)))
        for i, p in enumerate(preds)
    ]
    return predicted_class, confidence, all_scores


ACCEPTED_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}


@app.get("/", response_model=HealthResponse, tags=["Info"])
async def health():
    all_ready = all(x is not None for x in [yamnet_model, keras_model, scaler])
    return HealthResponse(
        status="ready" if all_ready else "loading",
        classes=CLASSES,
        model="YAMNet + Sequential Keras H5",
    )


@app.get("/classes", tags=["Info"])
async def get_classes():
    return {
        "num_classes": len(CLASSES),
        "classes": CLASSES,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    file: UploadFile = File(..., description="File audio mesin motor (.wav / .mp3 / .m4a)"),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ACCEPTED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Format '{ext}' tidak didukung. Gunakan: {sorted(ACCEPTED_EXT)}",
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="File kosong.")

    if any(x is None for x in [yamnet_model, keras_model, scaler]):
        raise HTTPException(status_code=503, detail="Model belum selesai dimuat, coba lagi sebentar.")

    try:
        t0 = time.perf_counter()
        embedding = preprocess_audio(audio_bytes)
        pred_class, conf, all_scores = run_inference(embedding)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Inference error untuk file: %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Inference gagal: {exc}") from exc

    logger.info(
        "✓ %s → %s (%.1f%%)  [%.0f ms]",
        file.filename, pred_class, conf * 100, elapsed_ms,
    )

    return PredictionResponse(
        filename=file.filename or "unknown",
        predicted_class=pred_class,
        confidence=round(conf, 4),
        all_scores=all_scores,
        inference_ms=round(elapsed_ms, 1),
    )
