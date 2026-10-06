"""Local FastAPI service for real-data ForensiScan inference."""
from __future__ import annotations

import csv
import io
import json
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image, ImageOps
from starlette.concurrency import run_in_threadpool

from ml.config import load_config, root_path
from ml.data.face_crop import crop_face, detect_largest_face, load_detectors
from ml.features.extractor import extract_features

ROOT = root_path(".")
STORAGE = ROOT / "backend" / "storage"
DB_PATH = STORAGE / "history.sqlite3"
MAX_BYTES = 10 * 1024 * 1024
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
predictor: "Predictor | None" = None


class Predictor:
    """Load the trained real-data model once and perform CPU inference."""

    def __init__(self) -> None:
        self.model = joblib.load(MODEL_PATH)
        self.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
        self.dnn, self.haar = load_detectors(ROOT / "models" / "face_detector")

    def analyze(self, payload: bytes) -> dict:
        """Decode an image, detect its largest face, and classify the crop."""
        started = time.perf_counter()
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(payload))).convert("RGB")
        rgb = np.asarray(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        box = detect_largest_face(bgr, self.dnn, self.haar)
        warnings: list[str] = []
        if box is None:
            return {"face_detected": False, "verdict": "inconclusive", "fake_probability": None, "confidence": None, "processing_ms": round((time.perf_counter() - started) * 1000), "warnings": ["No face was detected in the image."]}
        if min(box[2], box[3]) < 64:
            warnings.append("Small face; accuracy may be reduced.")
        crop = crop_face(bgr, box)
        features, _ = extract_features(crop)
        probability = float(self.model.predict_proba(features.reshape(1, -1))[0, 1])
        threshold = float(self.metadata.get("decision_threshold", 0.5))
        verdict = "fake" if probability >= threshold else "real"
        confidence = probability if verdict == "fake" else 1.0 - probability
        return {"face_detected": True, "face_box": {"x": box[0], "y": box[1], "w": box[2], "h": box[3]}, "verdict": verdict, "fake_probability": probability, "confidence": confidence, "threshold": threshold, "model_name": self.metadata.get("model_name", "ForensiScan"), "processing_ms": round((time.perf_counter() - started) * 1000), "warnings": warnings}


def db() -> sqlite3.Connection:
    """Open the local history database."""
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize() -> None:
    """Create local storage and history schema."""
    (STORAGE / "uploads").mkdir(parents=True, exist_ok=True)
    (STORAGE / "thumbnails").mkdir(parents=True, exist_ok=True)
    with db() as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS analysis (id TEXT PRIMARY KEY, filename TEXT, created_at TEXT, verdict TEXT, fake_probability REAL, confidence REAL, face_detected INTEGER, result_json TEXT, upload_path TEXT, thumbnail_path TEXT)")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize persistence and load the model if a real artifact exists."""
    global predictor
    initialize()
    predictor = Predictor() if MODEL_PATH.exists() else None
    yield


app = FastAPI(title="ForensiScan API", version="0.1.0", lifespan=lifespan)
config = load_config()
app.add_middleware(CORSMiddleware, allow_origins=config["backend"]["cors_origins"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health() -> dict:
    """Return service and model readiness."""
    return {"status": "ok", "model_loaded": predictor is not None, "model_name": predictor.metadata.get("model_name") if predictor else None, "version": app.version}


@app.get("/api/model/info")
def model_info() -> dict:
    """Return trained model metadata or a clear untrained state."""
    if not METADATA_PATH.exists():
        return {"model_loaded": False, "message": "Train on real FaceForensics++ data first."}
    return {"model_loaded": predictor is not None, **json.loads(METADATA_PATH.read_text(encoding="utf-8"))}


async def read_upload(file: UploadFile) -> bytes:
    """Read and validate a local image upload."""
    payload = await file.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise HTTPException(422, "Image exceeds the 10 MB limit")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
    except Exception as error:
        raise HTTPException(422, "Upload is not a valid image") from error
    return payload


async def save_result(file: UploadFile, payload: bytes, result: dict) -> dict:
    """Persist an upload, thumbnail, and analysis result."""
    identifier = str(uuid.uuid4())
    upload_path = STORAGE / "uploads" / identifier
    thumbnail_path = STORAGE / "thumbnails" / f"{identifier}.jpg"
    upload_path.write_bytes(payload)
    with Image.open(io.BytesIO(payload)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((256, 256))
        image.save(thumbnail_path, format="JPEG", quality=88)
    result.update({"id": identifier, "filename": file.filename or "upload", "created_at": datetime.now(timezone.utc).isoformat(), "thumbnail_url": f"/api/files/{identifier}/thumbnail"})
    with db() as connection:
        connection.execute("INSERT INTO analysis VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (identifier, result["filename"], result["created_at"], result.get("verdict"), result.get("fake_probability"), result.get("confidence"), int(result.get("face_detected", False)), json.dumps(result), str(upload_path), str(thumbnail_path)))
    return result


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    """Analyze one image using the trained model."""
    if predictor is None:
        raise HTTPException(503, "Model is not loaded. Train on real FaceForensics++ data first.")
    payload = await read_upload(file)
    result = await run_in_threadpool(predictor.analyze, payload)
    return await save_result(file, payload, result)


@app.get("/api/history")
def history(page: int = 1, page_size: int = 20, label: str | None = None, q: str | None = None) -> dict:
    """Return a paginated local history list."""
    clauses, values = [], []
    if label in {"real", "fake", "inconclusive"}:
        clauses.append("verdict = ?"); values.append(label)
    if q:
        clauses.append("filename LIKE ?"); values.append(f"%{q}%")
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with db() as connection:
        rows = connection.execute(f"SELECT result_json FROM analysis{where} ORDER BY created_at DESC LIMIT ? OFFSET ?", (*values, page_size, (page - 1) * page_size)).fetchall()
        total = connection.execute(f"SELECT COUNT(*) FROM analysis{where}", values).fetchone()[0]
    return {"items": [json.loads(row[0]) for row in rows], "page": page, "page_size": page_size, "total": total}


@app.get("/api/history/{identifier}")
def history_item(identifier: str) -> dict:
    """Return one local analysis record."""
    with db() as connection:
        row = connection.execute("SELECT result_json FROM analysis WHERE id = ?", (identifier,)).fetchone()
    if row is None:
        raise HTTPException(404, "Analysis not found")
    return json.loads(row[0])


@app.delete("/api/history/{identifier}")
def delete_history_item(identifier: str) -> dict:
    """Delete one local analysis and its stored files."""
    with db() as connection:
        row = connection.execute("SELECT upload_path, thumbnail_path FROM analysis WHERE id = ?", (identifier,)).fetchone()
        if row is None:
            raise HTTPException(404, "Analysis not found")
        connection.execute("DELETE FROM analysis WHERE id = ?", (identifier,))
    for path in row:
        Path(path).unlink(missing_ok=True)
    return {"deleted": identifier}


@app.get("/api/files/{identifier}/thumbnail")
def thumbnail(identifier: str) -> FileResponse:
    """Serve a stored thumbnail."""
    with db() as connection:
        row = connection.execute("SELECT thumbnail_path FROM analysis WHERE id = ?", (identifier,)).fetchone()
    if not row or not Path(row[0]).exists():
        raise HTTPException(404, "Thumbnail not found")
    return FileResponse(row[0], media_type="image/jpeg")


@app.delete("/api/history")
def clear_history() -> dict:
    """Delete all local analysis records and files."""
    with db() as connection:
        rows = connection.execute("SELECT upload_path, thumbnail_path FROM analysis").fetchall()
        connection.execute("DELETE FROM analysis")
    for row in rows:
        for path in row:
            Path(path).unlink(missing_ok=True)
    return {"deleted": len(rows)}


@app.get("/api/history/export.csv")
def export_history() -> StreamingResponse:
    """Export local history as CSV."""
    with db() as connection:
        rows = connection.execute("SELECT result_json FROM analysis ORDER BY created_at DESC").fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "filename", "created_at", "verdict", "fake_probability", "confidence"])
    for row in rows:
        value = json.loads(row[0]); writer.writerow([value.get(key) for key in ("id", "filename", "created_at", "verdict", "fake_probability", "confidence")])
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=forensiscan-history.csv"})