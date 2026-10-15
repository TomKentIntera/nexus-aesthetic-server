"""Aesthetic Predictor V2.5 image scoring sidecar.

Wraps https://github.com/discus0434/aesthetic-predictor-v2-5 (SigLIP-based
1–10 aesthetic score). Exposes multipart POST /upload and GET /health for
Laravel (AestheticPredictorClient / ScorePostAestheticJob), and POST /embed for a
SigLIP vector of an image that is not a post (the Emaki rating model's Danbooru
anchors). Model load/score lives in predictor.py so the local pull worker can use
the same in-process path.
"""

from __future__ import annotations

import base64
import os
import struct
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

import predictor

SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "5050"))

app = FastAPI(title="aesthetic-predictor-server")


class ScoreResponse(BaseModel):
    score: float = Field(..., description="Aesthetic score roughly in 1–10 range")
    device: str


class BatchScoreResponse(BaseModel):
    scores: list[float] = Field(..., description="Aesthetic scores, one per uploaded image")
    device: str


@app.on_event("startup")
def startup() -> None:
    predictor.load_model()


@app.post("/upload", response_model=ScoreResponse)
async def upload(file: UploadFile = File(...)) -> ScoreResponse:
    health = predictor.health_payload()
    if health["status"] != "ok":
        raise HTTPException(
            status_code=503,
            detail=health["error"] or "model not loaded",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        # CPU inference is blocking; run it in the threadpool so /health and
        # other requests are not queued behind an in-flight score.
        score = await run_in_threadpool(predictor.score_image_bytes, image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ScoreResponse(
        score=score,
        device=str(health["device"] or "unknown"),
    )


class EmbedResponse(BaseModel):
    siglip: str = Field(..., description="Base64 little-endian float32 SigLIP vector (L2-normalised)")
    dims: int
    model: str


@app.post("/embed", response_model=EmbedResponse)
async def embed(file: UploadFile = File(...)) -> EmbedResponse:
    health = predictor.health_payload()
    if health["status"] != "ok":
        raise HTTPException(status_code=503, detail=health["error"] or "model not loaded")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        vector = await run_in_threadpool(predictor.embed_image_bytes, image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    blob = struct.pack(f"<{len(vector)}f", *vector)
    return EmbedResponse(
        siglip=base64.b64encode(blob).decode("ascii"),
        dims=len(vector),
        model="google/siglip-so400m-patch14-384",
    )


@app.post("/upload/batch", response_model=BatchScoreResponse)
async def upload_batch(files: list[UploadFile] = File(...)) -> BatchScoreResponse:
    health = predictor.health_payload()
    if health["status"] != "ok":
        raise HTTPException(
            status_code=503,
            detail=health["error"] or "model not loaded",
        )

    images_bytes = [await file.read() for file in files]
    if not images_bytes or any(not image_bytes for image_bytes in images_bytes):
        raise HTTPException(status_code=400, detail="empty file")

    try:
        scores = await run_in_threadpool(predictor.score_image_batch, images_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return BatchScoreResponse(
        scores=scores,
        device=str(health["device"] or "unknown"),
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return predictor.health_payload()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
