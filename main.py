"""Aesthetic Predictor V2.5 image scoring sidecar.

Wraps https://github.com/discus0434/aesthetic-predictor-v2-5 (SigLIP-based
1–10 aesthetic score). Exposes multipart POST /upload and GET /health for
Laravel (AestheticPredictorClient / ScorePostAestheticJob). Model load/score
lives in predictor.py so the local pull worker can use the same in-process path.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

import predictor

SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "5050"))

app = FastAPI(title="aesthetic-predictor-server")


class ScoreResponse(BaseModel):
    score: float = Field(..., description="Aesthetic score roughly in 1–10 range")
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
        score = predictor.score_image_bytes(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ScoreResponse(
        score=score,
        device=str(health["device"] or "unknown"),
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return predictor.health_payload()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
