"""In-process Aesthetic Predictor V2.5 (SigLIP) loader.

Shared by the FastAPI sidecar (`main.py`, Swarm/Laravel jobs) and the local
pull worker (`python/aesthetic-score/score_aesthetic.py`).
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger("aesthetic-predictor")

MODEL_CACHE_DIR = Path(
    os.environ.get("MODEL_CACHE_DIR", str(Path(__file__).resolve().parent / "models"))
).resolve()
SKIP_AUTO_DOWNLOAD = os.environ.get("SKIP_AUTO_DOWNLOAD", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEVICE_ENV = os.environ.get("DEVICE", "").strip().lower()

_model: Any = None
_preprocessor: Any = None
_device: torch.device | None = None
_dtype: torch.dtype | None = None
_load_error: str | None = None


def _resolve_device() -> torch.device:
    if DEVICE_ENV in {"cpu", "cuda", "mps"}:
        if DEVICE_ENV == "cuda" and not torch.cuda.is_available():
            logger.warning("DEVICE=cuda requested but CUDA unavailable; falling back to CPU")
            return torch.device("cpu")
        if DEVICE_ENV == "mps" and not torch.backends.mps.is_available():
            logger.warning("DEVICE=mps requested but MPS unavailable; falling back to CPU")
            return torch.device("cpu")
        return torch.device(DEVICE_ENV)

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_dtype(device: torch.device) -> torch.dtype:
    # bfloat16 is preferred on CUDA; float32 is safer/portable on CPU.
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def _predictor_weights_path() -> Path:
    return MODEL_CACHE_DIR / "aesthetic_predictor_v2_5.pth"


def load_model() -> None:
    """Load SigLIP encoder + aesthetic head into process memory."""
    global _model, _preprocessor, _device, _dtype, _load_error

    if _model is not None and _preprocessor is not None:
        return

    if SKIP_AUTO_DOWNLOAD and not _predictor_weights_path().is_file():
        _load_error = (
            f"SKIP_AUTO_DOWNLOAD set and weights missing at {_predictor_weights_path()}"
        )
        logger.error(_load_error)
        return

    try:
        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR / "hf"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(MODEL_CACHE_DIR / "hf"))

        device = _resolve_device()
        dtype = _resolve_dtype(device)
        weights = _predictor_weights_path()
        predictor_path = str(weights) if weights.is_file() else None

        logger.info(
            "Loading aesthetic predictor (device=%s dtype=%s cache=%s)",
            device,
            dtype,
            MODEL_CACHE_DIR,
        )

        model, preprocessor = convert_v2_5_from_siglip(
            predictor_name_or_path=predictor_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model = model.to(device=device, dtype=dtype)
        model.eval()

        # Persist head weights locally so later boots skip the GitHub download.
        if predictor_path is None and not weights.is_file():
            try:
                torch.save(model.layers.state_dict(), weights)
                logger.info("Cached aesthetic head weights at %s", weights)
            except OSError as exc:
                logger.warning("Could not cache aesthetic head weights: %s", exc)

        _model = model
        _preprocessor = preprocessor
        _device = device
        _dtype = dtype
        _load_error = None
        logger.info("Aesthetic predictor ready")
    except Exception as exc:  # noqa: BLE001 — surface load failures via /health
        _model = None
        _preprocessor = None
        _device = None
        _dtype = None
        _load_error = str(exc)
        logger.exception("Failed to load aesthetic predictor: %s", exc)


def score_image_bytes(image_bytes: bytes) -> float:
    if _model is None or _preprocessor is None or _device is None or _dtype is None:
        raise RuntimeError(_load_error or "model not loaded")

    from PIL import Image

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"invalid image: {exc}") from exc

    pixel_values = (
        _preprocessor(images=image, return_tensors="pt")
        .pixel_values.to(device=_device, dtype=_dtype)
    )

    with torch.inference_mode():
        logits = _model(pixel_values).logits.squeeze()
        score = float(logits.float().cpu().numpy())

    return score


def score_image_batch(images_bytes: list[bytes]) -> list[float]:
    """Score several images in one forward pass (much faster than looping score_image_bytes on a GPU)."""
    if _model is None or _preprocessor is None or _device is None or _dtype is None:
        raise RuntimeError(_load_error or "model not loaded")

    from PIL import Image

    images = []
    for image_bytes in images_bytes:
        try:
            images.append(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"invalid image: {exc}") from exc

    pixel_values = (
        _preprocessor(images=images, return_tensors="pt")
        .pixel_values.to(device=_device, dtype=_dtype)
    )

    with torch.inference_mode():
        # squeeze(-1) (not squeeze()) keeps the batch dimension when len(images) == 1.
        logits = _model(pixel_values).logits.squeeze(-1)
        scores = logits.float().cpu().numpy()

    return [float(score) for score in scores]


def health_payload() -> dict[str, Any]:
    ready = _model is not None and _preprocessor is not None
    return {
        "status": "ok" if ready else "starting",
        "device": str(_device) if _device is not None else None,
        "dtype": str(_dtype) if _dtype is not None else None,
        "error": _load_error,
    }
