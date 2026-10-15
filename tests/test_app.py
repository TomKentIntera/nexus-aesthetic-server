from __future__ import annotations

import asyncio
import io
import threading
import time
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    import main as app_module
    import predictor

    # Avoid downloading SigLIP / aesthetic weights in unit tests.
    monkeypatch.setenv("SKIP_AUTO_DOWNLOAD", "true")
    predictor._model = MagicMock()
    predictor._preprocessor = MagicMock()
    predictor._device = MagicMock()
    predictor._device.__str__ = lambda self: "cpu"  # type: ignore[method-assign]
    predictor._dtype = MagicMock()
    predictor._load_error = None

    def fake_score(_bytes: bytes) -> float:
        return 6.25

    monkeypatch.setattr(predictor, "score_image_bytes", fake_score)
    monkeypatch.setattr(predictor, "load_model", lambda: None)

    return TestClient(app_module.app)


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=(120, 40, 200)).save(buf, format="PNG")
    return buf.getvalue()


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["device"] == "cpu"


def test_upload_returns_score(client: TestClient):
    response = client.post(
        "/upload",
        files={"file": ("sample.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["score"] == pytest.approx(6.25)
    assert body["device"] == "cpu"


def test_upload_rejects_empty_file(client: TestClient):
    response = client.post(
        "/upload",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400


def test_upload_batch_returns_one_score_per_file(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    import predictor

    monkeypatch.setattr(predictor, "score_image_batch", lambda _images: [1.5, 2.5])

    response = client.post(
        "/upload/batch",
        files=[
            ("files", ("a.png", _png_bytes(), "image/png")),
            ("files", ("b.png", _png_bytes(), "image/png")),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["scores"] == pytest.approx([1.5, 2.5])
    assert body["device"] == "cpu"


def test_health_stays_responsive_while_scoring(monkeypatch: pytest.MonkeyPatch):
    import main as app_module
    import predictor

    predictor._model = MagicMock()
    predictor._preprocessor = MagicMock()
    predictor._device = MagicMock()
    predictor._device.__str__ = lambda self: "cpu"  # type: ignore[method-assign]
    predictor._dtype = MagicMock()
    predictor._load_error = None

    started = threading.Event()
    release = threading.Event()

    def slow_score(_bytes: bytes) -> float:
        started.set()
        # Blocking CPU inference stand-in; must not run on the event loop.
        assert release.wait(timeout=5)
        return 6.25

    monkeypatch.setattr(predictor, "score_image_bytes", slow_score)
    monkeypatch.setattr(predictor, "load_model", lambda: None)

    async def scenario() -> tuple[httpx.Response, float, httpx.Response]:
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as async_client:
            upload_task = asyncio.create_task(
                async_client.post(
                    "/upload",
                    files={"file": ("sample.png", _png_bytes(), "image/png")},
                )
            )
            # Wait until scoring is actually in flight before probing health.
            assert await asyncio.to_thread(started.wait, 2)

            probe_start = time.perf_counter()
            health_response = await async_client.get("/health")
            health_elapsed = time.perf_counter() - probe_start

            release.set()
            upload_response = await upload_task
            return health_response, health_elapsed, upload_response

    health_response, health_elapsed, upload_response = asyncio.run(scenario())

    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"
    assert health_elapsed < 1.0
    assert upload_response.status_code == 200
    assert upload_response.json()["score"] == pytest.approx(6.25)


def test_health_reports_starting_without_model(monkeypatch: pytest.MonkeyPatch):
    import main as app_module
    import predictor

    monkeypatch.setattr(predictor, "load_model", lambda: None)
    predictor._model = None
    predictor._preprocessor = None
    predictor._device = None
    predictor._dtype = None
    predictor._load_error = "not loaded"

    with TestClient(app_module.app) as local_client:
        response = local_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "starting"


def test_embed_returns_a_base64_float32_vector(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import base64
    import struct

    import predictor

    monkeypatch.setattr(predictor, "embed_image_bytes", lambda _bytes: [0.6, 0.8])
    response = client.post("/embed", files={"file": ("x.png", _png_bytes(), "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert body["dims"] == 2
    assert body["model"] == "google/siglip-so400m-patch14-384"
    assert struct.unpack("<2f", base64.b64decode(body["siglip"])) == pytest.approx((0.6, 0.8))


def test_embed_rejects_undecodable_bytes(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import predictor

    def bad(_bytes: bytes) -> list[float]:
        raise ValueError("cannot decode image")

    monkeypatch.setattr(predictor, "embed_image_bytes", bad)
    response = client.post("/embed", files={"file": ("x.png", b"nope", "image/png")})
    assert response.status_code == 400
