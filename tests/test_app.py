from __future__ import annotations

import io
from unittest.mock import MagicMock

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
