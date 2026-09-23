from __future__ import annotations

import io

import pytest
import torch
from PIL import Image


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _mock_model(monkeypatch: pytest.MonkeyPatch):
    import predictor

    class FakePreprocessor:
        def __call__(self, images, return_tensors):
            batch = images if isinstance(images, list) else [images]
            return type("Out", (), {"pixel_values": torch.zeros(len(batch), 3, 4, 4)})()

    class FakeOutput:
        def __init__(self, n: int):
            self.logits = torch.arange(1, n + 1, dtype=torch.float32).unsqueeze(-1)

    class FakeModel:
        def __call__(self, pixel_values):
            return FakeOutput(pixel_values.shape[0])

    monkeypatch.setattr(predictor, "_model", FakeModel())
    monkeypatch.setattr(predictor, "_preprocessor", FakePreprocessor())
    monkeypatch.setattr(predictor, "_device", torch.device("cpu"))
    monkeypatch.setattr(predictor, "_dtype", torch.float32)
    monkeypatch.setattr(predictor, "_load_error", None)
    yield


def test_score_image_batch_returns_one_score_per_image():
    import predictor

    images = [_png_bytes((10, 20, 30)), _png_bytes((200, 100, 50)), _png_bytes((0, 0, 0))]
    scores = predictor.score_image_batch(images)

    assert scores == [1.0, 2.0, 3.0]


def test_score_image_batch_of_one_keeps_batch_dimension():
    import predictor

    scores = predictor.score_image_batch([_png_bytes((10, 20, 30))])

    assert scores == [1.0]


def test_score_image_batch_rejects_invalid_image():
    import predictor

    with pytest.raises(ValueError):
        predictor.score_image_batch([b"not an image"])


def test_score_image_batch_requires_loaded_model(monkeypatch: pytest.MonkeyPatch):
    import predictor

    monkeypatch.setattr(predictor, "_model", None)

    with pytest.raises(RuntimeError):
        predictor.score_image_batch([_png_bytes((1, 2, 3))])
