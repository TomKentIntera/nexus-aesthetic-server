# aesthetic-predictor-server

Standalone FastAPI service wrapping Aesthetic Predictor V2.5 (SigLIP). Sibling to
`wd14-tagger-server/` / `rating-classifier-server/`. Called over HTTP by Laravel
(`AestheticPredictorClient`, `ScorePostAestheticJob`, `posts:aesthetic-score:dispatch`).
The local pull worker (`python/aesthetic-score`) imports `predictor.py` in-process
and does not run this FastAPI server.

## Structure

- `main.py` — FastAPI app, `/upload` + `/health`, entrypoint for Swarm.
- `predictor.py` — model load + `score_image_bytes`; shared with `python/aesthetic-score`.
- `requirements.txt` — fastapi, torch (installed CPU-first in Dockerfile), transformers, aesthetic-predictor-v2-5.
- `Dockerfile` — CPU torch wheels; models cached under `MODEL_CACHE_DIR` volume.
- `tests/` — pytest suite (mocks model; no weight download).

## Key conventions

- Config via `SERVER_HOST` / `SERVER_PORT` / `MODEL_CACHE_DIR` / `DEVICE` / `SKIP_AUTO_DOWNLOAD`.
- Image I/O is multipart `file` upload (same pattern as WD14 `/upload`).
- Scores are floats roughly in 1–10; Laravel stores them on `posts.aesthetic_score`.
