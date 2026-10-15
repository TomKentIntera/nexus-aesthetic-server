# aesthetic-predictor-server

Internal FastAPI sidecar wrapping
[Aesthetic Predictor V2.5](https://github.com/discus0434/aesthetic-predictor-v2-5)
(SigLIP-based image aesthetic score, roughly **1–10**; **5.5+** is considered strong).

Laravel (`AestheticPredictorClient` / `ScorePostAestheticJob`) calls this
service on Swarm. The local pull worker (`python/aesthetic-score/score_aesthetic.py`)
loads `predictor.py` **in-process** instead — do not run `main.py` on the GPU box.

## API

| Method | Path | Body | Response |
| --- | --- | --- | --- |
| `POST` | `/upload` | multipart `file` | `{ "score": 6.42, "device": "cpu" }` |
| `POST` | `/upload/batch` | multipart `files` (repeated) | `{ "scores": [6.42, 5.91], "device": "cpu" }` |
| `POST` | `/embed` | multipart `file` | `{ "siglip": "<base64 LE float32>", "dims": 1152, "model": "google/siglip-so400m-patch14-384" }` — L2-normalised SigLIP vector, decoded exactly as `python/image-embeddings` does, for images that are not posts (Emaki rating model Danbooru anchors) |
| `GET` | `/health` | — | `{ "status": "ok"\|"starting", "device", "dtype", "error" }` |

Uploads score on the FastAPI threadpool, so `/health` stays responsive (sub-second)
while a CPU inference is in flight.

## Configuration

| Env | Default | Meaning |
| --- | --- | --- |
| `SERVER_HOST` | `0.0.0.0` (Docker) | Bind address |
| `SERVER_PORT` | `5050` | Listen port |
| `MODEL_CACHE_DIR` | `/usr/src/app/models` | Cache for HF SigLIP + aesthetic head `.pth` |
| `DEVICE` | `cpu` | `cpu` / `cuda` / `mps` (auto CUDA if unset and available) |
| `SKIP_AUTO_DOWNLOAD` | `false` | Fail startup if weights are missing instead of downloading |

First boot downloads `google/siglip-so400m-patch14-384` and the aesthetic head
weights into `MODEL_CACHE_DIR` (use a Docker volume in Swarm/Compose).

## Docker / Portainer

Pushes to GHCR on `main`, `v*` tags, or manual workflow dispatch
(`.github/workflows/publish.yml`):

```text
ghcr.io/tomkentintera/nexus-aesthetic-server:latest
ghcr.io/tomkentintera/nexus-aesthetic-server:sha-<short>
ghcr.io/tomkentintera/nexus-aesthetic-server:vX.Y.Z   # tag pushes
```

Deploy with the included `docker-compose.yml` (Portainer stack or
`docker compose up -d`). Persist `MODEL_CACHE_DIR` via the
`aesthetic-predictor-models` volume. If the package is private, add a
`ghcr.io` registry credential in Portainer (PAT with `read:packages`).

## Local

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python3 main.py
```

```bash
curl -F file=@sample.jpg http://127.0.0.1:5050/upload
```

## Tests

```bash
pip install -r requirements.txt
pytest -q
```

Unit tests mock the model (no weight download required).
