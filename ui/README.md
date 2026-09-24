# Submission control panel

A small local deployment UI: plain HTML, CSS Flexbox and JavaScript, served by
FastAPI. No frontend build step, external fonts or CDN assets.

## Start

With Docker Engine running (Linux containers in Docker Desktop), run from `submission`:

```sh
docker compose -f ui/compose.yaml up --build -d
```

Open http://localhost:8080, select an intermediate estimator and main component,
then choose **Apply configuration**. The first build downloads Python dependencies
and model weights; later builds reuse Docker's cache.

The panel starts only the selected CPU detector and orchestrator in the separate
`submission-ui-runtime` Compose project. It builds images, waits for container
health checks, and displays the API address only when both services are healthy
and match the applied selection. Apply recreates services when their configuration
changes; this resets the orchestrator's process-local stream state.

`None` disables temporal estimation and its policy/interval controls.
Hold last, Linear motion and Kalman / SORT support **Fixed cadence** with a
configurable detector interval N (1–1000 frames, default 3). Kalman / SORT also
supports **Adaptive cadence**, using the existing runtime v1 quality triggers to
request earlier anchors between scheduled detector frames. N is the maximum
scheduled interval in adaptive mode. Adaptive v2 is not exposed by this runtime.
Unsupported estimator/policy pairs are disabled in the UI and rejected by the API. Estimation applies to `/frames`; `/predict`
always uses the detector. These are configuration choices, not quality guarantees.
The existing submission defaults for target FPS and latency remain in effect.

The UI defaults to port 8080 and the API to port 8000. Set `UI_PORT` and `API_PORT`
in the shell before starting the panel to override them. Both ports bind to
127.0.0.1. The detector has no published host port.

## Stop

Stop the managed runtime before stopping its control panel:

```sh
docker compose -f ui/compose.yaml exec ui docker compose -p submission-ui-runtime -f /state/runtime.json down
docker compose -f ui/compose.yaml down
```

The first command applies after a configuration has been submitted. Named volumes
retain model weights and the last selection. Stopping the panel alone leaves
the inference API running. On restart, it checks the actual container health;
build logs are bounded to the last 400 lines and are not persisted.

Completed steps display measured wall-clock durations in milliseconds, saved
across panel restarts. Build images measures `compose build`; Start containers
measures `compose up --detach` (including the model dependency health wait);
API ready measures the subsequent readiness polling for both services. Timings
are separate stage durations, not cumulative times or model inference latency.
A new Apply clears previous durations; unfinished steps have no completion time.

## Static screenshot pages

The panel image also includes four non-interactive pages for presentation
screenshots. They do not call the API or change Docker state:

- `http://localhost:8080/static/screenshots/initial.html`
- `http://localhost:8080/static/screenshots/deploying.html`
- `http://localhost:8080/static/screenshots/ready.html`
- `http://localhost:8080/static/screenshots/error.html`

## Component catalog

`submission/compose.yaml` is the source of the model catalog. Services with these
labels appear in the selector even before their containers exist:

```yaml
labels:
  org.lpnu.component.role: anchor
  org.lpnu.component.id: yolo
  org.lpnu.component.name: YOLO11n
  org.lpnu.component.description: General-purpose object detection · CPU
  org.lpnu.component.port: "8001"
```

Each model must implement the existing model HTTP contract and a Docker
healthcheck. Rebuild the panel after changing submission source or the catalog:
the image contains its own copy of the submission build contexts. The optional
CUDA service is not exposed in this initial UI catalog.

## Boundaries

`catalog.py` translates Compose definitions into typed choices and a deployment
configuration. `runtime.py` owns Docker subprocesses. `deployment.py` owns the
single active deployment, selection and bounded log. `app.py` provides HTTP
endpoints and static assets. Frontend state and presentation live in `static/`.

The panel mounts the host Docker socket to manage sibling containers. It is a
local, trusted-machine tool, not an authenticated public administration service.
Commands use argument lists and validated catalog choices, never shell input.
Inference service code is unchanged. Runtime images, network and volumes use
the dedicated project name, separate from the research deployment.

## Checks

From `submission`, with the UI requirements, pytest, httpx and Ruff installed:

```sh
python -m pytest ui/tests -q
python -m ruff check ui
python -m ruff format --check ui
node --check ui/static/app.js
docker compose -f ui/compose.yaml config --quiet
```

The UI dependencies are FastAPI and Uvicorn, already used by the submission
services. Docker CLI, Compose and Buildx come from the official Docker CLI image.
