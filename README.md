# Master's Diploma 2026

1. Choose a model:

```text
DEFAULT_MODEL_ID=yolo
```

Use `ssd` for SSDLite. To use another YOLO model:

```text
YOLO_MODEL_NAME=yolo11s
YOLO_MODEL_VERSION=yolo11s
```

2. Build the containers:

```text
docker compose build
```

3. Start the project:

```text
docker compose up
```
