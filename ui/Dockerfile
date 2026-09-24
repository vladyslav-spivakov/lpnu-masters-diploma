FROM docker:29-cli AS docker
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UI_STATE_DIR=/state
COPY --from=docker /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker /usr/local/libexec/docker/cli-plugins/docker-compose /usr/local/libexec/docker/cli-plugins/docker-compose
COPY --from=docker /usr/local/libexec/docker/cli-plugins/docker-buildx /usr/local/libexec/docker/cli-plugins/docker-buildx
WORKDIR /workspace/submission/ui
COPY ui/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY compose.yaml /workspace/submission/compose.yaml
COPY services /workspace/submission/services
COPY ui/control_panel ./control_panel
COPY ui/static ./static
EXPOSE 8080
CMD ["python", "-m", "uvicorn", "control_panel.app:app", "--host", "0.0.0.0", "--port", "8080"]
