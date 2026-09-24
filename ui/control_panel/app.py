import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .catalog import ESTIMATORS
from .deployment import Deployment, Selection
from .runtime import DockerRuntime

UI_ROOT = Path(__file__).resolve().parent.parent


class ApplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    component: str
    estimator: str
    policy: str = "fixed"
    every_n: int = Field(default=3, ge=1, le=1000, strict=True)


def create_app(runtime: DockerRuntime | None = None) -> FastAPI:
    runtime = runtime or DockerRuntime(
        UI_ROOT.parent,
        Path(os.getenv("UI_STATE_DIR", str(UI_ROOT / ".state"))),
        int(os.getenv("API_PORT", "8000")),
    )
    deployment = Deployment(runtime)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await runtime.load_catalog()
        yield
        await deployment.close()

    app = FastAPI(title="Submission control panel", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]"])

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        if request.method == "POST":
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                return JSONResponse(
                    {"detail": "Use the control panel on its own origin."},
                    status_code=403,
                )
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {"status": "ready"}

    @app.get("/api/catalog")
    async def catalog():
        return {
            "components": [asdict(item) for item in runtime.components],
            "estimators": [asdict(item) for item in ESTIMATORS],
        }

    @app.get("/api/status")
    async def status():
        return await deployment.snapshot()

    @app.post("/api/apply", status_code=202)
    async def apply(body: ApplyRequest):
        try:
            deployment.start(Selection(body.component, body.estimator, body.policy, body.every_n))
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error
        return await deployment.snapshot()

    @app.get("/")
    async def index():
        return FileResponse(UI_ROOT / "static" / "index.html")

    app.mount("/static", StaticFiles(directory=UI_ROOT / "static"), name="static")
    return app


app = create_app()
