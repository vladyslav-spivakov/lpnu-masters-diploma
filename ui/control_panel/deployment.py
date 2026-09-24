import asyncio
import json
from collections import deque
from dataclasses import asdict
from time import monotonic

from .catalog import ESTIMATORS, Selection
from .runtime import DockerRuntime


class Deployment:
    def __init__(self, runtime: DockerRuntime, clock=monotonic):
        self.runtime = runtime
        self.clock = clock

        self.timings: dict[str, int] = {}
        self.timing_file = runtime.state / "timings.json"

        if self.timing_file.exists():
            self.timings = json.loads(self.timing_file.read_text(encoding="utf-8"))

        self.phase = "idle"
        self.error: str | None = None
        self.selection: Selection | None = None
        self.logs: deque[str] = deque(maxlen=400)
        self.task: asyncio.Task | None = None

        self.saved_selection = runtime.state / "selection.json"

        if self.saved_selection.exists():
            self.selection = Selection(
                **json.loads(self.saved_selection.read_text(encoding="utf-8"))
            )

    @property
    def busy(self) -> bool:
        return self.task is not None and not self.task.done()

    def start(self, selection: Selection) -> None:
        if self.busy:
            raise RuntimeError("A deployment is already in progress.")

        if selection.component not in {item.id for item in self.runtime.components}:
            raise ValueError("Unknown main component.")

        if selection.estimator not in {item.id for item in ESTIMATORS}:
            raise ValueError("Unknown intermediate estimator.")

        self.selection = selection
        self.phase = "preparing"
        self.error = None
        self.logs.clear()
        self.timings.clear()

        self.task = asyncio.create_task(self.run(selection))

    async def run(self, selection: Selection) -> None:
        try:
            await self.runtime.available()

            component = next(
                item for item in self.runtime.components if item.id == selection.component
            )

            self.runtime.prepare(component, selection)
            self.timing_file.write_text("{}", encoding="utf-8")
            self.saved_selection.write_text(json.dumps(asdict(selection)), encoding="utf-8")

            self.phase = "building"
            started = self.clock()
            self.logs.append("Building the selected model and API images.")

            await self.runtime.compose("build", log=self.logs.append)
            self.record_timing("building", started)

            self.phase = "starting"
            started = self.clock()
            self.logs.append("Starting containers and waiting for their health checks.")

            await self.runtime.compose(
                "up",
                "--detach",
                "--no-build",
                "--remove-orphans",
                log=self.logs.append,
            )
            self.record_timing("starting", started)

            self.phase = "checking"
            started = self.clock()
            self.logs.append("Checking model and API readiness.")

            await self.runtime.wait_ready()
            self.record_timing("ready", started)

            self.phase = "ready"
            self.logs.append("The model and API are healthy. Ready to receive requests.")
        except (OSError, RuntimeError, TimeoutError) as error:
            self.phase = "failed"
            self.error = str(error) or "Deployment timed out. Review the logs and try again."
            self.logs.append(self.error)

    def record_timing(self, stage: str, started: float) -> None:
        self.timings[stage] = round((self.clock() - started) * 1000)
        self.timing_file.write_text(json.dumps(self.timings), encoding="utf-8")

    async def snapshot(self) -> dict:
        docker_available = True

        if not self.busy:
            previous_task = self.task

            try:
                ready = await self.runtime.ready()

                if self.task is not previous_task or self.busy:
                    return await self.snapshot()

                if self.phase != "failed":
                    self.phase = "ready" if ready else "idle"

                self.error = self.error if self.phase == "failed" else None
            except (OSError, RuntimeError, TimeoutError) as error:
                if self.task is not previous_task or self.busy:
                    return await self.snapshot()

                docker_available = False

                if self.phase != "failed":
                    self.phase = "unavailable"
                    self.error = str(error)

        return {
            "phase": self.phase,
            "timings_ms": dict(self.timings),
            "busy": self.busy,
            "docker_available": docker_available,
            "selection": asdict(self.selection) if self.selection else None,
            "error": self.error,
            "logs": list(self.logs),
            "port": self.runtime.port,
        }

    async def close(self) -> None:
        if self.busy:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
