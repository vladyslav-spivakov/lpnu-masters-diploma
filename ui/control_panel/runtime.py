import asyncio
import json
from collections import deque
from collections.abc import Callable
from pathlib import Path

from .catalog import Component, Selection, deployment_config, read_components


class DockerRuntime:
    def __init__(self, root: Path, state: Path, port: int):
        self.root = root
        self.state = state
        self.port = port
        self.config: dict = {}
        self.components: tuple[Component, ...] = ()
        self.compose_file = state / "runtime.json"

    async def command(self, *arguments: str, log: Callable[[str], None] | None = None) -> str:
        process = await asyncio.create_subprocess_exec(
            "docker",
            *arguments,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        output = deque(maxlen=400) if log else []
        try:
            async with asyncio.timeout(1800 if log else 20):
                async for line in process.stdout:
                    value = line.decode(errors="replace").rstrip()
                    output.append(value)
                    if log:
                        log(value)
                await process.wait()
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        result = "\n".join(output)
        if process.returncode:
            raise RuntimeError(result[-2000:] or "Docker command failed.")
        return result

    async def load_catalog(self) -> None:
        output = await self.command(
            "compose",
            "--file",
            str(self.root / "compose.yaml"),
            "config",
            "--format",
            "json",
        )

        self.config = json.loads(output)
        self.components = read_components(self.config)

    async def available(self) -> None:
        await self.command("info", "--format", "{{.ServerVersion}}")

    async def compose(self, *arguments: str, log: Callable[[str], None] | None = None) -> str:
        return await self.command(
            "compose",
            "--project-name",
            "submission-ui-runtime",
            "--file",
            str(self.compose_file),
            "--ansi",
            "never",
            *arguments,
            log=log,
        )

    def prepare(self, component: Component, selection: Selection) -> None:
        self.state.mkdir(parents=True, exist_ok=True)
        config = deployment_config(self.config, component, selection, self.port)
        self.compose_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    async def ready(self) -> bool:
        if not self.compose_file.exists():
            await self.available()
            return False

        output = await self.compose("ps", "--all", "--format", "json")
        rows = [json.loads(line) for line in output.splitlines() if line.strip()]
        services = json.loads(self.compose_file.read_text(encoding="utf-8"))["services"]

        if {row.get("Service") for row in rows} != set(services):
            return False

        for row in rows:
            expected = (
                services[row["Service"]].get("labels", {}).get("org.lpnu.deployment.selection")
            )
            labels = dict(
                item.split("=", 1) for item in row.get("Labels", "").split(",") if "=" in item
            )

            if (
                not expected
                or row.get("State") != "running"
                or row.get("Health") != "healthy"
                or labels.get("org.lpnu.deployment.selection") != expected
            ):
                return False

        return True

    async def wait_ready(self) -> None:
        async with asyncio.timeout(240):
            while not await self.ready():
                await asyncio.sleep(1)
