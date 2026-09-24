from copy import deepcopy
from dataclasses import dataclass


@dataclass(frozen=True)
class Component:
    id: str
    name: str
    description: str
    service: str
    port: int


@dataclass(frozen=True)
class Estimator:
    id: str
    name: str
    description: str


ESTIMATORS = (
    Estimator(
        "disabled", "None · detector only", "Run the main component for every admitted frame."
    ),
    Estimator("hold-last", "Hold last", "Reuse the last detection between anchor frames."),
    Estimator("linear-motion", "Linear motion", "Project detections using their recent movement."),
    Estimator(
        "kalman-sort", "Kalman / SORT", "Estimate motion and associate objects between frames."
    ),
)


@dataclass(frozen=True)
class Selection:
    component: str
    estimator: str
    policy: str = "fixed"
    every_n: int = 3

    def __post_init__(self) -> None:
        if self.policy not in {"fixed", "adaptive"}:
            raise ValueError("Unknown execution policy.")
        if self.policy == "adaptive" and self.estimator != "kalman-sort":
            raise ValueError("Adaptive policy requires Kalman / SORT.")

        if type(self.every_n) is not int or not 1 <= self.every_n <= 1000:
            raise ValueError("Detector interval must be an integer between 1 and 1000.")


def read_components(config: dict) -> tuple[Component, ...]:
    components = []

    for service, spec in config["services"].items():
        labels = spec.get("labels", {})
        if labels.get("org.lpnu.component.role") == "anchor":
            components.append(
                Component(
                    id=labels["org.lpnu.component.id"],
                    name=labels["org.lpnu.component.name"],
                    description=labels["org.lpnu.component.description"],
                    service=service,
                    port=int(labels["org.lpnu.component.port"]),
                )
            )

    return tuple(components)


def deployment_config(config: dict, component: Component, selection: Selection, port: int) -> dict:
    estimator = selection.estimator

    if estimator not in {item.id for item in ESTIMATORS}:
        raise ValueError("Unknown intermediate estimator.")

    model = deepcopy(config["services"][component.service])
    model.pop("profiles", None)
    model.pop("ports", None)

    orchestrator = deepcopy(config["services"]["orchestrator"])
    orchestrator["depends_on"] = {component.service: {"condition": "service_healthy"}}
    orchestrator["ports"] = [{"target": 8000, "published": str(port), "host_ip": "127.0.0.1"}]

    orchestrator["environment"].update(
        {
            "MODEL_CONTAINERS": f"{component.id}=http://{component.service}:{component.port}",
            "DEFAULT_MODEL_ID": component.id,
            "TEMPORAL_ENABLED": str(estimator != "disabled").lower(),
            "TEMPORAL_ESTIMATOR": "hold-last" if estimator == "disabled" else estimator,
            "TEMPORAL_DETECTOR_EVERY_N": str(selection.every_n),
            "ADAPTIVE_TEMPORAL_ENABLED": str(selection.policy == "adaptive").lower(),
        }
    )

    for service in (model, orchestrator):
        service.pop("networks", None)
        service.pop("image", None)
        service.setdefault("labels", {})["org.lpnu.deployment.selection"] = (
            f"{component.id}/{estimator}/{selection.policy}/{selection.every_n}/{port}"
        )
    volumes = {
        entry["source"]: {} for entry in model.get("volumes", []) if entry["type"] == "volume"
    }

    return {
        "services": {"orchestrator": orchestrator, component.service: model},
        "volumes": volumes,
    }
