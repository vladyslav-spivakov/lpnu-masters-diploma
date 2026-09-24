const element = (id) => document.getElementById(id);
const form = element("configuration");
const component = element("component");
const estimator = element("estimator");
const policy = element("policy");
const interval = element("every-n");
const apply = element("apply");

const phaseCopy = {
  idle: ["Ready when you are.", "Apply a configuration to start the model and API."],
  preparing: ["Preparing your setup.", "Checking Docker and applying your selection."],
  building: ["Building the images.", "The first build takes longer. You can follow the output below."],
  starting: ["Starting your workspace.", "Loading the model and checking that both services are healthy."],
  checking: ["Checking readiness.", "Waiting for the model and API health checks."],
  ready: ["Your API is ready.", "The model and API are running. You can start sending requests."],
  failed: ["Startup needs attention.", "Review the details below, then apply the configuration to try again."],
  unavailable: ["Docker is unavailable.", "Start Docker, then return here. We’ll check the connection automatically."],
};
let catalog;
let status;
let submitting = false;
let initialized = false;


async function request(path, options) {
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "The request could not be completed.");
  return result;
}


function populate(select, items) {
  select.replaceChildren(...items.map((item) => new Option(item.name, item.id)));
}


function describeSelection() {
  const temporal = estimator.value !== "disabled";
  const supportsAdaptive = estimator.value === "kalman-sort";

  policy.querySelector('[value="adaptive"]').disabled = !supportsAdaptive;
  if (!supportsAdaptive) policy.value = "fixed";
  element("interval-label").textContent = policy.value === "adaptive"
    ? "Max interval (frames)" : "Detector interval (frames)";
  if (status) renderControls();
  policy.disabled = !temporal || submitting || Boolean(status?.busy);
  interval.disabled = !temporal || submitting || Boolean(status?.busy);
}


function selection() {
  return {component: component.value, estimator: estimator.value, policy: policy.value, every_n: Number(interval.value)};
}


function renderControls() {
  const busy = submitting || status.busy;
  element("fields").disabled = busy;
  policy.disabled = busy || estimator.value === "disabled";
  interval.disabled = busy || estimator.value === "disabled";
  apply.disabled = busy || !status.docker_available || !component.value;
  apply.firstChild.textContent = busy ? "Applying configuration…" : "Apply configuration";
  const changed = status.phase === "ready" && status.selection && (
    Object.entries(selection()).some(([key, value]) => status.selection[key] !== value)
  );
  element("apply-note").textContent = changed
    ? "Your changes will take effect when you apply."
    : "The first build may take a few minutes.";
}


function render(next) {
  status = next;
  if (!initialized) {
    if (status.selection) {
      component.value = status.selection.component;
      estimator.value = status.selection.estimator;
      policy.value = status.selection.policy || "fixed";
      interval.value = status.selection.every_n || 3;
    }
    initialized = true;
    describeSelection();
  }
  const [title, description] = phaseCopy[status.phase] || phaseCopy.idle;
  element("status-title").textContent = title;
  element("status-description").textContent = description;
  element("status-block").dataset.phase = status.phase;
  element("connection").textContent = status.docker_available ? "Docker connected" : "Docker unavailable";
  element("connection").dataset.online = String(status.docker_available);
  const progress = ["building", "starting", "checking"].indexOf(status.phase);
  document.querySelectorAll("[data-step]").forEach((step, index) => {
    step.classList.toggle("active", index === progress && status.busy);
    const duration = status.timings_ms?.[step.dataset.step];
    const done = duration !== undefined;
    step.classList.toggle("done", done || status.phase === "ready");
    step.querySelector("i").textContent = done
      ? `${duration.toLocaleString("en-US")} ms ✓`
      : index === progress && status.busy ? "In progress" : status.phase === "ready" ? "✓" : "";
  });
  element("endpoint").hidden = status.phase !== "ready";
  const address = `http://${location.hostname}:${status.port}`;
  element("endpoint-url").textContent = address;
  element("api-docs").href = `${address}/docs`;
  element("port").textContent = `Port ${status.port}`;
  element("error").hidden = !status.error;
  element("error").textContent = status.error || "";
  const logs = element("logs");
  const atBottom = logs.scrollHeight - logs.scrollTop - logs.clientHeight < 30;
  logs.textContent = status.logs.length ? status.logs.join("\n") : "Logs will appear here when you apply a configuration.";
  if (atBottom) logs.scrollTop = logs.scrollHeight;
  element("log-count").textContent = status.logs.length ? `${status.logs.length} lines` : "No output yet";
  renderControls();
}


async function refresh() {
  try {
    if (!catalog) {
      catalog = await request("/api/catalog");
      populate(component, catalog.components);
      populate(estimator, catalog.estimators);
      describeSelection();
    }
    render(await request("/api/status"));
  } catch (error) {
    element("connection").textContent = "Panel disconnected";
    element("connection").dataset.online = "false";
    element("endpoint").hidden = true;
    element("error").hidden = false;
    element("error").textContent = "Cannot reach the control panel. Reconnecting…";
    apply.disabled = true;
  } finally {
    window.setTimeout(refresh, 2000);
  }
}


form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitting = true;
  renderControls();
  element("logs-panel").open = true;
  try {
    render(await request("/api/apply", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(selection()),
    }));
  } catch (error) {
    element("error").hidden = false;
    element("error").textContent = error.message;
  } finally {
    submitting = false;
    renderControls();
  }
});

policy.addEventListener("change", describeSelection);
interval.addEventListener("input", describeSelection);
component.addEventListener("change", describeSelection);
estimator.addEventListener("change", describeSelection);

element("copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(element("endpoint-url").textContent);
    element("copy").textContent = "Copied";
  } catch {
    element("copy").textContent = "Select and copy the address above";
  }
  window.setTimeout(() => { element("copy").textContent = "Copy address"; }, 2500);
});

refresh();
