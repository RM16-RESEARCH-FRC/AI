const sensorSpecs = [
  ["N", "Nitrogen", "mg/kg"],
  ["P", "Phosphorus", "mg/kg"],
  ["K", "Potassium", "mg/kg"],
  ["EC", "EC", "uS/cm"],
  ["pH", "pH", ""],
  ["moisture", "Moisture", "%"],
  ["temp", "Temperature", "C"],
];

const predictionSpecs = [
  ["delta_N", "Delta N", "mg/kg"],
  ["delta_P", "Delta P", "mg/kg"],
  ["delta_K", "Delta K", "mg/kg"],
  ["irrigation_ml", "Irrigation", "mL"],
  ["pH_adj", "pH Adj", ""],
];

let currentState = null;

const $ = (id) => document.getElementById(id);

function formatNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toFixed(digits);
}

function statusForValue(key, value, ranges) {
  if (value === null || value === undefined || !ranges[key]) return "unknown";
  const [min, max] = ranges[key];
  if (key === "pH" && (value < 4.5 || value > 7.0)) return "critical";
  if (key === "EC" && value >= 1500) return "critical";
  if (key === "moisture" && value <= 15) return "critical";
  if (value < min) return "low";
  if (value > max) return "high";
  return "ok";
}

function renderSensors(state) {
  const grid = $("sensorGrid");
  grid.innerHTML = "";
  sensorSpecs.forEach(([key, label, unit]) => {
    const value = state.sensors?.[key];
    const range = state.optimal_ranges?.[key];
    const status = statusForValue(key, value, state.optimal_ranges || {});
    const card = document.createElement("article");
    card.className = `metric ${status}`;
    const digits = key === "pH" ? 2 : 1;
    card.innerHTML = `
      <div class="label-row"><span>${label}</span><span>${status.toUpperCase()}</span></div>
      <div class="value">${formatNumber(value, digits)} <small>${unit}</small></div>
      <div class="range">Optimal ${range ? range.join(" - ") : "--"}</div>
    `;
    grid.appendChild(card);
  });
}

function renderPredictions(state) {
  const prediction = state.predictions || {};
  const actions = prediction.npk_action || prediction;
  $("predictionSource").textContent = prediction.source || "unknown";
  const grid = $("predictionGrid");
  grid.innerHTML = "";

  predictionSpecs.forEach(([key, label, unit]) => {
    const value = actions?.[key];
    const item = document.createElement("div");
    const digits = key === "pH_adj" ? 3 : 2;
    item.innerHTML = `<span>${label}</span><strong>${formatNumber(value, digits)}</strong><small>${unit}</small>`;
    grid.appendChild(item);
  });
}

function renderVision(state) {
  const vision = state.vision || {};
  $("leafStatus").textContent = vision.leaf_status || "--";
  $("leafConfidence").textContent = vision.leaf_confidence !== null && vision.leaf_confidence !== undefined ? `${Math.round(vision.leaf_confidence * 100)}% confidence` : "--";
  $("leafSeverity").textContent = vision.leaf_severity || "--";
  $("leafDetectionCount").textContent = vision.leaf_detection_count ?? "--";
  $("fruitCount").textContent = vision.fruit_count ?? "--";
  $("ripeness").textContent = vision.ripeness || "--";
  $("ripenessConfidence").textContent = vision.ripeness_confidence !== null && vision.ripeness_confidence !== undefined ? `${Math.round(vision.ripeness_confidence * 100)}% confidence` : "--";
  $("weight").textContent = vision.estimated_weight_kg !== null && vision.estimated_weight_kg !== undefined ? `${formatNumber(vision.estimated_weight_kg, 2)} kg` : "--";

  const leafModel = state.model?.leaf || {};
  $("leafModelClasses").textContent = leafModel.classes?.length ? leafModel.classes.join(" / ") : "classes pending";
}

function renderSystem(state) {
  const connection = state.connection || "unknown";
  const pill = $("connectionPill");
  pill.textContent = connection.toUpperCase();
  pill.className = `pill ${connection}`;

  $("subtitle").textContent = `Updated ${state.updated_at || "--"} | Growth stage ${state.sensors?.growth_stage ?? "--"}`;
  $("updatedAt").textContent = state.updated_at || "--";
  $("jetsonName").textContent = state.system?.jetson_name || "--";
  $("latency").textContent = state.system?.latency_ms ? `${formatNumber(state.system.latency_ms, 1)} ms` : "--";
  $("modelStatus").textContent = state.model?.status || state.predictions?.model_status || "--";
  $("leafModelStatus").textContent = state.model?.leaf?.status || "--";
  $("usbFps").textContent = state.system?.fps_usb ? `${formatNumber(state.system.fps_usb, 1)} fps` : "-- fps";
  $("simulationButton").textContent = state.simulation_enabled ? "S" : "J";
  $("simulationButton").title = state.simulation_enabled ? "Simulation is on" : "Waiting for Jetson telemetry";
}

function setFeed(elementId, placeholderId, url) {
  const img = $(elementId);
  const frame = img.parentElement;
  if (url) {
    if (img.src !== url) img.src = url;
    frame.classList.add("has-feed");
  } else {
    img.removeAttribute("src");
    frame.classList.remove("has-feed");
    $(placeholderId).style.display = "";
  }
}

function renderStreams(state) {
  const storedUsb = localStorage.getItem("farmbot_usb_stream") || "";
  const usb = storedUsb || state.streams?.usb_cam || "";
  if (document.activeElement !== $("usbUrlInput")) {
    $("usbUrlInput").value = usb;
  }
  setFeed("usbFeed", "usbPlaceholder", usb);
}

function renderEvents(state) {
  const list = $("eventList");
  list.innerHTML = "";
  const events = [...(state.events || [])].reverse();
  if (!events.length) {
    const empty = document.createElement("div");
    empty.className = "event";
    empty.textContent = "No events yet.";
    list.appendChild(empty);
    return;
  }
  events.slice(0, 20).forEach((event) => {
    const row = document.createElement("div");
    row.className = `event ${event.level || "info"}`;
    const level = document.createElement("strong");
    level.textContent = event.level || "info";
    const message = document.createTextNode(` ${event.message || ""}`);
    const breakLine = document.createElement("br");
    const time = document.createElement("small");
    time.textContent = event.time || "";
    row.append(level, message, breakLine, time);
    list.appendChild(row);
  });
}

function drawTrend(state) {
  const canvas = $("trendCanvas");
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const history = state.history || [];
  if (history.length < 2) {
    ctx.fillStyle = "#627066";
    ctx.font = "14px sans-serif";
    ctx.fillText("Trend will appear after a few samples.", 24, 42);
    return;
  }

  const series = [
    ["moisture", "#1f7a4d", 0, 100],
    ["pH", "#b7791f", 0, 14],
    ["EC", "#2764a3", 400, 1500],
  ];
  ctx.strokeStyle = "#d8ded9";
  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i += 1) {
    const y = 24 + (i * (height - 48)) / 4;
    ctx.beginPath();
    ctx.moveTo(32, y);
    ctx.lineTo(width - 18, y);
    ctx.stroke();
  }

  series.forEach(([key, color, min, max]) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    history.forEach((row, idx) => {
      const x = 32 + (idx * (width - 54)) / (history.length - 1);
      const raw = Number(row[key]);
      const normalized = Math.max(0, Math.min(1, (raw - min) / (max - min)));
      const y = height - 24 - normalized * (height - 52);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  ctx.font = "13px sans-serif";
  ctx.fillStyle = "#1f7a4d";
  ctx.fillText("Moisture", 38, 22);
  ctx.fillStyle = "#b7791f";
  ctx.fillText("pH", 118, 22);
  ctx.fillStyle = "#2764a3";
  ctx.fillText("EC", 152, 22);
}

function render(state) {
  currentState = state;
  renderSystem(state);
  renderStreams(state);
  renderSensors(state);
  renderPredictions(state);
  renderVision(state);
  renderEvents(state);
  drawTrend(state);
}

async function refresh() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    render(await response.json());
  } catch (error) {
    $("connectionPill").textContent = "OFFLINE";
    $("connectionPill").className = "pill stale";
  }
}

$("applyStreams").addEventListener("click", () => {
  localStorage.setItem("farmbot_usb_stream", $("usbUrlInput").value.trim());
  if (currentState) renderStreams(currentState);
});

$("simulationButton").addEventListener("click", async () => {
  const next = !(currentState?.simulation_enabled);
  await fetch(`/api/simulation/${next}`, { method: "POST" });
  await refresh();
});

refresh();
setInterval(refresh, 1000);
