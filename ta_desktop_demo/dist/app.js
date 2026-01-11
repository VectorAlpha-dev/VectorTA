const $ = (id) => document.getElementById(id);

function tauriInvoke() {
  const t = window.__TAURI__ || {};
  return t?.core?.invoke || t?.invoke || null;
}

function setStatus(msg, kind) {
  const el = $("status");
  el.textContent = msg;
  el.className = "mono " + (kind || "");
}

function parseMaTypes(id) {
  const raw = ($(id)?.value || "").trim();
  if (!raw) return [];
  return raw
    .split(/[,\s]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

function pretty(obj) {
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
}

function getRequest() {
  const dataId = $("dataId").textContent.trim();
  if (!dataId || dataId === "—") throw new Error("Load CSV first (missing data id).");

  const backend = $("backend").value;
  const deviceId = Number($("deviceId").value || 0);

  const fast = [
    Number($("fastStart").value),
    Number($("fastEnd").value),
    Number($("fastStep").value),
  ];
  const slow = [
    Number($("slowStart").value),
    Number($("slowEnd").value),
    Number($("slowStep").value),
  ];

  const fastMaTypes = parseMaTypes("fastMaTypes");
  const slowMaTypes = parseMaTypes("slowMaTypes");
  if (fastMaTypes.length === 0) throw new Error("Enter at least one fast MA type.");
  if (slowMaTypes.length === 0) throw new Error("Enter at least one slow MA type.");

  const maSource = $("maSource").value;

  const objective = $("objective").value;
  const mode = $("mode").value;

  const topK = Number($("topK").value || 0);
  const includeAll = $("includeAll").value === "true";

  const backendObj =
    backend === "GpuOnly"
      ? { backend: "GpuOnly", device_id: deviceId }
      : { backend: "CpuOnly" };

  return {
    backend: backendObj,
    data_id: dataId,
    fast_range: fast,
    slow_range: slow,
    fast_ma_types: fastMaTypes,
    slow_ma_types: slowMaTypes,
    ma_source: maSource,
    objective,
    mode,
    top_k: topK,
    include_all: includeAll,
  };
}

async function loadCsv() {
  const invoke = tauriInvoke();
  if (!invoke) throw new Error("Tauri invoke not available (window.__TAURI__ missing).");
  const path = $("csvPath").value.trim();
  if (!path) throw new Error("Enter a CSV path.");
  setStatus("Loading CSV...", "muted");
  const id = await invoke("load_price_data", { path });
  $("dataId").textContent = id;
  setStatus("Loaded.", "ok");
  $("output").textContent = pretty({ data_id: id });
}

async function runOptimization() {
  const invoke = tauriInvoke();
  if (!invoke) throw new Error("Tauri invoke not available (window.__TAURI__ missing).");
  const req = getRequest();
  setStatus("Running optimization...", "muted");
  const t0 = performance.now();
  const res = await invoke("run_double_ma_optimization", { req });
  const dt = performance.now() - t0;
  setStatus(`Done in ${dt.toFixed(0)}ms (frontend)`, "ok");
  $("output").textContent = pretty(res);
}

$("btnLoad").addEventListener("click", async () => {
  try {
    await loadCsv();
  } catch (e) {
    setStatus(String(e), "danger");
  }
});

$("btnClear").addEventListener("click", () => {
  $("dataId").textContent = "—";
  $("output").textContent = "{}";
  setStatus("Ready.", "");
});

$("btnRun").addEventListener("click", async () => {
  try {
    await runOptimization();
  } catch (e) {
    setStatus(String(e), "danger");
  }
});

// Default: disable device id unless GPU selected.
$("backend").addEventListener("change", () => {
  $("deviceId").disabled = $("backend").value !== "GpuOnly";
});
$("deviceId").disabled = $("backend").value !== "GpuOnly";
