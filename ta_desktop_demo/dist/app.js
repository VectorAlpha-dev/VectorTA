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

function readCheckedValues(selector) {
  return Array.from(document.querySelectorAll(selector))
    .filter((el) => el.checked)
    .map((el) => Number(el.value));
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

  const fastMaIds = readCheckedValues("input.fastMa");
  const slowMaIds = readCheckedValues("input.slowMa");
  if (fastMaIds.length === 0) throw new Error("Select at least one fast MA type.");
  if (slowMaIds.length === 0) throw new Error("Select at least one slow MA type.");

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
    fast_ma_ids: fastMaIds,
    slow_ma_ids: slowMaIds,
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
