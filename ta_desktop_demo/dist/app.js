const $ = (id) => document.getElementById(id);

function tauriInvoke() {
  const t = window.__TAURI__ || {};
  return t?.core?.invoke || t?.invoke || null;
}

function tauriListen() {
  const t = window.__TAURI__ || {};
  return t?.event?.listen || null;
}

let MA_LIST = [];
let MA_BY_ID = {};
let RUNNING = false;
let MA_PARAMS_STATE = { fast: {}, slow: {} };

const SETTINGS_KEY = "vectorbt_settings_v1";
const HISTORY_KEY = "vectorbt_history_v1";
const RECENT_CSV_KEY = "vectorbt_recent_csv_v1";
const RECENT_CSV_MAX = 12;

let CURRENT_RUN = null; // { ts, req, res }
let TOP_ROWS = [];
let TOP_ROWS_FILTERED = [];
let TOP_SORT = { key: "rank", dir: "asc" };
let HEATMAP_META_BASE = "";
let DRILL_SEQ = 0;
let DRILL_SELECTED_KEY = "";
let PARETO_POINTS_RENDERED = [];
let CURRENT_DATA_META = null;

function setStatus(msg, kind) {
  const el = $("status");
  el.textContent = msg;
  el.className = "mono " + (kind || "");
}

function pretty(obj) {
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
}

function maParamSchema(maId) {
  const m = MA_BY_ID[maId];
  const params = m?.params;
  return Array.isArray(params) ? params : [];
}

function ensureMaParamState(side, maId) {
  if (!MA_PARAMS_STATE || typeof MA_PARAMS_STATE !== "object") MA_PARAMS_STATE = { fast: {}, slow: {} };
  if (!MA_PARAMS_STATE[side] || typeof MA_PARAMS_STATE[side] !== "object") MA_PARAMS_STATE[side] = {};
  if (!MA_PARAMS_STATE[side][maId] || typeof MA_PARAMS_STATE[side][maId] !== "object") MA_PARAMS_STATE[side][maId] = {};
  return MA_PARAMS_STATE[side][maId];
}

function resolvedMaParams(side, maId) {
  const schema = maParamSchema(maId);
  if (!schema.length) return null;
  const state = ensureMaParamState(side, maId);
  const out = {};
  for (const p of schema) {
    const k = String(p.key || "");
    if (!k) continue;
    const v = Number(state[k]);
    out[k] = Number.isFinite(v) ? v : Number(p.default || 0);
  }
  return out;
}

function renderMaParamForm(side, maId) {
  const wrap = $(side === "fast" ? "fastMaParamsWrap" : "slowMaParamsWrap");
  const body = $(side === "fast" ? "fastMaParamsBody" : "slowMaParamsBody");
  if (!wrap || !body) return;

  const schema = maParamSchema(maId);
  body.innerHTML = "";
  if (!schema.length) {
    wrap.style.display = "none";
    return;
  }
  wrap.style.display = "block";

  const state = ensureMaParamState(side, maId);
  for (const p of schema) {
    const key = String(p.key || "");
    const labelText = String(p.label || key || "param");
    if (!key) continue;

    const id = `${side}_${maId}_param_${key}`;

    const block = document.createElement("div");
    block.style.marginTop = "0.6rem";

    const label = document.createElement("label");
    label.htmlFor = id;
    label.textContent = labelText;

    const input = document.createElement("input");
    input.id = id;
    input.type = "number";
    input.dataset.maParam = "1";
    input.dataset.maSide = side;
    input.dataset.maId = maId;
    input.dataset.maKey = key;
    if (p.min != null) input.min = String(p.min);
    if (p.max != null) input.max = String(p.max);
    if (p.step != null) input.step = String(p.step);
    else input.step = p.kind === "int" ? "1" : "any";

    const v = Number(state[key]);
    input.value = String(Number.isFinite(v) ? v : Number(p.default || 0));

    const onChange = () => {
      const x = Number(input.value);
      if (Number.isFinite(x)) state[key] = x;
      saveSettingsSoon();
    };
    input.addEventListener("change", onChange);
    input.addEventListener("input", onChange);

    block.appendChild(label);
    block.appendChild(input);

    const notes = String(p.notes || "");
    if (notes) {
      const note = document.createElement("div");
      note.className = "muted";
      note.style.marginTop = "0.25rem";
      note.style.fontSize = "0.8rem";
      note.textContent = notes;
      block.appendChild(note);
    }

    body.appendChild(block);
  }
}

function refreshMaParamsUi() {
  const fast = $("fastMaType")?.value || "";
  const slow = $("slowMaType")?.value || "";
  renderMaParamForm("fast", fast);
  renderMaParamForm("slow", slow);
}

function num(v, fallback) {
  const x = Number(v);
  return Number.isFinite(x) ? x : fallback;
}

function fmt(x, digits) {
  const n = Number(x);
  if (!Number.isFinite(n)) return "-";
  const d = Number.isFinite(digits) ? digits : 4;
  return n.toFixed(d);
}

function expandRangeU16Like(start, end, step) {
  const s0 = Math.floor(Number(start));
  const e0 = Math.floor(Number(end));
  const st0 = Math.floor(Number(step));
  if (!Number.isFinite(s0) || !Number.isFinite(e0) || !Number.isFinite(st0)) return [];

  if (st0 === 0 || s0 === e0) return [s0];
  const st = Math.max(1, st0);
  const lo = Math.min(s0, e0);
  const hi = Math.max(s0, e0);
  const out = [];
  let v = lo;
  for (;;) {
    out.push(v);
    if (v === hi) break;
    const next = v + st;
    if (next <= v || next > hi) break;
    v = next;
  }
  return out;
}

function countPairsFastLtSlow(fastPeriods, slowPeriods) {
  const f = Array.isArray(fastPeriods) ? fastPeriods : [];
  const s = Array.isArray(slowPeriods) ? slowPeriods : [];
  if (!f.length || !s.length) return 0;

  const slowSorted = [...s].sort((a, b) => a - b);
  let total = 0;
  for (const ff of f) {
    let lo = 0;
    let hi = slowSorted.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (slowSorted[mid] <= ff) lo = mid + 1;
      else hi = mid;
    }
    total += slowSorted.length - lo;
  }
  return total;
}

function maRequiresVolumeField(maId) {
  const id = String(maId || "").trim().toLowerCase();
  return id === "vwma" || id === "vpwma";
}

function maRequiresHighLowField(maId) {
  const id = String(maId || "").trim().toLowerCase();
  return id === "frama";
}

function requiredFieldsForMaSource(source) {
  const s = String(source || "").trim().toLowerCase();
  if (s === "close") return ["close"];
  if (s === "open") return ["open"];
  if (s === "high") return ["high"];
  if (s === "low") return ["low"];
  if (s === "hl2") return ["high", "low"];
  if (s === "hlc3") return ["high", "low", "close"];
  if (s === "ohlc4") return ["open", "high", "low", "close"];
  if (s === "hlcc" || s === "hlcc4") return ["high", "low", "close"];
  return [];
}

function validateCandleFieldRequirements(meta, fastMaType, slowMaType, maSource) {
  if (!meta || typeof meta !== "object") return;

  const missing = new Set();
  const require = (field) => {
    if (field === "open" && !meta.has_open) missing.add("open");
    if (field === "high" && !meta.has_high) missing.add("high");
    if (field === "low" && !meta.has_low) missing.add("low");
    if (field === "close" && !meta.has_close) missing.add("close");
    if (field === "volume" && !meta.has_volume) missing.add("volume");
  };

  if (maRequiresVolumeField(fastMaType) || maRequiresVolumeField(slowMaType)) require("volume");
  if (maRequiresHighLowField(fastMaType) || maRequiresHighLowField(slowMaType)) {
    require("high");
    require("low");
  }
  for (const f of requiredFieldsForMaSource(maSource)) require(f);

  if (missing.size) {
    const parts = [];
    if (missing.has("volume")) parts.push("volume (needed by VWMA/VPWMA)");
    if (missing.has("high") || missing.has("low")) parts.push("high/low (needed by FRAMA and some sources)");
    if (missing.has("open")) parts.push("open (needed by some sources)");
    if (missing.has("close")) parts.push("close");
    throw new Error(`Loaded CSV is missing required candle fields: ${parts.join(", ")}.`);
  }
}

function fieldPresent(meta, field) {
  if (!meta || typeof meta !== "object") return false;
  if (field === "open") return !!meta.has_open;
  if (field === "high") return !!meta.has_high;
  if (field === "low") return !!meta.has_low;
  if (field === "close") return !!meta.has_close;
  if (field === "volume") return !!meta.has_volume;
  return false;
}

function renderDataMeta(meta) {
  const lenEl = $("dataLen");
  if (lenEl) lenEl.textContent = meta && Number.isFinite(Number(meta.len)) ? String(meta.len) : "—";

  const fieldsEl = $("dataFields");
  const hintEl = $("dataFieldHint");
  if (!fieldsEl) return;

  fieldsEl.innerHTML = "";
  if (hintEl) hintEl.textContent = "";

  if (!meta || typeof meta !== "object") {
    fieldsEl.textContent = "—";
    return;
  }

  const fields = [
    ["open", !!meta.has_open],
    ["high", !!meta.has_high],
    ["low", !!meta.has_low],
    ["close", !!meta.has_close],
    ["volume", !!meta.has_volume],
  ];
  for (const [name, ok] of fields) {
    const badge = document.createElement("span");
    badge.className = `badge ${ok ? "on" : "off"}`;
    badge.textContent = name;
    fieldsEl.appendChild(badge);
  }

  const msgs = [];
  if (!meta.has_close) msgs.push("Missing close (required).");
  if (!meta.has_volume) msgs.push("No volume → VWMA/VPWMA disabled.");
  if (!meta.has_high || !meta.has_low) msgs.push("No high/low → FRAMA + some sources disabled.");
  if (!meta.has_open) msgs.push("No open → open/ohlc4 disabled.");
  if (hintEl) hintEl.textContent = msgs.join(" · ");
}

function applyMaSourceConstraints() {
  const sel = $("maSource");
  if (!sel) return;

  const meta = CURRENT_DATA_META;
  for (const opt of sel.options) {
    const required = requiredFieldsForMaSource(opt.value);
    let ok = true;
    if (meta) {
      for (const f of required) {
        if (!fieldPresent(meta, f)) {
          ok = false;
          break;
        }
      }
    }
    opt.disabled = !ok;
  }

  if (sel.value && sel.selectedOptions?.[0]?.disabled) {
    let picked = "";
    for (const opt of sel.options) {
      if (!opt.disabled) {
        picked = opt.value;
        break;
      }
    }
    if (picked) sel.value = picked;
  }
}

function downloadText(filename, text, mime) {
  const blob = new Blob([text], { type: mime || "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2500);
}

function setRunning(running) {
  RUNNING = !!running;
  const btnRun = $("btnRun");
  const btnCancel = $("btnCancel");
  if (btnRun) btnRun.disabled = RUNNING;
  if (btnCancel) btnCancel.disabled = !RUNNING;

  const ids = [
    "csvPath",
    "btnBrowseCsv",
    "btnLoad",
    "btnClear",
    "backend",
    "deviceId",
    "btnPresetKernelSmaAlma",
    "fastStart",
    "fastEnd",
    "fastStep",
    "slowStart",
    "slowEnd",
    "slowStep",
    "fastMaType",
    "slowMaType",
    "maSource",
    "objective",
    "mode",
    "topK",
    "includeAll",
    "exportAllCsvPath",
    "btnBrowseExportAll",
    "heatmapBins",
    "commission",
    "epsRel",
    "longOnly",
    "allowFlip",
    "tradeOnNext",
  ];
  for (const id of ids) {
    if (id === "btnRun" || id === "btnCancel") continue;
    const el = $(id);
    if (!el) continue;
    el.disabled = RUNNING;
  }
  for (const el of document.querySelectorAll("input[data-ma-param]")) {
    el.disabled = RUNNING;
  }
  for (const el of document.querySelectorAll("#sampleCsvs button")) {
    el.disabled = RUNNING;
  }
  if (RUNNING) {
    const wrap = $("progressWrap");
    const fill = $("progressFill");
    if (wrap) wrap.style.display = "block";
    if (fill) fill.style.width = "0%";
    const meta = $("progressMeta");
    if (meta) meta.textContent = "";
  } else {
    clearProgressUi();
  }
  updateDeviceIdEnabled();
}

function maSupportsKernel(id) {
  const info = MA_BY_ID[id];
  return info ? !!info.supports_cuda_kernel : false;
}

function maSupportsSweep(id) {
  const info = MA_BY_ID[id];
  return info ? !!info.supports_cuda_ma_sweep : false;
}

function maSupportsCpu(id) {
  const info = MA_BY_ID[id];
  return info ? !!info.supports_cpu : false;
}

function maSupportsCuda(id) {
  const info = MA_BY_ID[id];
  return info ? !!info.supports_cuda : false;
}

function maSelectableForBackend(id, backend) {
  const info = MA_BY_ID[id];
  if (!info) return false;
  if (!info.period_based || !info.single_output) return false;

  const b = String(backend || "CpuOnly");
  if (b === "CpuOnly") return maSupportsCpu(id);
  if (b === "GpuOnly") return maSupportsCuda(id);
  if (b === "GpuKernel") return maSupportsKernel(id);
  if (b === "Auto") return maSupportsCpu(id) || maSupportsCuda(id) || maSupportsKernel(id);
  return maSupportsCpu(id);
}

function applyBackendMaConstraints() {
  const backend = $("backend")?.value || "CpuOnly";
  const meta = CURRENT_DATA_META;
  const fastSel = $("fastMaType");
  const slowSel = $("slowMaType");
  if (!fastSel || !slowSel) return;

  for (const sel of [fastSel, slowSel]) {
    for (const opt of sel.options) {
      const id = opt.value;
      const info = MA_BY_ID[id];
      const baseOk = info && !!info.period_based && !!info.single_output;
      let dataOk = true;
      if (meta) {
        if (maRequiresVolumeField(id) && !meta.has_volume) dataOk = false;
        if (maRequiresHighLowField(id) && (!meta.has_high || !meta.has_low)) dataOk = false;
      }
      opt.disabled = !(baseOk && maSelectableForBackend(id, backend) && dataOk);
    }

    if (sel.value && sel.selectedOptions?.[0]?.disabled) {
      let picked = "";
      for (const opt of sel.options) {
        if (!opt.disabled) {
          picked = opt.value;
          break;
        }
      }
      if (picked) sel.value = picked;
    }
  }
  refreshMaParamsUi();
}

async function initMaSelects() {
  const invoke = tauriInvoke();
  if (!invoke) return;

  try {
    MA_LIST = await invoke("list_moving_averages");
    MA_LIST = Array.isArray(MA_LIST) ? MA_LIST : [];
    MA_BY_ID = {};
    for (const m of MA_LIST) MA_BY_ID[m.id] = m;

    const fastSel = $("fastMaType");
    const slowSel = $("slowMaType");
    if (!fastSel || !slowSel) return;

    const prevFast = (fastSel.value || "").trim();
    const prevSlow = (slowSel.value || "").trim();

    const mas = [...MA_LIST].sort((a, b) => (a.label || a.id).localeCompare(b.label || b.id));
    for (const sel of [fastSel, slowSel]) {
      sel.innerHTML = "";
      for (const m of mas) {
        const opt = document.createElement("option");
        opt.value = m.id;
        const isOk = !!m.period_based && !!m.single_output;
        opt.disabled = !isOk;
        opt.textContent = isOk ? `${m.label} (${m.id})` : `${m.label} (${m.id}) [not supported]`;
        sel.appendChild(opt);
      }
    }

    const isOk = (id) => {
      const m = MA_BY_ID[id];
      return m && !!m.period_based && !!m.single_output;
    };
    const defaultId = mas.find((m) => m.id === "sma") ? "sma" : (mas[0]?.id || "");
    fastSel.value = isOk(prevFast) ? prevFast : defaultId;
    slowSel.value = isOk(prevSlow) ? prevSlow : defaultId;
    applyBackendMaConstraints();
    refreshMaParamsUi();

    fastSel.addEventListener("change", refreshMaParamsUi);
    slowSel.addEventListener("change", refreshMaParamsUi);
  } catch (e) {
    setStatus(`Failed to load MA list: ${String(e)}`, "danger");
  }
}

function clearProgressUi() {
  const wrap = $("progressWrap");
  const fill = $("progressFill");
  const meta = $("progressMeta");
  if (fill) fill.style.width = "0%";
  if (meta) meta.textContent = "";
  if (wrap) wrap.style.display = "none";
}

function setProgressUi(processed, total, phase) {
  const wrap = $("progressWrap");
  const fill = $("progressFill");
  const meta = $("progressMeta");
  if (!wrap || !fill) return;

  const proc = Number(processed || 0);
  const tot = Number(total || 0);
  const ph = String(phase || "").trim();

  wrap.style.display = "block";

  let frac = tot > 0 ? proc / tot : 0;
  if (!Number.isFinite(frac)) frac = 0;
  if (frac < 0) frac = 0;
  if (frac > 1) frac = 1;
  fill.style.width = `${(frac * 100).toFixed(1)}%`;

  if (meta) {
    const pct = tot > 0 ? ` (${(frac * 100).toFixed(1)}%)` : "";
    meta.textContent = `${ph} ${proc}/${tot}${pct}`.trim();
  }
}

async function initProgressListener() {
  const listen = tauriListen();
  if (!listen) return;
  try {
    await listen("double_ma_progress", (event) => {
      const p = event?.payload || {};
      const processed = Number(p.processed_pairs || 0);
      const total = Number(p.total_pairs || 0);
      const phase = String(p.phase || "");
      const pct = total > 0 ? ((processed / total) * 100).toFixed(1) : "";
      setStatus(`${phase} ${processed}/${total}${pct ? ` (${pct}%)` : ""}`, "muted");
      setProgressUi(processed, total, phase);
    });
  } catch {
    // no-op (events are optional for this demo)
  }
}

async function initSampleCsvs() {
  const wrap = $("sampleCsvWrap");
  const body = $("sampleCsvs");
  if (!wrap || !body) return;

  const invoke = tauriInvoke();
  if (!invoke) {
    wrap.style.display = "none";
    return;
  }

  try {
    const list = await invoke("list_sample_csvs");
    const samples = Array.isArray(list) ? list : [];
    body.innerHTML = "";

    for (const s of samples) {
      const path = String(s?.path || "").trim();
      const label = String(s?.label || s?.id || "").trim();
      if (!path || !label) continue;

      const btn = document.createElement("button");
      btn.textContent = label;
      btn.title = path;
      btn.disabled = RUNNING;
      btn.addEventListener("click", async () => {
        if (RUNNING) return;
        if ($("csvPath")) $("csvPath").value = path;
        saveSettingsSoon();
        try {
          await loadCsv();
        } catch (e) {
          setStatus(String(e), "danger");
        }
      });
      body.appendChild(btn);
    }

    wrap.style.display = body.children.length ? "block" : "none";
  } catch {
    wrap.style.display = "none";
  }
}

function updateDeviceIdEnabled() {
  const backend = $("backend")?.value || "CpuOnly";
  const isGpu = ["GpuOnly", "GpuKernel", "Auto"].includes(backend);
  const dev = $("deviceId");
  if (dev) dev.disabled = RUNNING || !isGpu;
}

function readSettingsFromUi() {
  const obj = {};
  const fields = [
    "csvPath",
    "backend",
    "deviceId",
    "compareSelect",
    "fastStart",
    "fastEnd",
    "fastStep",
    "slowStart",
    "slowEnd",
    "slowStep",
    "fastMaType",
    "slowMaType",
    "maSource",
    "objective",
    "mode",
    "topK",
    "includeAll",
    "exportAllCsvPath",
    "heatmapBins",
    "commission",
    "epsRel",
    "filterMinPnl",
    "filterMinSharpe",
    "filterMaxDd",
    "filterMinTrades",
    "filterMinExposure",
    "paretoX",
    "paretoY",
  ];
  for (const id of fields) {
    const el = $(id);
    if (!el) continue;
    obj[id] = el.value;
  }
  const checks = ["longOnly", "allowFlip", "tradeOnNext"];
  for (const id of checks) {
    const el = $(id);
    if (!el) continue;
    obj[id] = !!el.checked;
  }
  obj.maParams = MA_PARAMS_STATE;
  return obj;
}

let SAVE_TIMER = null;
function saveSettingsSoon() {
  if (SAVE_TIMER) clearTimeout(SAVE_TIMER);
  SAVE_TIMER = setTimeout(() => {
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(readSettingsFromUi()));
    } catch {}
  }, 120);
}

function loadSettings() {
  let settings = null;
  try {
    settings = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "null");
  } catch {
    settings = null;
  }
  if (!settings || typeof settings !== "object") return;

  if (settings.maParams && typeof settings.maParams === "object") {
    MA_PARAMS_STATE = settings.maParams;
    if (!MA_PARAMS_STATE || typeof MA_PARAMS_STATE !== "object") MA_PARAMS_STATE = { fast: {}, slow: {} };
    if (!MA_PARAMS_STATE.fast || typeof MA_PARAMS_STATE.fast !== "object") MA_PARAMS_STATE.fast = {};
    if (!MA_PARAMS_STATE.slow || typeof MA_PARAMS_STATE.slow !== "object") MA_PARAMS_STATE.slow = {};
  }

  const legacyOffset = num(settings.almaOffset, NaN);
  const legacySigma = num(settings.almaSigma, NaN);
  if (Number.isFinite(legacyOffset) || Number.isFinite(legacySigma)) {
    for (const side of ["fast", "slow"]) {
      const st = ensureMaParamState(side, "alma");
      if (Number.isFinite(legacyOffset) && st.offset == null) st.offset = legacyOffset;
      if (Number.isFinite(legacySigma) && st.sigma == null) st.sigma = legacySigma;
    }
  }

  for (const [id, value] of Object.entries(settings)) {
    if (id === "maParams") continue;
    const el = $(id);
    if (!el) continue;
    if (el.type === "checkbox") el.checked = !!value;
    else el.value = String(value);
  }
  updateDeviceIdEnabled();
}

function applySettingsObject(settings) {
  if (!settings || typeof settings !== "object") return;

  if (settings.maParams && typeof settings.maParams === "object") {
    MA_PARAMS_STATE = settings.maParams;
    if (!MA_PARAMS_STATE || typeof MA_PARAMS_STATE !== "object") MA_PARAMS_STATE = { fast: {}, slow: {} };
    if (!MA_PARAMS_STATE.fast || typeof MA_PARAMS_STATE.fast !== "object") MA_PARAMS_STATE.fast = {};
    if (!MA_PARAMS_STATE.slow || typeof MA_PARAMS_STATE.slow !== "object") MA_PARAMS_STATE.slow = {};
  }

  for (const [id, value] of Object.entries(settings)) {
    if (id === "maParams") continue;
    const el = $(id);
    if (!el) continue;
    if (el.type === "checkbox") el.checked = !!value;
    else el.value = String(value);
  }

  updateDeviceIdEnabled();
  applyBackendMaConstraints();
  refreshMaParamsUi();
  saveSettingsSoon();
}

function applyRequestToUi(req) {
  if (!req || typeof req !== "object") return;

  const backend = String(req?.backend?.backend || "CpuOnly");
  const deviceId = Number(req?.backend?.device_id || 0);
  const fast = Array.isArray(req.fast_range) ? req.fast_range : [0, 0, 0];
  const slow = Array.isArray(req.slow_range) ? req.slow_range : [0, 0, 0];

  const fastMa = String((req.fast_ma_types?.[0] || "sma")).trim();
  const slowMa = String((req.slow_ma_types?.[0] || "sma")).trim();

  if ($("backend")) $("backend").value = backend;
  if ($("deviceId")) $("deviceId").value = String(deviceId);

  if ($("fastStart")) $("fastStart").value = String(Number(fast[0] || 0));
  if ($("fastEnd")) $("fastEnd").value = String(Number(fast[1] || 0));
  if ($("fastStep")) $("fastStep").value = String(Number(fast[2] || 0));

  if ($("slowStart")) $("slowStart").value = String(Number(slow[0] || 0));
  if ($("slowEnd")) $("slowEnd").value = String(Number(slow[1] || 0));
  if ($("slowStep")) $("slowStep").value = String(Number(slow[2] || 0));

  if ($("fastMaType")) $("fastMaType").value = fastMa;
  if ($("slowMaType")) $("slowMaType").value = slowMa;

  if ($("maSource")) $("maSource").value = String(req.ma_source || "close");
  if ($("objective")) $("objective").value = String(req.objective || "Sharpe");
  if ($("mode")) $("mode").value = String(req.mode || "Grid");

  if ($("topK")) $("topK").value = String(Number(req.top_k || 0));
  if ($("includeAll")) $("includeAll").value = (req.include_all ? "true" : "false");
  if ($("heatmapBins")) $("heatmapBins").value = String(Number(req.heatmap_bins || 0));
  if ($("exportAllCsvPath")) $("exportAllCsvPath").value = String(req.export_all_csv_path || "");

  const st = req.strategy || {};
  if ($("commission")) $("commission").value = String(num(st.commission, 0));
  if ($("epsRel")) $("epsRel").value = String(num(st.eps_rel, 0));
  if ($("longOnly")) $("longOnly").checked = !!st.long_only;
  if ($("allowFlip")) $("allowFlip").checked = !!st.allow_flip;
  if ($("tradeOnNext")) $("tradeOnNext").checked = !!st.trade_on_next_bar;

  const fastParams = req.fast_ma_params && typeof req.fast_ma_params === "object" ? req.fast_ma_params : null;
  const slowParams = req.slow_ma_params && typeof req.slow_ma_params === "object" ? req.slow_ma_params : null;
  if (fastParams) {
    const s = ensureMaParamState("fast", fastMa);
    for (const [k, v] of Object.entries(fastParams)) {
      const n = Number(v);
      if (Number.isFinite(n)) s[k] = n;
    }
  }
  if (slowParams) {
    const s = ensureMaParamState("slow", slowMa);
    for (const [k, v] of Object.entries(slowParams)) {
      const n = Number(v);
      if (Number.isFinite(n)) s[k] = n;
    }
  }

  updateDeviceIdEnabled();
  applyBackendMaConstraints();
  refreshMaParamsUi();
  saveSettingsSoon();
}

function getRecentCsvPaths() {
  try {
    const v = JSON.parse(localStorage.getItem(RECENT_CSV_KEY) || "[]");
    if (!Array.isArray(v)) return [];
    return v
      .filter((x) => typeof x === "string")
      .map((x) => x.trim())
      .filter((x) => !!x);
  } catch {
    return [];
  }
}

function setRecentCsvPaths(paths) {
  try {
    localStorage.setItem(RECENT_CSV_KEY, JSON.stringify(paths));
  } catch {}
}

function renderRecentCsvDatalist(paths) {
  const dl = $("csvRecentList");
  if (!dl) return;
  dl.innerHTML = "";
  const items = Array.isArray(paths) ? paths : [];
  for (const p of items) {
    const opt = document.createElement("option");
    opt.value = String(p || "");
    dl.appendChild(opt);
  }
}

function addRecentCsvPath(path) {
  const p = String(path || "").trim();
  if (!p) return;
  const prev = getRecentCsvPaths();
  const next = [p, ...prev.filter((x) => x !== p)].slice(0, RECENT_CSV_MAX);
  setRecentCsvPaths(next);
  renderRecentCsvDatalist(next);
}

function getHistory() {
  try {
    const v = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(v) ? v : [];
  } catch {
    return [];
  }
}

function setHistory(history) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  } catch {}
}

function populateHistorySelect(history, selectedTs) {
  const sel = $("historySelect");
  if (!sel) return;
  sel.innerHTML = "";

  const opt0 = document.createElement("option");
  opt0.value = "";
  opt0.textContent = history.length ? "Select a run..." : "No runs yet";
  sel.appendChild(opt0);

  for (const h of history) {
    const opt = document.createElement("option");
    opt.value = h.ts || "";
    opt.textContent = `${h.ts || "unknown"} | ${h.backend || ""} | ${h.pairs || 0} pairs`;
    sel.appendChild(opt);
  }
  sel.value = selectedTs || "";
}

function populateCompareSelect(history, selectedTs) {
  const sel = $("compareSelect");
  if (!sel) return;
  sel.innerHTML = "";

  const opt0 = document.createElement("option");
  opt0.value = "";
  opt0.textContent = history.length ? "Compare against..." : "No runs yet";
  sel.appendChild(opt0);

  for (const h of history) {
    const opt = document.createElement("option");
    opt.value = h.ts || "";
    opt.textContent = `${h.ts || "unknown"} | ${h.backend || ""} | ${h.pairs || 0} pairs`;
    sel.appendChild(opt);
  }
  sel.value = selectedTs || "";
}

function addHistoryEntry(req, res) {
  const history = getHistory();
  const ts = new Date().toISOString();
  const backend = req?.backend?.backend || "CpuOnly";
  const pairs = Number(res?.num_combos || 0);
  history.unshift({ ts, backend, pairs, req, res });
  while (history.length > 20) history.pop();
  setHistory(history);
  populateHistorySelect(history, ts);
  populateCompareSelect(history, $("compareSelect")?.value || "");
  CURRENT_RUN = { ts, req, res };
}

function scoreOf(metrics, objective) {
  const m = metrics || {};
  const obj = String(objective || "Sharpe");
  if (obj === "Pnl") return Number(m.pnl || 0);
  if (obj === "MaxDrawdown") return -Number(m.max_dd || 0);
  return Number(m.sharpe || 0);
}

function optNum(s) {
  const t = String(s ?? "").trim();
  if (!t) return NaN;
  const x = Number(t);
  return Number.isFinite(x) ? x : NaN;
}

function readTopFiltersFromUi() {
  const f = {
    minPnl: optNum($("filterMinPnl")?.value),
    minSharpe: optNum($("filterMinSharpe")?.value),
    maxDd: optNum($("filterMaxDd")?.value),
    minTrades: optNum($("filterMinTrades")?.value),
    minExposure: optNum($("filterMinExposure")?.value),
  };
  if (!Number.isFinite(f.minPnl)) delete f.minPnl;
  if (!Number.isFinite(f.minSharpe)) delete f.minSharpe;
  if (!Number.isFinite(f.maxDd)) delete f.maxDd;
  if (!Number.isFinite(f.minTrades)) delete f.minTrades;
  if (!Number.isFinite(f.minExposure)) delete f.minExposure;
  return f;
}

function applyTopFilters(rows, filters) {
  const f = filters || {};
  return rows.filter((row) => {
    const m = row.metrics || {};
    const pnl = Number(m.pnl || 0);
    const sharpe = Number(m.sharpe || 0);
    const dd = Number(m.max_dd || 0);
    const trades = Number(m.trades || 0);
    const expo = Number(m.exposure || 0);

    if (f.minPnl != null && pnl < f.minPnl) return false;
    if (f.minSharpe != null && sharpe < f.minSharpe) return false;
    if (f.maxDd != null && dd > f.maxDd) return false;
    if (f.minTrades != null && trades < f.minTrades) return false;
    if (f.minExposure != null && expo < f.minExposure) return false;
    return true;
  });
}

function collectTopRows(res) {
  const out = [];
  const top = Array.isArray(res?.top) ? res.top : [];
  for (let i = 0; i < top.length; i++) {
    const [params, metrics] = top[i] || [];
    out.push({ rank: i + 1, params: params || {}, metrics: metrics || {} });
  }
  if (!out.length && res?.best_params && res?.best_metrics) {
    out.push({ rank: 1, params: res.best_params, metrics: res.best_metrics });
  }
  return out;
}

function sortRows(rows, key, dir, objective) {
  const d = dir === "desc" ? -1 : 1;
  const k = String(key || "rank");
  const obj = String(objective || "Sharpe");
  const cmpNum = (a, b) => (a === b ? 0 : a < b ? -1 : 1);

  return [...rows].sort((ra, rb) => {
    const pa = ra.params || {};
    const pb = rb.params || {};
    const ma = ra.metrics || {};
    const mb = rb.metrics || {};

    if (k === "rank") return d * cmpNum(ra.rank, rb.rank);
    if (k === "fast") {
      const sa = `${pa.fast_ma_type || ""}`.toLowerCase();
      const sb = `${pb.fast_ma_type || ""}`.toLowerCase();
      if (sa !== sb) return d * cmpNum(sa, sb);
      return d * cmpNum(Number(pa.fast_len || 0), Number(pb.fast_len || 0));
    }
    if (k === "slow") {
      const sa = `${pa.slow_ma_type || ""}`.toLowerCase();
      const sb = `${pb.slow_ma_type || ""}`.toLowerCase();
      if (sa !== sb) return d * cmpNum(sa, sb);
      return d * cmpNum(Number(pa.slow_len || 0), Number(pb.slow_len || 0));
    }
    if (k === "pnl") return d * cmpNum(Number(ma.pnl || 0), Number(mb.pnl || 0));
    if (k === "sharpe") return d * cmpNum(Number(ma.sharpe || 0), Number(mb.sharpe || 0));
    if (k === "max_dd") return d * cmpNum(Number(ma.max_dd || 0), Number(mb.max_dd || 0));
    if (k === "trades") return d * cmpNum(Number(ma.trades || 0), Number(mb.trades || 0));
    if (k === "exposure") return d * cmpNum(Number(ma.exposure || 0), Number(mb.exposure || 0));
    if (k === "score") return d * cmpNum(scoreOf(ma, obj), scoreOf(mb, obj));
    return d * cmpNum(ra.rank, rb.rank);
  });
}

function paretoMetricValue(metrics, key) {
  const m = metrics || {};
  const k = String(key || "");
  if (k === "pnl") return Number(m.pnl || 0);
  if (k === "sharpe") return Number(m.sharpe || 0);
  if (k === "max_dd") return Number(m.max_dd || 0);
  if (k === "trades") return Number(m.trades || 0);
  if (k === "exposure") return Number(m.exposure || 0);
  return 0;
}

function paretoSign(key) {
  // Convert "lower is better" metrics into a "higher is better" score for Pareto dominance checks.
  const k = String(key || "");
  if (k === "max_dd") return -1;
  return 1;
}

function computePareto(rows, xKey, yKey) {
  const sx = paretoSign(xKey);
  const sy = paretoSign(yKey);
  const pts = rows.map((row) => {
    const m = row.metrics || {};
    const x = paretoMetricValue(m, xKey);
    const y = paretoMetricValue(m, yKey);
    return { row, x, y, tx: sx * x, ty: sy * y };
  });

  const front = [];
  for (let i = 0; i < pts.length; i++) {
    let dominated = false;
    for (let j = 0; j < pts.length; j++) {
      if (i === j) continue;
      const a = pts[j];
      const b = pts[i];
      if (a.tx >= b.tx && a.ty >= b.ty && (a.tx > b.tx || a.ty > b.ty)) {
        dominated = true;
        break;
      }
    }
    if (!dominated) front.push(pts[i]);
  }
  return { points: pts, front };
}

function renderPareto(rows) {
  const wrap = $("paretoWrap");
  const canvas = $("paretoCanvas");
  const meta = $("paretoMeta");
  if (!wrap || !canvas || !meta) return;

  const ptsRows = Array.isArray(rows) ? rows : [];
  if (!ptsRows.length) {
    wrap.style.display = "none";
    PARETO_POINTS_RENDERED = [];
    return;
  }

  wrap.style.display = "block";
  const xKey = $("paretoX")?.value || "max_dd";
  const yKey = $("paretoY")?.value || "pnl";
  const { points, front } = computePareto(ptsRows, xKey, yKey);

  meta.textContent = `points=${points.length} pareto=${front.length}`;

  const w = canvas.width || 1024;
  const h = canvas.height || 240;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of points) {
    if (Number.isFinite(p.x)) { if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x; }
    if (Number.isFinite(p.y)) { if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y; }
  }
  if (!Number.isFinite(minX) || !Number.isFinite(maxX) || !Number.isFinite(minY) || !Number.isFinite(maxY)) {
    PARETO_POINTS_RENDERED = [];
    return;
  }
  if (minX === maxX) { minX -= 1; maxX += 1; }
  if (minY === maxY) { minY -= 1; maxY += 1; }

  const padL = 36, padR = 10, padT = 10, padB = 22;
  const plotW = Math.max(1, w - padL - padR);
  const plotH = Math.max(1, h - padT - padB);
  const xOf = (x) => padL + ((x - minX) / (maxX - minX)) * plotW;
  const yOf = (y) => padT + (1 - (y - minY) / (maxY - minY)) * plotH;

  // Axes
  ctx.strokeStyle = "rgba(148, 163, 184, 0.35)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL + 0.5, padT + 0.5);
  ctx.lineTo(padL + 0.5, padT + plotH + 0.5);
  ctx.lineTo(padL + plotW + 0.5, padT + plotH + 0.5);
  ctx.stroke();

  // Points
  const frontKeys = new Set(front.map((p) => paramsKey(p.row.params)));
  PARETO_POINTS_RENDERED = [];

  for (const p of points) {
    const sx = xOf(p.x);
    const sy = yOf(p.y);
    const key = paramsKey(p.row.params);
    const isFront = frontKeys.has(key);
    ctx.fillStyle = isFront ? "rgba(56, 189, 248, 0.95)" : "rgba(148, 163, 184, 0.55)";
    const r = isFront ? 3 : 2;
    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.fill();
    PARETO_POINTS_RENDERED.push({ sx, sy, row: p.row, key });
  }

  // Selected highlight
  if (DRILL_SELECTED_KEY) {
    const hit = PARETO_POINTS_RENDERED.find((p) => p.key === DRILL_SELECTED_KEY);
    if (hit) {
      ctx.strokeStyle = "rgba(52, 211, 153, 0.95)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(hit.sx, hit.sy, 6, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  canvas.onclick = (ev) => {
    const rect = canvas.getBoundingClientRect();
    const cx = ((ev.clientX - rect.left) / rect.width) * w;
    const cy = ((ev.clientY - rect.top) / rect.height) * h;
    let best = null;
    let bestD2 = 1e18;
    for (const p of PARETO_POINTS_RENDERED) {
      const dx = p.sx - cx;
      const dy = p.sy - cy;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestD2) { bestD2 = d2; best = p; }
    }
    if (best && bestD2 <= 12 * 12) {
      DRILL_SELECTED_KEY = best.key;
      requestDrilldown(best.row.params);
      renderTopTable(CURRENT_RUN?.res, CURRENT_RUN?.req);
    }
  };
}

function renderTopTable(res, req) {
  TOP_ROWS = collectTopRows(res);
  const obj = req?.objective || "Sharpe";
  const sorted = sortRows(TOP_ROWS, TOP_SORT.key, TOP_SORT.dir, obj);
  const filters = readTopFiltersFromUi();
  TOP_ROWS_FILTERED = applyTopFilters(sorted, filters);

  const meta = $("topFilterMeta");
  if (meta) {
    meta.textContent = `showing ${TOP_ROWS_FILTERED.length}/${sorted.length} rows`;
  }

  const tbody = $("topTableBody");
  if (!tbody) return;
  tbody.innerHTML = "";

  if (!TOP_ROWS_FILTERED.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 8;
    td.className = "left muted";
    td.textContent = sorted.length ? "No rows match the current filters." : "No rows.";
    tr.appendChild(td);
    tbody.appendChild(tr);
    renderPareto([]);
    return;
  }

  for (const row of TOP_ROWS_FILTERED) {
    const p = row.params || {};
    const m = row.metrics || {};
    const key = paramsKey(p);
    const tr = document.createElement("tr");

    const tdRank = document.createElement("td");
    tdRank.textContent = String(row.rank);
    tdRank.className = "left mono";

    const tdFast = document.createElement("td");
    tdFast.textContent = `${p.fast_ma_type || "?"}(${p.fast_len || "?"})`;
    tdFast.className = "left mono";

    const tdSlow = document.createElement("td");
    tdSlow.textContent = `${p.slow_ma_type || "?"}(${p.slow_len || "?"})`;
    tdSlow.className = "left mono";

    const tdPnl = document.createElement("td");
    tdPnl.textContent = fmt(m.pnl, 6);
    tdPnl.className = "mono";

    const tdSharpe = document.createElement("td");
    tdSharpe.textContent = fmt(m.sharpe, 6);
    tdSharpe.className = "mono";

    const tdDd = document.createElement("td");
    tdDd.textContent = fmt(m.max_dd, 6);
    tdDd.className = "mono";

    const tdTrades = document.createElement("td");
    tdTrades.textContent = String(Number(m.trades || 0));
    tdTrades.className = "mono";

    const tdExpo = document.createElement("td");
    tdExpo.textContent = fmt(m.exposure, 4);
    tdExpo.className = "mono";

    tr.appendChild(tdRank);
    tr.appendChild(tdFast);
    tr.appendChild(tdSlow);
    tr.appendChild(tdPnl);
    tr.appendChild(tdSharpe);
    tr.appendChild(tdDd);
    tr.appendChild(tdTrades);
    tr.appendChild(tdExpo);
    if (key && key === DRILL_SELECTED_KEY) tr.classList.add("selected");
    tr.addEventListener("click", () => {
      for (const el of tbody.querySelectorAll("tr.selected")) el.classList.remove("selected");
      tr.classList.add("selected");
      DRILL_SELECTED_KEY = key;
      requestDrilldown(p);
    });
    tbody.appendChild(tr);
  }

  renderPareto(TOP_ROWS_FILTERED);
}

function renderCompare() {
  const el = $("compareSummary");
  if (!el) return;

  const history = getHistory();
  const baseTs = $("compareSelect")?.value || "";
  const base = history.find((h) => h.ts === baseTs) || null;
  const cur = CURRENT_RUN;

  if (!baseTs) {
    el.textContent = "-";
    return;
  }
  if (!base) {
    el.textContent = "Baseline run not found (it may have been trimmed from history).";
    return;
  }
  if (!cur?.res) {
    el.textContent = "Select a current run to compare.";
    return;
  }

  const bReq = base.req || {};
  const bRes = base.res || {};
  const cReq = cur.req || {};
  const cRes = cur.res || {};

  const bm = bRes.best_metrics || {};
  const cm = cRes.best_metrics || {};
  const bp = bRes.best_params || {};
  const cp = cRes.best_params || {};

  const bObj = bReq.objective || "Sharpe";
  const cObj = cReq.objective || "Sharpe";
  const bScore = scoreOf(bm, bObj);
  const cScore = scoreOf(cm, cObj);

  const bRt = Number(bRes.runtime_ms || 0);
  const cRt = Number(cRes.runtime_ms || 0);
  const rtRatio = bRt > 0 && cRt > 0 ? (bRt / cRt) : 0;

  const delta = (a, b) => Number(a || 0) - Number(b || 0);

  const lines = [];
  lines.push(`baseline: ${base.ts} | backend=${bReq?.backend?.backend || base.backend || "?"} | combos=${Number(bRes.num_combos || 0)} | runtime_ms=${bRt || 0}`);
  lines.push(`current:  ${cur.ts} | backend=${cReq?.backend?.backend || "?"} | combos=${Number(cRes.num_combos || 0)} | runtime_ms=${cRt || 0}${rtRatio ? ` | speedup=${rtRatio.toFixed(2)}x` : ""}`);
  lines.push("");
  lines.push(`baseline best: ${bp.fast_ma_type || "?"}(${bp.fast_len || "?"}) vs ${bp.slow_ma_type || "?"}(${bp.slow_len || "?"}) | pnl=${fmt(bm.pnl, 6)} sharpe=${fmt(bm.sharpe, 6)} dd=${fmt(bm.max_dd, 6)} trades=${Number(bm.trades || 0)} expo=${fmt(bm.exposure, 4)} score(${bObj})=${fmt(bScore, 6)}`);
  lines.push(`current best:  ${cp.fast_ma_type || "?"}(${cp.fast_len || "?"}) vs ${cp.slow_ma_type || "?"}(${cp.slow_len || "?"}) | pnl=${fmt(cm.pnl, 6)} sharpe=${fmt(cm.sharpe, 6)} dd=${fmt(cm.max_dd, 6)} trades=${Number(cm.trades || 0)} expo=${fmt(cm.exposure, 4)} score(${cObj})=${fmt(cScore, 6)}`);
  lines.push("");
  lines.push(`deltas (current - baseline): pnl=${fmt(delta(cm.pnl, bm.pnl), 6)} sharpe=${fmt(delta(cm.sharpe, bm.sharpe), 6)} dd=${fmt(delta(cm.max_dd, bm.max_dd), 6)} trades=${delta(cm.trades, bm.trades)} expo=${fmt(delta(cm.exposure, bm.exposure), 4)}`);

  if (bObj === cObj) {
    lines.push(`objective delta: score(${cObj})=${fmt(delta(cScore, bScore), 6)}`);
  } else {
    lines.push(`objective delta: (objectives differ: baseline=${bObj}, current=${cObj})`);
  }

  el.textContent = lines.join("\n");
}

function paramsKey(p) {
  const pp = p || {};
  return `${pp.fast_ma_type || ""}|${pp.fast_len || 0}|${pp.slow_ma_type || ""}|${pp.slow_len || 0}`;
}

function desiredDrillBins() {
  const canvas = $("equityCanvas");
  if (!canvas) return 1024;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.floor((rect.width || 0) * dpr);
  if (w < 64) return 1024;
  return Math.max(64, Math.min(4096, w));
}

function setDrillVisible(visible) {
  const wrap = $("drillWrap");
  if (!wrap) return;
  wrap.style.display = visible ? "block" : "none";
}

function setDrillStatus(msg, kind) {
  const el = $("drillStatus");
  if (!el) return;
  el.textContent = msg;
  el.className = "muted " + (kind || "");
}

function drawMinMaxChart(canvas, minmax, opts) {
  const mm = Array.isArray(minmax) ? minmax : [];
  const bins = Math.floor(mm.length / 2);
  if (!canvas || bins <= 0) return;

  const height = Number(opts?.height || 160);
  canvas.width = bins;
  canvas.height = height;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, bins, height);

  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < mm.length; i++) {
    const v = Number(mm[i]);
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return;
  if (min === max) {
    min -= 1;
    max += 1;
  }

  const pad = 6;
  const h = height;
  const scale = (h - 1 - 2 * pad) / (max - min);
  const yOf = (v) => pad + (max - v) * scale;

  if (Number.isFinite(opts?.baseline)) {
    const by = yOf(Number(opts.baseline));
    ctx.strokeStyle = "rgba(148, 163, 184, 0.45)";
    ctx.beginPath();
    ctx.moveTo(0, by + 0.5);
    ctx.lineTo(bins, by + 0.5);
    ctx.stroke();
  }

  ctx.strokeStyle = opts?.stroke || "rgba(52, 211, 153, 0.95)";
  ctx.lineWidth = 1;
  for (let x = 0; x < bins; x++) {
    const a = Number(mm[2 * x + 0]);
    const b = Number(mm[2 * x + 1]);
    if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
    const y1 = yOf(a);
    const y2 = yOf(b);
    ctx.beginPath();
    ctx.moveTo(x + 0.5, y1);
    ctx.lineTo(x + 0.5, y2);
    ctx.stroke();
  }
}

function renderDrilldown(curves, params) {
  const c = curves || {};
  const eq = c.equity || {};
  const dd = c.drawdown || {};

  const bins = Number(eq.bins || 0);
  if (!bins) {
    setDrillVisible(false);
    return;
  }

  setDrillVisible(true);
  const p = params || {};
  const meta = $("drillMeta");
  if (meta) {
    meta.textContent =
      `${p.fast_ma_type || "?"}(${p.fast_len || "?"}) vs ${p.slow_ma_type || "?"}(${p.slow_len || "?"}) | ` +
      `bins=${bins} start_t=${Number(eq.start_t || 0)} end_t=${Number(eq.end_t || 0)} (min/max per bin)`;
  }

  drawMinMaxChart($("equityCanvas"), eq.minmax, {
    stroke: "rgba(52, 211, 153, 0.95)",
    baseline: 1.0,
    height: 160,
  });
  drawMinMaxChart($("drawdownCanvas"), dd.minmax, {
    stroke: "rgba(251, 113, 133, 0.95)",
    baseline: 0.0,
    height: 160,
  });

  renderTrades(c);
}

function renderTrades(curves) {
  const wrap = $("tradeStatsWrap");
  const statsEl = $("tradeStats");
  const body = $("tradeTableBody");
  if (!wrap || !statsEl || !body) return;

  const trades = Array.isArray(curves?.trades) ? curves.trades : [];
  const stats = curves?.trade_stats && typeof curves.trade_stats === "object" ? curves.trade_stats : null;

  if (!trades.length && !stats) {
    wrap.style.display = "none";
    statsEl.textContent = "-";
    body.innerHTML = "";
    return;
  }

  wrap.style.display = "block";

  if (stats) {
    const closed = Number(stats.closed_trades || 0);
    const open = Number(stats.open_trades || 0);
    const wins = Number(stats.wins || 0);
    const losses = Number(stats.losses || 0);
    const winRate = Number(stats.win_rate || 0);
    const avgWin = Number(stats.avg_win || 0);
    const avgLoss = Number(stats.avg_loss || 0);
    const pf = Number(stats.profit_factor);
    const exp = Number(stats.expectancy || 0);

    const pfStr = Number.isFinite(pf) ? (pf === Infinity ? "inf" : pf.toFixed(3)) : "-";
    const wrStr = Number.isFinite(winRate) ? `${(winRate * 100).toFixed(1)}%` : "-";
    const avgWinStr = Number.isFinite(avgWin) ? `${(avgWin * 100).toFixed(2)}%` : "-";
    const avgLossStr = Number.isFinite(avgLoss) ? `${(avgLoss * 100).toFixed(2)}%` : "-";
    const expStr = Number.isFinite(exp) ? `${(exp * 100).toFixed(2)}%` : "-";

    statsEl.textContent =
      `closed=${closed} open=${open} wins=${wins} losses=${losses} ` +
      `win_rate=${wrStr} avg_win=${avgWinStr} avg_loss=${avgLossStr} ` +
      `pf=${pfStr} expectancy=${expStr}`;
  } else {
    statsEl.textContent = `trades=${trades.length}`;
  }

  const maxRows = 200;
  const start = Math.max(0, trades.length - maxRows);
  const shown = trades.slice(start);

  body.innerHTML = "";
  for (let i = 0; i < shown.length; i++) {
    const t = shown[i] || {};
    const idx = start + i + 1;
    const dirVal = Number(t.direction || 0);
    const dir = dirVal > 0 ? "Long" : dirVal < 0 ? "Short" : "Flat";
    const entry = Number(t.entry_t || 0);
    const exit = Number(t.exit_t || 0);
    const bars = Number(t.bars || 0);
    const pnl = Number(t.pnl || 0);
    const open = !!t.open;
    const state = open ? "open" : "closed";

    const pnlStr = Number.isFinite(pnl) ? `${(pnl * 100).toFixed(2)}%` : "-";

    const tr = document.createElement("tr");

    const tdN = document.createElement("td");
    tdN.className = "left mono";
    tdN.textContent = String(idx);

    const tdDir = document.createElement("td");
    tdDir.className = "left mono";
    tdDir.textContent = dir;

    const tdEntry = document.createElement("td");
    tdEntry.className = "left mono";
    tdEntry.textContent = String(entry);

    const tdExit = document.createElement("td");
    tdExit.className = "left mono";
    tdExit.textContent = String(exit);

    const tdBars = document.createElement("td");
    tdBars.className = "mono";
    tdBars.textContent = String(bars);

    const tdPnl = document.createElement("td");
    tdPnl.className = "mono";
    tdPnl.textContent = pnlStr;
    if (Number.isFinite(pnl)) {
      if (pnl > 0) tdPnl.style.color = "rgba(52, 211, 153, 0.95)";
      else if (pnl < 0) tdPnl.style.color = "rgba(251, 113, 133, 0.95)";
    }

    const tdState = document.createElement("td");
    tdState.className = "left mono";
    tdState.textContent = state;

    tr.appendChild(tdN);
    tr.appendChild(tdDir);
    tr.appendChild(tdEntry);
    tr.appendChild(tdExit);
    tr.appendChild(tdBars);
    tr.appendChild(tdPnl);
    tr.appendChild(tdState);

    body.appendChild(tr);
  }
}

async function requestDrilldown(params, runReqOverride) {
  const invoke = tauriInvoke();
  if (!invoke) return;

  const runReq = runReqOverride || CURRENT_RUN?.req;
  if (!runReq || !runReq.data_id) return;
  if (!params || !params.fast_ma_type || !params.slow_ma_type) return;

  const seq = ++DRILL_SEQ;
  setDrillVisible(true);
  setDrillStatus("Computing drilldown…", "muted");

  const bins = desiredDrillBins();
  const asObj = (x) => (x && typeof x === "object" ? x : null);
  const legacyAlma = {
    offset: num(runReq?.alma_offset ?? runReq?.almaOffset, 0.85),
    sigma: num(runReq?.alma_sigma ?? runReq?.almaSigma, 6.0),
  };
  const fastId = String(params.fast_ma_type || "").trim().toLowerCase();
  const slowId = String(params.slow_ma_type || "").trim().toLowerCase();
  const fastParams =
    asObj(runReq.fast_ma_params) ||
    asObj(runReq.fastMaParams) ||
    (fastId === "alma" &&
    (runReq?.alma_offset != null || runReq?.almaOffset != null || runReq?.alma_sigma != null || runReq?.almaSigma != null)
      ? legacyAlma
      : null);
  const slowParams =
    asObj(runReq.slow_ma_params) ||
    asObj(runReq.slowMaParams) ||
    (slowId === "alma" &&
    (runReq?.alma_offset != null || runReq?.almaOffset != null || runReq?.alma_sigma != null || runReq?.almaSigma != null)
      ? legacyAlma
      : null);
  const req = {
    data_id: runReq.data_id,
    params: {
      fast_len: Number(params.fast_len || 0),
      slow_len: Number(params.slow_len || 0),
      fast_ma_type: String(params.fast_ma_type || ""),
      slow_ma_type: String(params.slow_ma_type || ""),
    },
    ma_source: String(runReq.ma_source || "close"),
    fast_ma_params: fastParams,
    slow_ma_params: slowParams,
    strategy: runReq.strategy || {},
    bins,
  };

  try {
    const curves = await invoke("compute_double_ma_drilldown", { req });
    if (seq !== DRILL_SEQ) return;
    setDrillStatus("Done.", "ok");
    renderDrilldown(curves, params);
  } catch (e) {
    if (seq !== DRILL_SEQ) return;
    setDrillStatus(String(e), "danger");
  }
}

function hslToRgb(h, s, l) {
  const hh = ((h % 360) + 360) % 360;
  const ss = Math.max(0, Math.min(1, s));
  const ll = Math.max(0, Math.min(1, l));
  const c = (1 - Math.abs(2 * ll - 1)) * ss;
  const x = c * (1 - Math.abs(((hh / 60) % 2) - 1));
  const m = ll - c / 2;
  let r = 0, g = 0, b = 0;
  if (hh < 60) [r, g, b] = [c, x, 0];
  else if (hh < 120) [r, g, b] = [x, c, 0];
  else if (hh < 180) [r, g, b] = [0, c, x];
  else if (hh < 240) [r, g, b] = [0, x, c];
  else if (hh < 300) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  return [
    Math.round((r + m) * 255),
    Math.round((g + m) * 255),
    Math.round((b + m) * 255),
  ];
}

function colorFor(t) {
  const x = Math.max(0, Math.min(1, t));
  const h = 220 * (1 - x);
  return hslToRgb(h, 0.85, 0.52);
}

function renderHeatmap(res, req) {
  const wrap = $("heatmapWrap");
  const canvas = $("heatmapCanvas");
  const meta = $("heatmapMeta");
  if (!wrap || !canvas || !meta) return;

  const hm = res?.heatmap;
  if (!hm || !Array.isArray(hm.values) || !hm.bins_fast || !hm.bins_slow) {
    wrap.style.display = "none";
    return;
  }

  const binsFast = Number(hm.bins_fast);
  const binsSlow = Number(hm.bins_slow);
  const values = hm.values;
  if (!binsFast || !binsSlow || values.length !== binsFast * binsSlow) {
    wrap.style.display = "none";
    return;
  }

  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (typeof v !== "number" || !Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    wrap.style.display = "none";
    return;
  }
  if (min === max) {
    min -= 1;
    max += 1;
  }

  wrap.style.display = "block";
  HEATMAP_META_BASE = `Objective heatmap (${req?.objective || "Sharpe"}) | fast=${hm.fast_min}..${hm.fast_max} slow=${hm.slow_min}..${hm.slow_max} | score=${fmt(min, 4)}..${fmt(max, 4)}`;
  meta.textContent = HEATMAP_META_BASE;

  canvas.width = binsSlow;
  canvas.height = binsFast;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(binsSlow, binsFast);
  for (let f = 0; f < binsFast; f++) {
    for (let s = 0; s < binsSlow; s++) {
      const idx = f * binsSlow + s;
      const v = values[idx];
      const px = (f * binsSlow + s) * 4;
      if (typeof v !== "number" || !Number.isFinite(v)) {
        img.data[px + 0] = 11;
        img.data[px + 1] = 18;
        img.data[px + 2] = 32;
        img.data[px + 3] = 255;
        continue;
      }
      const t = (v - min) / (max - min);
      const [r, g, b] = colorFor(t);
      img.data[px + 0] = r;
      img.data[px + 1] = g;
      img.data[px + 2] = b;
      img.data[px + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);

  canvas.onmousemove = (ev) => {
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor(((ev.clientX - rect.left) / rect.width) * binsSlow);
    const y = Math.floor(((ev.clientY - rect.top) / rect.height) * binsFast);
    if (x < 0 || y < 0 || x >= binsSlow || y >= binsFast) {
      meta.textContent = HEATMAP_META_BASE;
      return;
    }
    const v = values[y * binsSlow + x];
    const val = typeof v === "number" && Number.isFinite(v) ? fmt(v, 6) : "-";
    meta.textContent = `${HEATMAP_META_BASE} | hover: f_bin=${y} s_bin=${x} score=${val}`;
  };
  canvas.onmouseleave = () => {
    meta.textContent = HEATMAP_META_BASE;
  };
}

function renderBest(res, req) {
  const el = $("bestSummary");
  if (!el) return;
  const p = res?.best_params || {};
  const m = res?.best_metrics || {};
  const obj = req?.objective || "-";
  const backendUsed = String(res?.backend_used || "-");
  const modeUsed = String(res?.mode_used || "-");
  const combos = Number(res?.num_combos || 0);
  const runtimeMs = Number(res?.runtime_ms || 0);
  const s = fmt(scoreOf(m, obj), 6);
  el.textContent =
    `best: ${p.fast_ma_type || "?"}(${p.fast_len || "?"}) vs ${p.slow_ma_type || "?"}(${p.slow_len || "?"}) | ` +
    `pnl=${fmt(m.pnl, 6)} sharpe=${fmt(m.sharpe, 6)} dd=${fmt(m.max_dd, 6)} trades=${Number(m.trades || 0)} expo=${fmt(m.exposure, 4)} score(${obj})=${s} | ` +
    `backend=${backendUsed} mode=${modeUsed} combos=${combos} runtime_ms=${runtimeMs}`;
}

function renderRun(res, req) {
  const best = res?.best_params;
  DRILL_SELECTED_KEY = best ? paramsKey(best) : "";
  renderBest(res, req);
  renderTopTable(res, req);
  renderHeatmap(res, req);
  renderCompare();
  if (best) {
    requestDrilldown(best, req);
  } else {
    setDrillVisible(false);
  }
}

function getRequest() {
  const dataIdText = $("dataId").textContent.trim();
  const dataId = dataIdText === "—" ? "" : dataIdText;
  if (!dataId || dataId === "-") throw new Error("Load CSV first (missing data id).");

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

  const fastPeriods = expandRangeU16Like(fast[0], fast[1], fast[2]);
  const slowPeriods = expandRangeU16Like(slow[0], slow[1], slow[2]);
  const pairs = countPairsFastLtSlow(fastPeriods, slowPeriods);
  if (!pairs) throw new Error("No valid parameter combinations (fast must be < slow).");

  const fastMaType = ($("fastMaType")?.value || "").trim();
  const slowMaType = ($("slowMaType")?.value || "").trim();
  if (!fastMaType) throw new Error("Select a fast MA type.");
  if (!slowMaType) throw new Error("Select a slow MA type.");

  if (backend === "CpuOnly") {
    if (!maSupportsCpu(fastMaType) || !maSupportsCpu(slowMaType)) {
      throw new Error("CPU backend does not support one of the selected MA types. Choose a supported MA or switch backend.");
    }
  }

  if (backend === "GpuOnly") {
    if (!maSupportsCuda(fastMaType) || !maSupportsCuda(slowMaType)) {
      throw new Error("GPU backend requires CUDA support for both selected MAs. Choose a supported MA or switch backend.");
    }
  }

  if (backend === "GpuKernel") {
    if (!maSupportsKernel(fastMaType) || !maSupportsKernel(slowMaType)) {
      throw new Error("GPU (kernel) currently supports only sma/ema/wma/alma. Choose a supported MA or switch backend.");
    }
  } else if (backend === "GpuOnly") {
    if (!maSupportsCuda(fastMaType) || !maSupportsCuda(slowMaType)) {
      throw new Error("GPU (MA sweep) does not support the selected MA(s). Switch backend or choose a supported MA.");
    }
  } else if (backend === "CpuOnly") {
    if (!maSupportsCpu(fastMaType) || !maSupportsCpu(slowMaType)) {
      throw new Error("CPU backend does not support the selected MA(s). Switch backend or choose a supported MA.");
    }
  } else if (backend === "Auto") {
    const okFast = maSupportsCpu(fastMaType) || maSupportsCuda(fastMaType) || maSupportsKernel(fastMaType);
    const okSlow = maSupportsCpu(slowMaType) || maSupportsCuda(slowMaType) || maSupportsKernel(slowMaType);
    if (!okFast || !okSlow) {
      throw new Error("Auto backend cannot run the selected MA(s). Choose a period-based, single-output MA that supports CPU or CUDA.");
    }
  }

  const maSource = $("maSource").value;
  validateCandleFieldRequirements(CURRENT_DATA_META, fastMaType, slowMaType, maSource);

  const objective = $("objective").value;
  const mode = $("mode").value;

  const topK = Number($("topK").value || 0);
  const includeAll = $("includeAll").value === "true";
  if (includeAll && pairs > 2_000_000) {
    throw new Error(`includeAll=true would return ${pairs} rows; disable it or reduce ranges.`);
  }

  const exportAllCsvPath = ($("exportAllCsvPath")?.value || "").trim();
  if (exportAllCsvPath) {
    if (includeAll) throw new Error("exportAllCsvPath requires includeAll=false (stream export avoids huge RAM usage).");
    if (mode !== "Grid") throw new Error("exportAllCsvPath requires mode=Grid.");
    if (backend !== "GpuKernel") throw new Error("exportAllCsvPath is currently supported only for the GPU (kernel) backend.");
  }

  const heatmapBins = Math.max(0, Math.floor(num($("heatmapBins")?.value, 0)));
  if (heatmapBins > 512) throw new Error("heatmapBins must be <= 512.");
  if (mode === "CoarseToFine") {
    if (includeAll) throw new Error("Coarse→Fine mode requires includeAll=false.");
    if (heatmapBins > 0) throw new Error("Coarse→Fine mode currently requires heatmapBins=0.");
    if (exportAllCsvPath) throw new Error("Coarse→Fine mode cannot be used with exportAllCsvPath.");
  }

  const commission = num($("commission")?.value, 0);
  const epsRel = num($("epsRel")?.value, 0);
  if (!Number.isFinite(commission) || commission < 0 || commission >= 1) throw new Error("commission must be in [0, 1).");
  if (!Number.isFinite(epsRel) || epsRel < 0) throw new Error("eps_rel must be >= 0.");

  const strategy = {
    long_only: !!$("longOnly")?.checked,
    allow_flip: !!$("allowFlip")?.checked,
    trade_on_next_bar: !!$("tradeOnNext")?.checked,
    commission,
    eps_rel: epsRel,
  };

  const backendObj =
    backend === "GpuOnly" || backend === "GpuKernel" || backend === "Auto"
      ? { backend, device_id: deviceId }
      : { backend: "CpuOnly" };

  return {
    backend: backendObj,
    data_id: dataId,
    fast_range: fast,
    slow_range: slow,
    fast_ma_types: [fastMaType],
    slow_ma_types: [slowMaType],
    ma_source: maSource,
    export_all_csv_path: exportAllCsvPath || null,
    fast_ma_params: resolvedMaParams("fast", fastMaType),
    slow_ma_params: resolvedMaParams("slow", slowMaType),
    strategy,
    objective,
    mode,
    top_k: topK,
    include_all: includeAll,
    heatmap_bins: heatmapBins,
  };
}

async function loadCsv() {
  const invoke = tauriInvoke();
  if (!invoke) throw new Error("Tauri invoke not available (window.__TAURI__ missing).");
  const path = $("csvPath").value.trim();
  if (!path) throw new Error("Enter a CSV path.");
  setStatus("Loading CSV...", "muted");
  const resp = await invoke("load_price_data", { path });
  const id = String(resp?.id || "").trim();
  const meta = resp?.meta && typeof resp.meta === "object" ? resp.meta : null;
  CURRENT_DATA_META = meta;
  $("dataId").textContent = id;
  renderDataMeta(meta);
  applyMaSourceConstraints();
  applyBackendMaConstraints();
  setStatus("Loaded.", "ok");
  $("output").textContent = pretty({ data_id: id, meta });
  addRecentCsvPath(path);
  saveSettingsSoon();
}

async function runOptimization() {
  const invoke = tauriInvoke();
  if (!invoke) throw new Error("Tauri invoke not available (window.__TAURI__ missing).");
  const req = getRequest();
  setRunning(true);
  try {
    setStatus("Running optimization...", "muted");
    const t0 = performance.now();
    const res = await invoke("run_double_ma_optimization", { req });
    const dt = performance.now() - t0;
    setStatus(`Done in ${dt.toFixed(0)}ms (frontend)`, "ok");
    $("output").textContent = pretty(res);
    addHistoryEntry(req, res);
    renderRun(res, req);
    return res;
  } finally {
    setRunning(false);
  }
}

$("btnLoad").addEventListener("click", async () => {
  try {
    await loadCsv();
  } catch (e) {
    setStatus(String(e), "danger");
  }
});

$("btnBrowseCsv")?.addEventListener("click", async () => {
  const invoke = tauriInvoke();
  if (!invoke) {
    setStatus("Tauri invoke not available (window.__TAURI__ missing).", "danger");
    return;
  }
  try {
    const picked = await invoke("pick_csv_file");
    const path = String(picked || "").trim();
    if (!path) return;
    $("csvPath").value = path;
    saveSettingsSoon();
    await loadCsv();
  } catch (e) {
    setStatus(String(e), "danger");
  }
});

$("btnBrowseExportAll")?.addEventListener("click", async () => {
  const invoke = tauriInvoke();
  if (!invoke) {
    setStatus("Tauri invoke not available (window.__TAURI__ missing).", "danger");
    return;
  }
  try {
    const picked = await invoke("pick_save_csv", { defaultName: "all_pairs.csv" });
    const path = String(picked || "").trim();
    if (!path) return;
    $("exportAllCsvPath").value = path;
    saveSettingsSoon();
    setStatus("Selected export path.", "muted");
  } catch (e) {
    setStatus(String(e), "danger");
  }
});

$("btnPresetKernelSmaAlma")?.addEventListener("click", () => {
  if ($("backend")) $("backend").value = "GpuKernel";
  updateDeviceIdEnabled();

  if ($("maSource")) $("maSource").value = "close";

  if ($("fastMaType") && MA_BY_ID["sma"]) $("fastMaType").value = "sma";
  if ($("slowMaType") && MA_BY_ID["alma"]) $("slowMaType").value = "alma";

  applyMaSourceConstraints();
  applyBackendMaConstraints();
  refreshMaParamsUi();
  saveSettingsSoon();
  setStatus("Applied preset: GPU kernel (SMA/ALMA)", "muted");
});

$("btnClear").addEventListener("click", () => {
  clearProgressUi();
  $("dataId").textContent = "—";
  CURRENT_DATA_META = null;
  renderDataMeta(null);
  applyMaSourceConstraints();
  applyBackendMaConstraints();
  $("output").textContent = "{}";
  $("bestSummary").textContent = "-";
  $("topTableBody").innerHTML = "";
  if ($("historySelect")) $("historySelect").value = "";
  CURRENT_RUN = null;
  TOP_ROWS = [];
  TOP_ROWS_FILTERED = [];
  PARETO_POINTS_RENDERED = [];
  const meta = $("topFilterMeta");
  if (meta) meta.textContent = "-";
  const pareto = $("paretoWrap");
  if (pareto) pareto.style.display = "none";
  DRILL_SELECTED_KEY = "";
  DRILL_SEQ++;
  setDrillVisible(false);
  $("heatmapWrap").style.display = "none";
  renderCompare();
  setStatus("Ready.", "");
});

$("btnRun").addEventListener("click", async () => {
  try {
    await runOptimization();
  } catch (e) {
    setStatus(String(e), "danger");
  }
});

$("btnCancel").addEventListener("click", async () => {
  const invoke = tauriInvoke();
  if (!invoke) return;
  try {
    $("btnCancel").disabled = true;
    setStatus("Cancel requested... (waiting for current tile to finish)", "muted");
    await invoke("cancel_double_ma_optimization");
  } catch (e) {
    setStatus(String(e), "danger");
  }
});


$("backend").addEventListener("change", () => {
  updateDeviceIdEnabled();
  applyBackendMaConstraints();
  saveSettingsSoon();
});
$("deviceId").addEventListener("change", saveSettingsSoon);

for (const id of [
  "csvPath",
  "compareSelect",
  "fastMaType",
  "slowMaType",
  "fastStart",
  "fastEnd",
  "fastStep",
  "slowStart",
  "slowEnd",
  "slowStep",
  "maSource",
  "objective",
  "mode",
  "topK",
  "includeAll",
  "exportAllCsvPath",
  "heatmapBins",
  "commission",
  "epsRel",
  "longOnly",
  "allowFlip",
  "tradeOnNext",
  "filterMinPnl",
  "filterMinSharpe",
  "filterMaxDd",
  "filterMinTrades",
  "filterMinExposure",
  "paretoX",
  "paretoY",
]) {
  const el = $(id);
  if (!el) continue;
  el.addEventListener("change", saveSettingsSoon);
  el.addEventListener("input", saveSettingsSoon);
}

for (const id of [
  "filterMinPnl",
  "filterMinSharpe",
  "filterMaxDd",
  "filterMinTrades",
  "filterMinExposure",
  "paretoX",
  "paretoY",
]) {
  const el = $(id);
  if (!el) continue;
  const rerender = () => {
    if (CURRENT_RUN) renderTopTable(CURRENT_RUN.res, CURRENT_RUN.req);
  };
  el.addEventListener("change", rerender);
  el.addEventListener("input", rerender);
}

$("btnClearTopFilters")?.addEventListener("click", () => {
  for (const id of ["filterMinPnl", "filterMinSharpe", "filterMaxDd", "filterMinTrades", "filterMinExposure"]) {
    const el = $(id);
    if (el) el.value = "";
  }
  saveSettingsSoon();
  if (CURRENT_RUN) renderTopTable(CURRENT_RUN.res, CURRENT_RUN.req);
});

for (const th of document.querySelectorAll("#topTable th[data-key]")) {
  th.addEventListener("click", () => {
    const key = th.getAttribute("data-key") || "rank";
    if (TOP_SORT.key === key) TOP_SORT.dir = TOP_SORT.dir === "asc" ? "desc" : "asc";
    else {
      TOP_SORT.key = key;
      TOP_SORT.dir = key === "rank" ? "asc" : "desc";
    }
    if (CURRENT_RUN) renderRun(CURRENT_RUN.res, CURRENT_RUN.req);
  });
}

$("btnExportJson")?.addEventListener("click", () => {
  if (!CURRENT_RUN?.res) {
    setStatus("No result to export yet.", "danger");
    return;
  }
  downloadText(`vectorbt_run_${CURRENT_RUN.ts}.json`, pretty(CURRENT_RUN.res), "application/json");
});

$("btnExportCsv")?.addEventListener("click", () => {
  if (!CURRENT_RUN?.res) {
    setStatus("No result to export yet.", "danger");
    return;
  }

  const obj = CURRENT_RUN.req?.objective || "Sharpe";
  const baseRows = Array.isArray(TOP_ROWS_FILTERED) ? TOP_ROWS_FILTERED : TOP_ROWS;
  const sorted = sortRows(baseRows, TOP_SORT.key, TOP_SORT.dir, obj);
  const lines = ["rank,fast_ma,fast_len,slow_ma,slow_len,pnl,sharpe,max_dd,trades,exposure,net_exposure,score"];
  for (const row of sorted) {
    const p = row.params || {};
    const m = row.metrics || {};
    const score = scoreOf(m, obj);
    lines.push([
      row.rank,
      p.fast_ma_type || "",
      p.fast_len || 0,
      p.slow_ma_type || "",
      p.slow_len || 0,
      Number(m.pnl || 0),
      Number(m.sharpe || 0),
      Number(m.max_dd || 0),
      Number(m.trades || 0),
      Number(m.exposure || 0),
      Number(m.net_exposure || 0),
      Number(score || 0),
    ].join(","));
  }
  downloadText(`vectorbt_top_${CURRENT_RUN.ts}.csv`, lines.join("\n"), "text/csv");
});

$("btnClearHistory")?.addEventListener("click", () => {
  setHistory([]);
  populateHistorySelect([], "");
  populateCompareSelect([], "");
  CURRENT_RUN = null;
  DRILL_SELECTED_KEY = "";
  DRILL_SEQ++;
  setDrillVisible(false);
  renderCompare();
  setStatus("History cleared.", "muted");
});

$("btnApplyHistory")?.addEventListener("click", () => {
  if (!CURRENT_RUN?.req) {
    setStatus("Select a run from history first.", "muted");
    return;
  }
  applyRequestToUi(CURRENT_RUN.req);
  setStatus("Applied history run settings to the form.", "ok");
});

$("btnExportPreset")?.addEventListener("click", () => {
  const preset = readSettingsFromUi();
  const ts = new Date().toISOString().replaceAll(":", "-");
  const payload = { version: 1, kind: "vectorbt_preset", ts, preset };
  downloadText(`vectorbt_preset_${ts}.json`, JSON.stringify(payload, null, 2), "application/json");
  setStatus("Exported preset.", "ok");
});

$("btnImportPreset")?.addEventListener("click", () => {
  $("presetFile")?.click();
});

$("presetFile")?.addEventListener("change", async () => {
  const file = $("presetFile")?.files?.[0];
  if (!file) return;
  try {
    const text = await file.text();
    const obj = JSON.parse(text);
    const preset = obj?.preset && typeof obj.preset === "object" ? obj.preset : obj;
    applySettingsObject(preset);
    setStatus("Imported preset.", "ok");
  } catch (e) {
    setStatus(`Failed to import preset: ${String(e)}`, "danger");
  } finally {
    try { $("presetFile").value = ""; } catch {}
  }
});

$("historySelect")?.addEventListener("change", () => {
  const ts = $("historySelect").value || "";
  const history = getHistory();
  const entry = history.find((h) => h.ts === ts);
  if (!entry) return;
  CURRENT_RUN = { ts: entry.ts, req: entry.req, res: entry.res };
  $("output").textContent = pretty(entry.res);
  renderRun(entry.res, entry.req);
  setStatus(`Loaded run ${entry.ts}`, "muted");
});

$("compareSelect")?.addEventListener("change", () => {
  renderCompare();
  saveSettingsSoon();
});

$("btnSetBaseline")?.addEventListener("click", () => {
  if (!CURRENT_RUN?.ts) {
    setStatus("Run an optimization or select a run from history first.", "muted");
    return;
  }
  const sel = $("compareSelect");
  if (!sel) return;
  sel.value = CURRENT_RUN.ts;
  saveSettingsSoon();
  renderCompare();
});

$("btnClearBaseline")?.addEventListener("click", () => {
  const sel = $("compareSelect");
  if (!sel) return;
  sel.value = "";
  saveSettingsSoon();
  renderCompare();
});

loadSettings();
renderRecentCsvDatalist(getRecentCsvPaths());
renderDataMeta(null);
applyMaSourceConstraints();
const hist0 = getHistory();
populateHistorySelect(hist0, "");
populateCompareSelect(hist0, $("compareSelect")?.value || "");
updateDeviceIdEnabled();

initMaSelects();
initProgressListener();
initSampleCsvs();
