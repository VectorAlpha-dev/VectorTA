#![cfg(all(feature = "cuda", feature = "cuda-backtest-kernel"))]

use cust::memory::{mem_get_info, DeviceBuffer};
use cust::module::{ModuleJitOption, OptLevel};
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use cust::context::CurrentContext;
use my_project::utilities::data_loader::{source_type, Candles};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use ta_optimizer::stream::StreamAggregator;
use ta_optimizer::{ObjectiveKind, OptimizationHeatmap, OptimizationResult};
use ta_strategies::double_ma::{DoubleMaParams, Metrics, StrategyConfig};

use crate::progress::ProgressSink;
use crate::vram_ma::{supports_vram_kernel_ma, VramMaComputer, VramMaInputs};

const STRAT_LONG_ONLY: u32 = 1u32 << 0;
#[allow(dead_code)]
const STRAT_NO_FLIP: u32 = 1u32 << 1;
const STRAT_TRADE_ON_NEXT_BAR: u32 = 1u32 << 2;
const STRAT_ENFORCE_FAST_LT_SLOW: u32 = 1u32 << 3;
#[allow(dead_code)]
const STRAT_SIGNED_EXPOSURE: u32 = 1u32 << 4;

const METRICS_COUNT: usize = 7;
const TOPK_BLOCK_ITEMS: usize = 512;

#[derive(Default)]
struct KernelScratch {
    d_lr: Option<DeviceBuffer<f32>>,
    d_fast_p: Option<DeviceBuffer<i32>>,
    d_fast_ma: Option<DeviceBuffer<f32>>,
    d_fast_ma_tm: Option<DeviceBuffer<f32>>,
    d_slow_p: Option<DeviceBuffer<i32>>,
    d_slow_ma: Option<DeviceBuffer<f32>>,
    d_slow_ma_tm: Option<DeviceBuffer<f32>>,
    d_tile: Option<DeviceBuffer<f32>>,

    d_heatmap: Option<DeviceBuffer<f32>>,
    d_cand_a: Option<DeviceBuffer<i32>>,
    d_cand_b: Option<DeviceBuffer<i32>>,
    d_top_fast: Option<DeviceBuffer<i32>>,
    d_top_slow: Option<DeviceBuffer<i32>>,
    d_top_metrics: Option<DeviceBuffer<f32>>,
}

impl KernelScratch {
    fn ensure_len_f32(buf: &mut Option<DeviceBuffer<f32>>, len: usize) -> Result<(), String> {
        if buf.as_ref().map(|b| b.len()) != Some(len) {
            *buf = Some(unsafe { DeviceBuffer::<f32>::uninitialized(len) }.map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn ensure_len_i32(buf: &mut Option<DeviceBuffer<i32>>, len: usize) -> Result<(), String> {
        if buf.as_ref().map(|b| b.len()) != Some(len) {
            *buf = Some(unsafe { DeviceBuffer::<i32>::uninitialized(len) }.map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn clear(&mut self) {
        self.d_lr = None;
        self.d_fast_p = None;
        self.d_fast_ma = None;
        self.d_fast_ma_tm = None;
        self.d_slow_p = None;
        self.d_slow_ma = None;
        self.d_slow_ma_tm = None;
        self.d_tile = None;

        self.d_heatmap = None;
        self.d_cand_a = None;
        self.d_cand_b = None;
        self.d_top_fast = None;
        self.d_top_slow = None;
        self.d_top_metrics = None;
    }
}

struct CudaKernelRuntime {
    context: std::sync::Arc<Context>,
    module: Module,
    stream: Stream,
    ma: VramMaComputer,
    scratch: KernelScratch,
}

impl CudaKernelRuntime {
    fn new(device_id: u32) -> Result<Self, String> {
        cust::init(CudaFlags::empty()).map_err(|e| e.to_string())?;
        let device = Device::get_device(device_id).map_err(|e| e.to_string())?;
        let context = std::sync::Arc::new(Context::new(device).map_err(|e| e.to_string())?);

        let ptx: &str = include_str!(concat!(env!("OUT_DIR"), "/double_crossover.ptx"));
        let jit_opts = &[
            ModuleJitOption::DetermineTargetFromContext,
            ModuleJitOption::OptLevel(OptLevel::O2),
        ];
        let module = match Module::from_ptx(ptx, jit_opts) {
            Ok(m) => m,
            Err(_) => {
                if let Ok(m) = Module::from_ptx(ptx, &[ModuleJitOption::DetermineTargetFromContext])
                {
                    m
                } else {
                    Module::from_ptx(ptx, &[]).map_err(|e| e.to_string())?
                }
            }
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).map_err(|e| e.to_string())?;

        Ok(Self {
            context,
            module,
            stream,
            ma: VramMaComputer::new(device_id),
            scratch: KernelScratch::default(),
        })
    }

    fn set_current(&self) -> Result<(), String> {
        CurrentContext::set_current(self.context.as_ref()).map_err(|e| e.to_string())
    }

}

thread_local! {
    static CUDA_KERNEL_RUNTIME: RefCell<HashMap<u32, CudaKernelRuntime>> = RefCell::new(HashMap::new());
}

fn emit_progress(
    progress: Option<&dyn ProgressSink>,
    processed_pairs: usize,
    total_pairs: usize,
    phase: &'static str,
) {
    crate::progress::emit_double_ma_progress(progress, processed_pairs, total_pairs, phase);
}

pub struct KernelConfig {
    pub device_id: u32,
    pub fast_alma_offset: f64,
    pub fast_alma_sigma: f64,
    pub slow_alma_offset: f64,
    pub slow_alma_sigma: f64,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            fast_alma_offset: 0.85,
            fast_alma_sigma: 6.0,
            slow_alma_offset: 0.85,
            slow_alma_sigma: 6.0,
        }
    }
}

fn expand_u16_range((start, end, step): (u32, u32, u32)) -> Result<Vec<u16>, String> {
    let push_checked = |out: &mut Vec<u16>, v: u32| -> Result<(), String> {
        if v > u16::MAX as u32 {
            return Err(format!("period {v} exceeds u16::MAX"));
        }
        out.push(v as u16);
        Ok(())
    };

    if step == 0 || start == end {
        let mut out = Vec::with_capacity(1);
        push_checked(&mut out, start)?;
        return Ok(out);
    }

    let step = step.max(1);
    let mut out = Vec::new();
    if start <= end {
        let mut v = start;
        loop {
            push_checked(&mut out, v)?;
            if v == end {
                break;
            }
            match v.checked_add(step) {
                Some(next) if next > v && next <= end => v = next,
                _ => break,
            }
        }
    } else {
        let mut v = start;
        loop {
            push_checked(&mut out, v)?;
            if v == end {
                break;
            }
            let next = v.saturating_sub(step);
            if next == v || next < end {
                break;
            }
            v = next;
        }
    }
    if out.is_empty() {
        return Err("empty period range".to_string());
    }
    Ok(out)
}

#[inline]
fn first_finite_idx(series: &[f64]) -> Option<usize> {
    series.iter().position(|v| v.is_finite())
}

#[inline]
fn ma_uses_source(ma: &str) -> bool {
    ma != "frama"
}

#[inline]
fn ma_requires_volume(ma: &str) -> bool {
    ma == "vwma"
}

#[inline]
fn ma_requires_high_low_close(ma: &str) -> bool {
    ma == "frama"
}

#[inline]
fn validate_candles_for_selected_mas(
    candles: &Candles,
    needs_ma_prices: bool,
    ma_source: &str,
    needs_volume: bool,
    needs_high_low: bool,
) -> Result<(), String> {
    if !candles.fields.close {
        return Err("CSV is missing required 'close' data".to_string());
    }
    if needs_volume && !candles.fields.volume {
        return Err("Selected MA requires 'volume' but CSV has no volume column".to_string());
    }
    if needs_high_low && (!candles.fields.high || !candles.fields.low) {
        return Err("Selected MA requires 'high'/'low' but CSV is missing those columns".to_string());
    }

    if needs_ma_prices {
        let src = ma_source.trim().to_ascii_lowercase();
        match src.as_str() {
            "close" => {}
            "open" => {
                if !candles.fields.open {
                    return Err("ma_source=open requires an 'open' column in the CSV".to_string());
                }
            }
            "high" => {
                if !candles.fields.high {
                    return Err("ma_source=high requires a 'high' column in the CSV".to_string());
                }
            }
            "low" => {
                if !candles.fields.low {
                    return Err("ma_source=low requires a 'low' column in the CSV".to_string());
                }
            }
            "hl2" => {
                if !candles.fields.high || !candles.fields.low {
                    return Err("ma_source=hl2 requires 'high'/'low' columns in the CSV".to_string());
                }
            }
            "hlc3" => {
                if !candles.fields.high || !candles.fields.low || !candles.fields.close {
                    return Err("ma_source=hlc3 requires 'high'/'low'/'close' columns in the CSV".to_string());
                }
            }
            "hlcc" | "hlcc4" => {
                if !candles.fields.high || !candles.fields.low || !candles.fields.close {
                    return Err("ma_source=hlcc requires 'high'/'low'/'close' columns in the CSV".to_string());
                }
            }
            "ohlc4" => {
                if !candles.fields.open || !candles.fields.high || !candles.fields.low || !candles.fields.close {
                    return Err("ma_source=ohlc4 requires 'open'/'high'/'low'/'close' columns in the CSV".to_string());
                }
            }
            _ => {}
        }
    }

    Ok(())
}

fn first_valid_idx_required(
    close: &[f64],
    ma_prices: Option<&[f64]>,
    volume: Option<&[f64]>,
    high: Option<&[f64]>,
    low: Option<&[f64]>,
    start_at: usize,
) -> Option<usize> {
    let n = close.len();
    let start_at = start_at.min(n);
    for i in start_at..n {
        if !close[i].is_finite() {
            continue;
        }
        if let Some(p) = ma_prices {
            if !p[i].is_finite() {
                continue;
            }
        }
        if let Some(v) = volume {
            if !v[i].is_finite() {
                continue;
            }
        }
        if let Some(h) = high {
            if !h[i].is_finite() {
                continue;
            }
        }
        if let Some(l) = low {
            if !l[i].is_finite() {
                continue;
            }
        }
        return Some(i);
    }
    None
}

fn vram_budget_bytes() -> usize {
    let headroom = 512usize * 1024 * 1024;
    let min_budget = 64usize * 1024 * 1024;

    match std::env::var("VECTORBT_KERNEL_VRAM_BUDGET_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(mb) => {
            let mut bytes = mb.saturating_mul(1024).saturating_mul(1024);
            if bytes > headroom {
                bytes = bytes.saturating_sub(headroom);
            } else {
                bytes = bytes / 2;
            }
            bytes.max(min_budget)
        }
        None => mem_get_info()
            .ok()
            .map(|(free, _total)| {
                let usable = free.saturating_sub(headroom);
                if usable == 0 {
                    let half = free.saturating_mul(50).saturating_div(100);
                    return half.max(min_budget.min(free));
                }
                let pct = 85usize;
                let budget = usable.saturating_mul(pct).saturating_div(100);
                let floor = (256usize * 1024 * 1024).min(usable);
                budget.max(floor).max(min_budget.min(usable)).min(usable)
            })
            .unwrap_or(256usize * 1024 * 1024),
    }
}

#[inline]
fn kernel_persist_buffers_enabled() -> bool {
    match std::env::var("VECTORBT_KERNEL_PERSIST_BUFFERS") {
        Ok(v) => {
            let v = v.trim();
            !(v == "0" || v.eq_ignore_ascii_case("false"))
        }
        Err(_) => true,
    }
}

#[inline]
fn kernel_fast_path_enabled(fast_ma: &str, slow_ma: &str) -> bool {
    if !matches!(fast_ma, "sma" | "alma") || !matches!(slow_ma, "sma" | "alma") {
        return false;
    }
    match std::env::var("VECTORBT_KERNEL_FAST_PATH") {
        Ok(v) => {
            let v = v.trim();
            !(v == "0" || v.eq_ignore_ascii_case("false"))
        }
        Err(_) => true,
    }
}

fn estimate_tile_bytes(
    pf: usize,
    ps: usize,
    series_len: usize,
    max_fast_period: usize,
    max_slow_period: usize,
    fast_ma: &str,
    slow_ma: &str,
    ma_is_close: bool,
    metrics_count: usize,
    topk_select: usize,
    include_all: bool,
    heatmap_bins: usize,
) -> usize {
    let bytes_f32 = |n: usize| n.saturating_mul(std::mem::size_of::<f32>());
    let bytes_f64 = |n: usize| n.saturating_mul(std::mem::size_of::<f64>());
    let bytes_i32 = |n: usize| n.saturating_mul(std::mem::size_of::<i32>());

    let mut bytes = 0usize;

    // Input series and shared scratch.
    bytes = bytes.saturating_add(bytes_f32(series_len)); // d_close
    if !ma_is_close {
        bytes = bytes.saturating_add(bytes_f32(series_len)); // d_ma_buf
    }
    bytes = bytes.saturating_add(bytes_f32(series_len)); // d_lr

    if fast_ma == "vwma" || slow_ma == "vwma" {
        bytes = bytes.saturating_add(bytes_f32(series_len)); // d_volume
        bytes = bytes.saturating_add(bytes_f64(series_len).saturating_mul(2)); // pv_prefix + vol_prefix
    }
    if fast_ma == "frama" || slow_ma == "frama" {
        bytes = bytes.saturating_add(bytes_f32(series_len).saturating_mul(2)); // d_high + d_low
    }

    // SMA prefix (shared across sides).
    if fast_ma == "sma" || slow_ma == "sma" {
        bytes = bytes.saturating_add(bytes_f64(series_len.saturating_add(1)));
    }

    // Shared scratch buffers (can be reused across fast/slow computation).
    if fast_ma == "nma" || slow_ma == "nma" {
        bytes = bytes.saturating_add(bytes_f32(series_len)); // abs_diffs
    }
    if fast_ma == "edcf" || slow_ma == "edcf" {
        bytes = bytes.saturating_add(bytes_f32(series_len)); // dist scratch
    }
    if fast_ma == "cora_wave" || slow_ma == "cora_wave" {
        let p_max = pf.max(ps).max(1);
        bytes = bytes.saturating_add(bytes_f32(p_max.saturating_mul(series_len))); // y tmp (for smoothing)
    }

    let ma_matrix_mult = if kernel_fast_path_enabled(fast_ma, slow_ma) { 1 } else { 2 };

    // Fast MA buffers.
    bytes = bytes.saturating_add(bytes_i32(pf)); // periods
    bytes = bytes.saturating_add(bytes_f32(
        pf.saturating_mul(series_len).saturating_mul(ma_matrix_mult),
    )); // ma(+ma_tm) or ma_tm-only
    if fast_ma == "alma" || fast_ma == "cwma" || fast_ma == "cora_wave" {
        bytes = bytes.saturating_add(bytes_f32(pf.saturating_mul(max_fast_period))); // weights
        bytes = bytes.saturating_add(bytes_f32(pf)); // inv norms
    } else if fast_ma == "fwma" || fast_ma == "pwma" {
        bytes = bytes.saturating_add(bytes_f32(pf.saturating_mul(max_fast_period))); // weights
    } else if fast_ma == "srwma" {
        bytes = bytes.saturating_add(bytes_f32(
            pf.saturating_mul(max_fast_period.saturating_sub(1)),
        )); // weights
        bytes = bytes.saturating_add(bytes_f32(pf)); // inv norms
    } else if fast_ma == "ema" || fast_ma == "mwdx" {
        bytes = bytes.saturating_add(bytes_f32(pf)); // alphas/factors
    } else if fast_ma == "zlema" {
        bytes = bytes.saturating_add(bytes_i32(pf)); // lags
        bytes = bytes.saturating_add(bytes_f32(pf)); // alphas
    } else if fast_ma == "maaq" {
        bytes = bytes.saturating_add(bytes_f32(pf.saturating_mul(2))); // fast_sc + slow_sc
    } else if fast_ma == "jma" {
        bytes = bytes.saturating_add(bytes_f32(pf.saturating_mul(3))); // alpha + (1-beta) + phase_ratio
    } else if fast_ma == "hma" {
        let max_sqrt = ((max_fast_period as f64).sqrt().floor() as usize).max(1);
        bytes = bytes.saturating_add(bytes_f32(pf.saturating_mul(max_sqrt))); // ring scratch
    } else if fast_ma == "vpwma" {
        bytes = bytes.saturating_add(bytes_i32(pf)); // win lengths
        bytes = bytes.saturating_add(bytes_f32(pf.saturating_mul(max_fast_period.saturating_sub(1)))); // weights
        bytes = bytes.saturating_add(bytes_f32(pf)); // inv norms
    }
    if matches!(fast_ma, "jsa" | "smma" | "swma" | "trima" | "fwma" | "pwma" | "srwma") {
        bytes = bytes.saturating_add(bytes_i32(pf)); // warm indices
    } else if fast_ma == "wilders" {
        bytes = bytes.saturating_add(bytes_i32(pf)); // warm indices
        bytes = bytes.saturating_add(bytes_f32(pf)); // alphas
    } else if fast_ma == "epma" {
        bytes = bytes.saturating_add(bytes_i32(pf)); // offsets
    } else if fast_ma == "cora_wave" {
        bytes = bytes.saturating_add(bytes_i32(pf.saturating_mul(2))); // smooth_periods + warm0s
    } else if fast_ma == "ehlers_itrend" {
        bytes = bytes.saturating_add(bytes_i32(pf)); // warmups
    }
    if fast_ma == "frama" {
        bytes = bytes.saturating_add(bytes_i32(pf).saturating_mul(2)); // scs + fcs
    }

    // Slow MA buffers.
    bytes = bytes.saturating_add(bytes_i32(ps)); // periods
    bytes = bytes.saturating_add(bytes_f32(
        ps.saturating_mul(series_len).saturating_mul(ma_matrix_mult),
    )); // ma(+ma_tm) or ma_tm-only
    if slow_ma == "alma" || slow_ma == "cwma" || slow_ma == "cora_wave" {
        bytes = bytes.saturating_add(bytes_f32(ps.saturating_mul(max_slow_period))); // weights
        bytes = bytes.saturating_add(bytes_f32(ps)); // inv norms
    } else if slow_ma == "fwma" || slow_ma == "pwma" {
        bytes = bytes.saturating_add(bytes_f32(ps.saturating_mul(max_slow_period))); // weights
    } else if slow_ma == "srwma" {
        bytes = bytes.saturating_add(bytes_f32(
            ps.saturating_mul(max_slow_period.saturating_sub(1)),
        )); // weights
        bytes = bytes.saturating_add(bytes_f32(ps)); // inv norms
    } else if slow_ma == "ema" || slow_ma == "mwdx" {
        bytes = bytes.saturating_add(bytes_f32(ps)); // alphas/factors
    } else if slow_ma == "zlema" {
        bytes = bytes.saturating_add(bytes_i32(ps)); // lags
        bytes = bytes.saturating_add(bytes_f32(ps)); // alphas
    } else if slow_ma == "maaq" {
        bytes = bytes.saturating_add(bytes_f32(ps.saturating_mul(2))); // fast_sc + slow_sc
    } else if slow_ma == "jma" {
        bytes = bytes.saturating_add(bytes_f32(ps.saturating_mul(3))); // alpha + (1-beta) + phase_ratio
    } else if slow_ma == "hma" {
        let max_sqrt = ((max_slow_period as f64).sqrt().floor() as usize).max(1);
        bytes = bytes.saturating_add(bytes_f32(ps.saturating_mul(max_sqrt))); // ring scratch
    } else if slow_ma == "vpwma" {
        bytes = bytes.saturating_add(bytes_i32(ps)); // win lengths
        bytes = bytes.saturating_add(bytes_f32(ps.saturating_mul(max_slow_period.saturating_sub(1)))); // weights
        bytes = bytes.saturating_add(bytes_f32(ps)); // inv norms
    }
    if matches!(slow_ma, "jsa" | "smma" | "swma" | "trima" | "fwma" | "pwma" | "srwma") {
        bytes = bytes.saturating_add(bytes_i32(ps)); // warm indices
    } else if slow_ma == "wilders" {
        bytes = bytes.saturating_add(bytes_i32(ps)); // warm indices
        bytes = bytes.saturating_add(bytes_f32(ps)); // alphas
    } else if slow_ma == "epma" {
        bytes = bytes.saturating_add(bytes_i32(ps)); // offsets
    } else if slow_ma == "cora_wave" {
        bytes = bytes.saturating_add(bytes_i32(ps.saturating_mul(2))); // smooth_periods + warm0s
    } else if slow_ma == "ehlers_itrend" {
        bytes = bytes.saturating_add(bytes_i32(ps)); // warmups
    }
    if slow_ma == "frama" {
        bytes = bytes.saturating_add(bytes_i32(ps).saturating_mul(2)); // scs + fcs
    }

    // Per-pair metrics tile.
    let pairs = pf.saturating_mul(ps);
    bytes = bytes.saturating_add(bytes_f32(pairs.saturating_mul(metrics_count)));

    // Optional reductions (used only when include_all=false).
    if !include_all {
        let topk = topk_select.max(1);
        let blocks = (pairs + TOPK_BLOCK_ITEMS - 1) / TOPK_BLOCK_ITEMS;
        let cand_len = blocks.saturating_mul(topk).max(topk);

        bytes = bytes.saturating_add(bytes_i32(cand_len).saturating_mul(2)); // cand_a + cand_b
        bytes = bytes.saturating_add(bytes_i32(topk).saturating_mul(2)); // top_fast + top_slow
        bytes = bytes.saturating_add(bytes_f32(topk.saturating_mul(metrics_count))); // top_metrics

        if heatmap_bins > 0 {
            let heat_len = heatmap_bins.saturating_mul(heatmap_bins);
            bytes = bytes.saturating_add(bytes_f32(heat_len)); // heatmap
        }
    }

    // Small fudge factor for driver allocations / fragmentation.
    bytes.saturating_add(64usize * 1024 * 1024)
}

fn choose_tiles(
    fast_total: usize,
    slow_total: usize,
    series_len: usize,
    max_fast_period: usize,
    max_slow_period: usize,
    fast_ma: &str,
    slow_ma: &str,
    ma_is_close: bool,
    metrics_count: usize,
    topk_select: usize,
    include_all: bool,
    heatmap_bins: usize,
) -> (usize, usize) {
    let fast_total = fast_total.max(1);
    let slow_total = slow_total.max(1);

    let budget = vram_budget_bytes();

    let full_bytes = estimate_tile_bytes(
        fast_total,
        slow_total,
        series_len,
        max_fast_period,
        max_slow_period,
        fast_ma,
        slow_ma,
        ma_is_close,
        metrics_count,
        topk_select,
        include_all,
        heatmap_bins,
    );
    if full_bytes <= budget {
        if std::env::var("VECTORBT_KERNEL_TILE_DEBUG").ok().as_deref() == Some("1") {
            eprintln!(
                "VECTORBT_KERNEL_TILE_DEBUG: budget={}MiB full={}MiB pf_tile={}/{} ps_tile={}/{} (full fits)",
                budget / (1024 * 1024),
                full_bytes / (1024 * 1024),
                fast_total,
                fast_total,
                slow_total,
                slow_total
            );
        }
        return (fast_total, slow_total);
    }

    let mut best_pf = 1usize;
    let mut best_ps = 1usize;
    let mut best_pairs = 1usize;

    for pf in (1..=fast_total).rev() {
        if pf.saturating_mul(slow_total) <= best_pairs {
            break;
        }

        let mut lo = 1usize;
        let mut hi = slow_total;
        let mut best_here = 1usize;
        while lo <= hi {
            let mid = lo.saturating_add(hi) / 2;
            let bytes = estimate_tile_bytes(
                pf,
                mid,
                series_len,
                max_fast_period,
                max_slow_period,
                fast_ma,
                slow_ma,
                ma_is_close,
                metrics_count,
                topk_select,
                include_all,
                heatmap_bins,
            );
            if bytes <= budget {
                best_here = mid;
                lo = mid.saturating_add(1);
            } else if mid == 0 {
                break;
            } else {
                hi = mid.saturating_sub(1);
            }
        }

        let pairs = pf.saturating_mul(best_here).max(1);
        if pairs > best_pairs {
            best_pairs = pairs;
            best_pf = pf;
            best_ps = best_here;
        }
    }

    if std::env::var("VECTORBT_KERNEL_TILE_DEBUG").ok().as_deref() == Some("1") {
        let chosen_bytes = estimate_tile_bytes(
            best_pf,
            best_ps,
            series_len,
            max_fast_period,
            max_slow_period,
            fast_ma,
            slow_ma,
            ma_is_close,
            metrics_count,
            topk_select,
            include_all,
            heatmap_bins,
        );
        eprintln!(
            "VECTORBT_KERNEL_TILE_DEBUG: budget={}MiB full={}MiB chosen={}MiB pf_tile={}/{} ps_tile={}/{}",
            budget / (1024 * 1024),
            full_bytes / (1024 * 1024),
            chosen_bytes / (1024 * 1024),
            best_pf,
            fast_total,
            best_ps,
            slow_total
        );
    }

    (best_pf.max(1).min(fast_total), best_ps.max(1).min(slow_total))
}

pub fn eval_double_ma_batch_gpu_kernel(
    candles: &Candles,
    combos: &[DoubleMaParams],
    fast_range: (u32, u32, u32),
    slow_range: (u32, u32, u32),
    fast_ma_type: &str,
    slow_ma_type: &str,
    ma_source: &str,
    fast_ma_params: Option<&HashMap<String, f64>>,
    slow_ma_params: Option<&HashMap<String, f64>>,
    cfg: KernelConfig,
    strategy: &StrategyConfig,
) -> Result<Vec<Metrics>, String> {
    let device_id = cfg.device_id;
    CUDA_KERNEL_RUNTIME.with(|cell| -> Result<Vec<Metrics>, String> {
        let mut map = cell.borrow_mut();
        if !map.contains_key(&device_id) {
            let rt = CudaKernelRuntime::new(device_id)?;
            map.insert(device_id, rt);
        }
        let rt = map.get_mut(&device_id).expect("just inserted");
        rt.set_current()?;

        eval_double_ma_batch_gpu_kernel_impl(
            rt,
            candles,
            combos,
            fast_range,
            slow_range,
            fast_ma_type,
            slow_ma_type,
            ma_source,
            fast_ma_params,
            slow_ma_params,
            cfg,
            strategy,
        )
    })
}

fn eval_double_ma_batch_gpu_kernel_impl(
    rt: &mut CudaKernelRuntime,
    candles: &Candles,
    combos: &[DoubleMaParams],
    fast_range: (u32, u32, u32),
    slow_range: (u32, u32, u32),
    fast_ma_type: &str,
    slow_ma_type: &str,
    ma_source: &str,
    fast_ma_params: Option<&HashMap<String, f64>>,
    slow_ma_params: Option<&HashMap<String, f64>>,
    cfg: KernelConfig,
    strategy: &StrategyConfig,
) -> Result<Vec<Metrics>, String> {
    if combos.is_empty() {
        return Err("no parameter combinations".to_string());
    }

    let fast_ma = fast_ma_type.trim().to_ascii_lowercase();
    let slow_ma = slow_ma_type.trim().to_ascii_lowercase();
    let needs_ma_prices = ma_uses_source(fast_ma.as_str()) || ma_uses_source(slow_ma.as_str());
    let needs_volume = ma_requires_volume(fast_ma.as_str()) || ma_requires_volume(slow_ma.as_str());
    let needs_high_low =
        ma_requires_high_low_close(fast_ma.as_str()) || ma_requires_high_low_close(slow_ma.as_str());

    validate_candles_for_selected_mas(candles, needs_ma_prices, ma_source, needs_volume, needs_high_low)?;

    let close = source_type(candles, "close");
    let ma_prices = if needs_ma_prices {
        Some(source_type(candles, ma_source))
    } else {
        None
    };
    let volume = if needs_volume {
        Some(source_type(candles, "volume"))
    } else {
        None
    };
    let high = if needs_high_low {
        Some(source_type(candles, "high"))
    } else {
        None
    };
    let low = if needs_high_low {
        Some(source_type(candles, "low"))
    } else {
        None
    };
    let n = close.len();
    if n < 2 {
        return Err("not enough candles".to_string());
    }
    if ma_prices.map_or(false, |p| p.len() != n) {
        return Err("ma source length mismatch".to_string());
    }
    if volume.map_or(false, |v| v.len() != n) {
        return Err("volume length mismatch".to_string());
    }
    if high.map_or(false, |h| h.len() != n) {
        return Err("high length mismatch".to_string());
    }
    if low.map_or(false, |l| l.len() != n) {
        return Err("low length mismatch".to_string());
    }

    let fv_close =
        first_finite_idx(close).ok_or_else(|| "close series all non-finite".to_string())?;
    let mut start_at = fv_close;
    if let Some(p) = ma_prices {
        let fv_ma =
            first_finite_idx(p).ok_or_else(|| "ma source series all non-finite".to_string())?;
        start_at = start_at.max(fv_ma);
    }
    if let Some(v) = volume {
        let fv_vol =
            first_finite_idx(v).ok_or_else(|| "volume series all non-finite".to_string())?;
        start_at = start_at.max(fv_vol);
    }
    if let Some(h) = high {
        let fv_high =
            first_finite_idx(h).ok_or_else(|| "high series all non-finite".to_string())?;
        start_at = start_at.max(fv_high);
    }
    if let Some(l) = low {
        let fv_low = first_finite_idx(l).ok_or_else(|| "low series all non-finite".to_string())?;
        start_at = start_at.max(fv_low);
    }
    let first_valid = first_valid_idx_required(close, ma_prices, volume, high, low, start_at)
        .ok_or_else(|| "no index where all required series are finite".to_string())?;
    if !supports_vram_kernel_ma(fast_ma.as_str()) || !supports_vram_kernel_ma(slow_ma.as_str()) {
        return Err(format!(
             "GPU kernel backend does not support the selected MA(s) for VRAM-resident mode (fast='{fast_ma_type}', slow='{slow_ma_type}')"
        ));
    }

    let fast_periods = expand_u16_range(fast_range)?;
    let slow_periods = expand_u16_range(slow_range)?;
    let f_total = fast_periods.len();
    let s_total = slow_periods.len();
    let max_pf = *fast_periods.iter().max().unwrap_or(&1) as usize;
    let max_ps = *slow_periods.iter().max().unwrap_or(&1) as usize;

    if fast_ma == "wma" && fast_periods.iter().any(|&p| p <= 1) {
        return Err("WMA periods must be > 1 for the GPU kernel backend".to_string());
    }
    if slow_ma == "wma" && slow_periods.iter().any(|&p| p <= 1) {
        return Err("WMA periods must be > 1 for the GPU kernel backend".to_string());
    }
    if fast_ma == "vpwma" && fast_periods.iter().any(|&p| p <= 1) {
        return Err("VPWMA periods must be >= 2 for the GPU kernel backend".to_string());
    }
    if slow_ma == "vpwma" && slow_periods.iter().any(|&p| p <= 1) {
        return Err("VPWMA periods must be >= 2 for the GPU kernel backend".to_string());
    }
    if fast_ma == "frama"
        && fast_periods.iter().any(|&p| {
            let win = p as usize;
            let even = if win & 1 == 1 { win + 1 } else { win };
            even > 1024
        })
    {
        return Err("FRAMA evenized window exceeds CUDA limit (1024)".to_string());
    }
    if slow_ma == "frama"
        && slow_periods.iter().any(|&p| {
            let win = p as usize;
            let even = if win & 1 == 1 { win + 1 } else { win };
            even > 1024
        })
    {
        return Err("FRAMA evenized window exceeds CUDA limit (1024)".to_string());
    }
    if fast_ma == "vpwma" && fast_periods.iter().any(|&p| p <= 1) {
        return Err("VPWMA periods must be >= 2 for the GPU kernel backend".to_string());
    }
    if slow_ma == "vpwma" && slow_periods.iter().any(|&p| p <= 1) {
        return Err("VPWMA periods must be >= 2 for the GPU kernel backend".to_string());
    }
    if fast_ma == "frama"
        && fast_periods.iter().any(|&p| {
            let win = p as usize;
            let even = if win & 1 == 1 { win + 1 } else { win };
            even > 1024
        })
    {
        return Err("FRAMA evenized window exceeds CUDA limit (1024)".to_string());
    }
    if slow_ma == "frama"
        && slow_periods.iter().any(|&p| {
            let win = p as usize;
            let even = if win & 1 == 1 { win + 1 } else { win };
            even > 1024
        })
    {
        return Err("FRAMA evenized window exceeds CUDA limit (1024)".to_string());
    }

    if n - first_valid < max_pf.max(max_ps) {
        return Err(format!(
            "not enough valid data after first_valid={} for max period {} (len={})",
            first_valid,
            max_pf.max(max_ps),
            n
        ));
    }

    let mut fast_idx: HashMap<u16, usize> = HashMap::with_capacity(f_total);
    for (i, &p) in fast_periods.iter().enumerate() {
        fast_idx.insert(p, i);
    }
    let mut slow_idx: HashMap<u16, usize> = HashMap::with_capacity(s_total);
    for (i, &p) in slow_periods.iter().enumerate() {
        slow_idx.insert(p, i);
    }

    let pair_len = f_total
        .checked_mul(s_total)
        .ok_or_else(|| "fast_total*slow_total overflow".to_string())?;
    let mut pair_to_combo: Vec<Option<usize>> = vec![None; pair_len];
    for (ci, p) in combos.iter().enumerate() {
        let fi = *fast_idx
            .get(&p.fast_len)
            .ok_or_else(|| format!("fast_len {} not in requested fast range", p.fast_len))?;
        let si = *slow_idx
            .get(&p.slow_len)
            .ok_or_else(|| format!("slow_len {} not in requested slow range", p.slow_len))?;
        pair_to_combo[fi * s_total + si] = Some(ci);
    }

    let mut out_metrics = vec![
        Metrics {
            pnl: 0.0,
            sharpe: 0.0,
            max_dd: 0.0,
            trades: 0,
            exposure: 0.0,
            net_exposure: 0.0,
        };
        combos.len()
    ];

    let close_f32: Vec<f32> = close.iter().map(|&x| x as f32).collect();
    let d_close = DeviceBuffer::from_slice(&close_f32).map_err(|e| e.to_string())?;

    let ma_is_close = ma_source.trim().eq_ignore_ascii_case("close");
    let alloc_ma_buf = needs_ma_prices && !ma_is_close;
    let d_ma_buf: Option<DeviceBuffer<f32>> = if alloc_ma_buf {
        let ma_f32: Vec<f32> = ma_prices
            .expect("ma_prices required when alloc_ma_buf=true")
            .iter()
            .map(|&x| x as f32)
            .collect();
        Some(DeviceBuffer::from_slice(&ma_f32).map_err(|e| e.to_string())?)
    } else {
        None
    };
    let d_ma_prices: &DeviceBuffer<f32> = d_ma_buf.as_ref().unwrap_or(&d_close);

    let d_volume: Option<DeviceBuffer<f32>> = if needs_volume {
        let vol_f32: Vec<f32> = volume
            .expect("volume required when needs_volume=true")
            .iter()
            .map(|&x| x as f32)
            .collect();
        Some(DeviceBuffer::from_slice(&vol_f32).map_err(|e| e.to_string())?)
    } else {
        None
    };
    let d_high: Option<DeviceBuffer<f32>> = if needs_high_low {
        let high_f32: Vec<f32> = high
            .expect("high required when needs_high_low=true")
            .iter()
            .map(|&x| x as f32)
            .collect();
        Some(DeviceBuffer::from_slice(&high_f32).map_err(|e| e.to_string())?)
    } else {
        None
    };
    let d_low: Option<DeviceBuffer<f32>> = if needs_high_low {
        let low_f32: Vec<f32> = low
            .expect("low required when needs_high_low=true")
            .iter()
            .map(|&x| x as f32)
            .collect();
        Some(DeviceBuffer::from_slice(&low_f32).map_err(|e| e.to_string())?)
    } else {
        None
    };

    let vram_inputs = VramMaInputs {
        prices: d_ma_prices,
        close: &d_close,
        high: d_high.as_ref(),
        low: d_low.as_ref(),
        volume: d_volume.as_ref(),
    };

    let ma = &mut rt.ma;
    let scratch = &mut rt.scratch;
    let mut fast_params_from_cfg: HashMap<String, f64> = HashMap::new();
    let mut slow_params_from_cfg: HashMap<String, f64> = HashMap::new();
    let fast_params: Option<&HashMap<String, f64>> = if fast_ma == "alma" && fast_ma_params.is_none() {
        fast_params_from_cfg.insert("offset".to_string(), cfg.fast_alma_offset);
        fast_params_from_cfg.insert("sigma".to_string(), cfg.fast_alma_sigma);
        Some(&fast_params_from_cfg)
    } else {
        fast_ma_params
    };
    let slow_params: Option<&HashMap<String, f64>> = if slow_ma == "alma" && slow_ma_params.is_none() {
        slow_params_from_cfg.insert("offset".to_string(), cfg.slow_alma_offset);
        slow_params_from_cfg.insert("sigma".to_string(), cfg.slow_alma_sigma);
        Some(&slow_params_from_cfg)
    } else {
        slow_ma_params
    };
    if fast_ma == "sma" || slow_ma == "sma" {
        ma.ensure_sma_prefix_f64(d_ma_prices, n, first_valid)?;
    }
    if fast_ma == "vwma" || slow_ma == "vwma" {
        ma.ensure_vwma_prefix_pv_vol_f64(
            d_ma_prices,
            d_volume.as_ref().ok_or_else(|| "vwma requires volume".to_string())?,
            n,
            first_valid,
        )?;
    }

    let bt_module = &rt.module;
    let bt_stream = &rt.stream;
    let use_fast_path = kernel_fast_path_enabled(fast_ma.as_str(), slow_ma.as_str());

    let compute_lr = bt_module
        .get_function("compute_log_returns_f32")
        .map_err(|e| e.to_string())?;
    let transpose = bt_module
        .get_function("transpose_row_to_tm")
        .map_err(|e| e.to_string())?;
    let backtest = bt_module
        .get_function("double_cross_backtest_tm_flex_f32")
        .map_err(|e| e.to_string())?;

    KernelScratch::ensure_len_f32(&mut scratch.d_lr, n)?;
    let d_lr = scratch.d_lr.as_ref().unwrap();
    unsafe {
        let mut prices_ptr = d_close.as_device_ptr().as_raw();
        let mut t_i = n as i32;
        let mut out_ptr = d_lr.as_device_ptr().as_raw();
        let block_x: u32 = 256;
        let grid_x: u32 = ((n as u32) + block_x - 1) / block_x;
        let args: &mut [*mut c_void] = &mut [
            &mut prices_ptr as *mut _ as *mut c_void,
            &mut t_i as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
        ];
        bt_stream
            .launch(&compute_lr, (grid_x, 1, 1), (block_x, 1, 1), 0, args)
            .map_err(|e| e.to_string())?;
    }

    let (pf_tile, ps_tile) = choose_tiles(
        f_total,
        s_total,
        n,
        max_pf,
        max_ps,
        fast_ma.as_str(),
        slow_ma.as_str(),
        !alloc_ma_buf,
        METRICS_COUNT,
        0,
        true,
        0,
    );

    let fast_ma_cap = pf_tile
        .checked_mul(n)
        .ok_or_else(|| "pf_tile*n overflow".to_string())?;
    let slow_ma_cap = ps_tile
        .checked_mul(n)
        .ok_or_else(|| "ps_tile*n overflow".to_string())?;
    let tile_out_cap = pf_tile
        .checked_mul(ps_tile)
        .and_then(|v| v.checked_mul(METRICS_COUNT))
        .ok_or_else(|| "pf_tile*ps_tile*metrics overflow".to_string())?;

    KernelScratch::ensure_len_i32(&mut scratch.d_fast_p, pf_tile)?;
    if use_fast_path {
        scratch.d_fast_ma = None;
    } else {
        KernelScratch::ensure_len_f32(&mut scratch.d_fast_ma, fast_ma_cap)?;
    }
    KernelScratch::ensure_len_f32(&mut scratch.d_fast_ma_tm, fast_ma_cap)?;
    KernelScratch::ensure_len_i32(&mut scratch.d_slow_p, ps_tile)?;
    if use_fast_path {
        scratch.d_slow_ma = None;
    } else {
        KernelScratch::ensure_len_f32(&mut scratch.d_slow_ma, slow_ma_cap)?;
    }
    KernelScratch::ensure_len_f32(&mut scratch.d_slow_ma_tm, slow_ma_cap)?;
    KernelScratch::ensure_len_f32(&mut scratch.d_tile, tile_out_cap)?;

    let mut host_p_fast: Vec<i32> = Vec::with_capacity(pf_tile.max(1));
    let mut host_p_slow: Vec<i32> = Vec::with_capacity(ps_tile.max(1));
    let mut tile_host_buf: Vec<f32> = Vec::new();
    tile_host_buf.resize(tile_out_cap, 0.0);
	
	    let mut flags = STRAT_ENFORCE_FAST_LT_SLOW | STRAT_SIGNED_EXPOSURE;
	    if strategy.long_only {
	        flags |= STRAT_LONG_ONLY;
    }
    if !strategy.allow_flip {
        flags |= STRAT_NO_FLIP;
    }
    if strategy.trade_on_next_bar {
        flags |= STRAT_TRADE_ON_NEXT_BAR;
    }
    let commission = strategy.commission as f32;
    let eps_rel = strategy.eps_rel as f32;

    let mut f_start = 0usize;
    while f_start < f_total {
        let pf = pf_tile.min(f_total - f_start);

        host_p_fast.clear();
        host_p_fast.extend(fast_periods[f_start..f_start + pf].iter().map(|&x| x as i32));
        let f_pad = host_p_fast.last().copied().unwrap_or(2);
        host_p_fast.resize(pf_tile, f_pad);
        scratch
            .d_fast_p
            .as_mut()
            .unwrap()
            .copy_from(&host_p_fast)
            .map_err(|e| e.to_string())?;
        if use_fast_path {
            ma.compute_period_ma_tm_into(
                fast_ma.as_str(),
                fast_params,
                &vram_inputs,
                n,
                first_valid,
                &host_p_fast[..pf],
                scratch.d_fast_p.as_ref().unwrap(),
                scratch.d_fast_ma_tm.as_mut().unwrap(),
            )?;
        } else {
            ma.compute_period_ma_into(
                fast_ma.as_str(),
                fast_params,
                &vram_inputs,
                n,
                first_valid,
                &host_p_fast,
                scratch.d_fast_p.as_ref().unwrap(),
                scratch.d_fast_ma.as_mut().unwrap(),
            )?;

            unsafe {
                let mut in_ptr = scratch.d_fast_ma.as_ref().unwrap().as_device_ptr().as_raw();
                let mut rows = pf as i32;
                let mut cols = n as i32;
                let mut out_ptr = scratch.d_fast_ma_tm.as_ref().unwrap().as_device_ptr().as_raw();
                let block_x: u32 = 256;
                let total = (pf * n) as u32;
                let grid_x: u32 = (total + block_x - 1) / block_x;
                let args: &mut [*mut c_void] = &mut [
                    &mut in_ptr as *mut _ as *mut c_void,
                    &mut rows as *mut _ as *mut c_void,
                    &mut cols as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                bt_stream
                    .launch(&transpose, (grid_x, 1, 1), (block_x, 1, 1), 0, args)
                    .map_err(|e| e.to_string())?;
            }
        }

        let mut s_start = 0usize;
        while s_start < s_total {
            let ps = ps_tile.min(s_total - s_start);

            host_p_slow.clear();
            host_p_slow.extend(slow_periods[s_start..s_start + ps].iter().map(|&x| x as i32));
            let s_pad = host_p_slow.last().copied().unwrap_or(2);
            host_p_slow.resize(ps_tile, s_pad);
            scratch
                .d_slow_p
                .as_mut()
                .unwrap()
                .copy_from(&host_p_slow)
                .map_err(|e| e.to_string())?;
            if use_fast_path {
                ma.compute_period_ma_tm_into(
                    slow_ma.as_str(),
                    slow_params,
                    &vram_inputs,
                    n,
                    first_valid,
                    &host_p_slow[..ps],
                    scratch.d_slow_p.as_ref().unwrap(),
                    scratch.d_slow_ma_tm.as_mut().unwrap(),
                )?;
            } else {
                ma.compute_period_ma_into(
                    slow_ma.as_str(),
                    slow_params,
                    &vram_inputs,
                    n,
                    first_valid,
                    &host_p_slow,
                    scratch.d_slow_p.as_ref().unwrap(),
                    scratch.d_slow_ma.as_mut().unwrap(),
                )?;

                unsafe {
                    let mut in_ptr = scratch.d_slow_ma.as_ref().unwrap().as_device_ptr().as_raw();
                    let mut rows = ps as i32;
                    let mut cols = n as i32;
                    let mut out_ptr =
                        scratch.d_slow_ma_tm.as_ref().unwrap().as_device_ptr().as_raw();
                    let block_x: u32 = 256;
                    let total = (ps * n) as u32;
                    let grid_x: u32 = (total + block_x - 1) / block_x;
                    let args: &mut [*mut c_void] = &mut [
                        &mut in_ptr as *mut _ as *mut c_void,
                        &mut rows as *mut _ as *mut c_void,
                        &mut cols as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    bt_stream
                        .launch(&transpose, (grid_x, 1, 1), (block_x, 1, 1), 0, args)
                        .map_err(|e| e.to_string())?;
                }
            }

            let pairs = pf
                .checked_mul(ps)
                .ok_or_else(|| "pf*ps overflow".to_string())?;
            let out_len = pairs
                .checked_mul(METRICS_COUNT)
                .ok_or_else(|| "pairs*metrics overflow".to_string())?;

            if out_len > tile_out_cap {
                return Err(format!(
                    "tile output len {out_len} exceeds allocated capacity {tile_out_cap}"
                ));
            }

            unsafe {
                let mut f_ma_tm = scratch.d_fast_ma_tm.as_ref().unwrap().as_device_ptr().as_raw();
                let mut s_ma_tm = scratch.d_slow_ma_tm.as_ref().unwrap().as_device_ptr().as_raw();
                let mut lr = d_lr.as_device_ptr().as_raw();
                let mut t_i = n as i32;
                let mut pf_tile_i = pf as i32;
                let mut ps_tile_i = ps as i32;
                let mut pf_total_i = pf as i32;
                let mut ps_total_i = ps as i32;
                let mut f_off = 0i32;
                let mut s_off = 0i32;
                let mut f_per = scratch.d_fast_p.as_ref().unwrap().as_device_ptr().as_raw();
                let mut s_per = scratch.d_slow_p.as_ref().unwrap().as_device_ptr().as_raw();
                let mut fv = first_valid as i32;
                let mut commission = commission;
                let mut eps_rel = eps_rel;
                let mut flags_u = flags;
                let mut m_i = METRICS_COUNT as i32;
                let mut out_ptr = scratch.d_tile.as_ref().unwrap().as_device_ptr().as_raw();

                let block_x: u32 = 256;
                let grid_x: u32 = ((pairs as u32) + block_x - 1) / block_x;
                let args: &mut [*mut c_void] = &mut [
                    &mut f_ma_tm as *mut _ as *mut c_void,
                    &mut s_ma_tm as *mut _ as *mut c_void,
                    &mut lr as *mut _ as *mut c_void,
                    &mut t_i as *mut _ as *mut c_void,
                    &mut pf_tile_i as *mut _ as *mut c_void,
                    &mut ps_tile_i as *mut _ as *mut c_void,
                    &mut pf_total_i as *mut _ as *mut c_void,
                    &mut ps_total_i as *mut _ as *mut c_void,
                    &mut f_off as *mut _ as *mut c_void,
                    &mut s_off as *mut _ as *mut c_void,
                    &mut f_per as *mut _ as *mut c_void,
                    &mut s_per as *mut _ as *mut c_void,
                    &mut fv as *mut _ as *mut c_void,
                    &mut commission as *mut _ as *mut c_void,
                    &mut eps_rel as *mut _ as *mut c_void,
                    &mut flags_u as *mut _ as *mut c_void,
                    &mut m_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                bt_stream
                    .launch(&backtest, (grid_x, 1, 1), (block_x, 1, 1), 0, args)
                    .map_err(|e| e.to_string())?;
            }
            bt_stream.synchronize().map_err(|e| e.to_string())?;

            scratch
                .d_tile
                .as_ref()
                .unwrap()
                .copy_to(&mut tile_host_buf)
                .map_err(|e| e.to_string())?;

            for i in 0..pf {
                for j in 0..ps {
                    let global_f = f_start + i;
                    let global_s = s_start + j;
                    if let Some(combo_idx) = pair_to_combo[global_f * s_total + global_s] {
                        let src_base = (i * ps + j) * METRICS_COUNT;
                        let pnl = tile_host_buf[src_base + 0] as f64;
                        let trades = tile_host_buf[src_base + 1];
                        let max_dd = tile_host_buf[src_base + 2] as f64;
                        let mean = tile_host_buf[src_base + 3] as f64;
                        let std = tile_host_buf[src_base + 4] as f64;
                        let exposure = tile_host_buf[src_base + 5] as f64;
                        let net_exposure = tile_host_buf[src_base + 6] as f64;
                        let sharpe = if std > 0.0 { mean / std } else { 0.0 };
                        out_metrics[combo_idx] = Metrics {
                            pnl,
                            sharpe,
                            max_dd,
                            trades: if trades.is_finite() && trades >= 0.0 { trades as u32 } else { 0 },
                            exposure,
                            net_exposure,
                        };
                    }
                }
            }

            s_start += ps;
        }

        f_start += pf;
    }

    if !kernel_persist_buffers_enabled() {
        ma.clear_cached_constants();
        scratch.clear();
    }

    Ok(out_metrics)
}

pub fn optimize_double_ma_gpu_kernel(
    candles: &Candles,
    fast_periods: &[u16],
    slow_periods: &[u16],
    fast_ma_type: &str,
    slow_ma_type: &str,
    ma_source: &str,
    fast_ma_params: Option<&HashMap<String, f64>>,
    slow_ma_params: Option<&HashMap<String, f64>>,
    strategy: &StrategyConfig,
    objective: ObjectiveKind,
    top_k: usize,
    include_all: bool,
    cfg: KernelConfig,
    export_all_csv_path: Option<&str>,
    total_pairs: usize,
    heatmap_bins: usize,
    cancel: &AtomicBool,
    progress: Option<&dyn ProgressSink>,
) -> Result<OptimizationResult, String> {
    let device_id = cfg.device_id;
    CUDA_KERNEL_RUNTIME.with(|cell| -> Result<OptimizationResult, String> {
        let mut map = cell.borrow_mut();
        if !map.contains_key(&device_id) {
            let rt = CudaKernelRuntime::new(device_id)?;
            map.insert(device_id, rt);
        }
        let rt = map.get_mut(&device_id).expect("just inserted");
        rt.set_current()?;

        optimize_double_ma_gpu_kernel_impl(
            rt,
            candles,
            fast_periods,
            slow_periods,
            fast_ma_type,
            slow_ma_type,
            ma_source,
            fast_ma_params,
            slow_ma_params,
            strategy,
            objective,
            top_k,
            include_all,
            cfg,
            export_all_csv_path,
            total_pairs,
            heatmap_bins,
            cancel,
            progress,
        )
    })
}

fn optimize_double_ma_gpu_kernel_impl(
    rt: &mut CudaKernelRuntime,
    candles: &Candles,
    fast_periods: &[u16],
    slow_periods: &[u16],
    fast_ma_type: &str,
    slow_ma_type: &str,
    ma_source: &str,
    fast_ma_params: Option<&HashMap<String, f64>>,
    slow_ma_params: Option<&HashMap<String, f64>>,
    strategy: &StrategyConfig,
    objective: ObjectiveKind,
    top_k: usize,
    include_all: bool,
    cfg: KernelConfig,
    export_all_csv_path: Option<&str>,
    total_pairs: usize,
    heatmap_bins: usize,
    cancel: &AtomicBool,
    progress: Option<&dyn ProgressSink>,
) -> Result<OptimizationResult, String> {
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".to_string());
    }

    let export_path = export_all_csv_path.map(str::trim).filter(|s| !s.is_empty());
    if export_path.is_some() && include_all {
        return Err("export_all_csv_path requires include_all=false (stream export avoids huge RAM usage)".to_string());
    }
    let export_enabled = export_path.is_some();
    let mut export_writer: Option<BufWriter<File>> = if let Some(path) = export_path {
        let f = File::create(path)
            .map_err(|e| format!("failed to create export file '{path}': {e}"))?;
        let mut w = BufWriter::new(f);
        writeln!(
            w,
            "fast_ma,slow_ma,fast_len,slow_len,pnl,sharpe,max_dd,trades,exposure,net_exposure,score"
        )
        .map_err(|e| e.to_string())?;
        Some(w)
    } else {
        None
    };

    if fast_periods.is_empty() || slow_periods.is_empty() {
        return Err("empty fast/slow period list".to_string());
    }

    let fast_ma = fast_ma_type.trim().to_ascii_lowercase();
    let slow_ma = slow_ma_type.trim().to_ascii_lowercase();
    let needs_ma_prices = ma_uses_source(fast_ma.as_str()) || ma_uses_source(slow_ma.as_str());
    let needs_volume = ma_requires_volume(fast_ma.as_str()) || ma_requires_volume(slow_ma.as_str());
    let needs_high_low =
        ma_requires_high_low_close(fast_ma.as_str()) || ma_requires_high_low_close(slow_ma.as_str());

    validate_candles_for_selected_mas(candles, needs_ma_prices, ma_source, needs_volume, needs_high_low)?;

    let close = source_type(candles, "close");
    let ma_prices = if needs_ma_prices {
        Some(source_type(candles, ma_source))
    } else {
        None
    };
    let volume = if needs_volume {
        Some(source_type(candles, "volume"))
    } else {
        None
    };
    let high = if needs_high_low {
        Some(source_type(candles, "high"))
    } else {
        None
    };
    let low = if needs_high_low {
        Some(source_type(candles, "low"))
    } else {
        None
    };
    let n = close.len();
    if n < 2 {
        return Err("not enough candles".to_string());
    }
    if ma_prices.map_or(false, |p| p.len() != n) {
        return Err("ma source length mismatch".to_string());
    }
    if volume.map_or(false, |v| v.len() != n) {
        return Err("volume length mismatch".to_string());
    }
    if high.map_or(false, |h| h.len() != n) {
        return Err("high length mismatch".to_string());
    }
    if low.map_or(false, |l| l.len() != n) {
        return Err("low length mismatch".to_string());
    }

    let fv_close =
        first_finite_idx(close).ok_or_else(|| "close series all non-finite".to_string())?;
    let mut start_at = fv_close;
    if let Some(p) = ma_prices {
        let fv_ma =
            first_finite_idx(p).ok_or_else(|| "ma source series all non-finite".to_string())?;
        start_at = start_at.max(fv_ma);
    }
    if let Some(v) = volume {
        let fv_vol =
            first_finite_idx(v).ok_or_else(|| "volume series all non-finite".to_string())?;
        start_at = start_at.max(fv_vol);
    }
    if let Some(h) = high {
        let fv_high =
            first_finite_idx(h).ok_or_else(|| "high series all non-finite".to_string())?;
        start_at = start_at.max(fv_high);
    }
    if let Some(l) = low {
        let fv_low = first_finite_idx(l).ok_or_else(|| "low series all non-finite".to_string())?;
        start_at = start_at.max(fv_low);
    }
    let first_valid = first_valid_idx_required(close, ma_prices, volume, high, low, start_at)
        .ok_or_else(|| "no index where all required series are finite".to_string())?;
    if !crate::vram_ma::supports_vram_kernel_ma(fast_ma.as_str())
        || !crate::vram_ma::supports_vram_kernel_ma(slow_ma.as_str())
    {
        return Err(format!(
            "GPU kernel backend does not support fast='{fast_ma_type}', slow='{slow_ma_type}'"
        ));
    }

    let f_total = fast_periods.len();
    let s_total = slow_periods.len();
    let max_pf = *fast_periods.iter().max().unwrap_or(&1) as usize;
    let max_ps = *slow_periods.iter().max().unwrap_or(&1) as usize;

    if fast_ma == "wma" && fast_periods.iter().any(|&p| p <= 1) {
        return Err("WMA periods must be > 1 for the GPU kernel backend".to_string());
    }
    if slow_ma == "wma" && slow_periods.iter().any(|&p| p <= 1) {
        return Err("WMA periods must be > 1 for the GPU kernel backend".to_string());
    }

    if n - first_valid < max_pf.max(max_ps) {
        return Err(format!(
            "not enough valid data after first_valid={} for max period {} (len={})",
            first_valid,
            max_pf.max(max_ps),
            n
        ));
    }

    let heatmap_cfg = if heatmap_bins > 0 {
        Some((
            *fast_periods.iter().min().unwrap_or(&0),
            *fast_periods.iter().max().unwrap_or(&0),
            *slow_periods.iter().min().unwrap_or(&0),
            *slow_periods.iter().max().unwrap_or(&0),
        ))
    } else {
        None
    };

    let mut agg = StreamAggregator::new(objective, top_k, include_all, n);
    if include_all {
        if let Some((f_min, f_max, s_min, s_max)) = heatmap_cfg {
            agg = agg.with_heatmap(heatmap_bins, heatmap_bins, f_min, f_max, s_min, s_max);
        }
    }
    let mut processed = 0usize;
    emit_progress(progress, processed, total_pairs, "gpu-kernel");

    let fast_pass = objective != ObjectiveKind::Sharpe && !include_all && !export_enabled;
    let mut metrics_count: usize = if fast_pass { 3 } else { METRICS_COUNT };
    if !include_all && !export_enabled {
        if let Ok(v) = std::env::var("VECTORBT_KERNEL_METRICS_COUNT") {
            let v = v.trim();
            if !v.is_empty() {
                let parsed: usize = v
                    .parse()
                    .map_err(|_| format!("VECTORBT_KERNEL_METRICS_COUNT must be an integer (got '{v}')"))?;
                if parsed == 0 || parsed > METRICS_COUNT {
                    return Err(format!(
                        "VECTORBT_KERNEL_METRICS_COUNT must be in [1, {METRICS_COUNT}] (got {parsed})"
                    ));
                }
                metrics_count = parsed;
            }
        }
    }
    match objective {
        ObjectiveKind::Pnl => {}
        ObjectiveKind::MaxDrawdown => {
            if metrics_count < 3 {
                return Err(format!(
                    "VECTORBT_KERNEL_METRICS_COUNT={} too small for objective MaxDrawdown (need >= 3)",
                    metrics_count
                ));
            }
        }
        ObjectiveKind::Sharpe => {
            if metrics_count < 5 {
                return Err(format!(
                    "VECTORBT_KERNEL_METRICS_COUNT={} too small for objective Sharpe (need >= 5)",
                    metrics_count
                ));
            }
        }
    }
    let topk_select: usize = if top_k == 0 { 1 } else { top_k };
    if !include_all && topk_select > 256 {
        return Err("GPU kernel backend top_k too large (max 256 when include_all=false)".to_string());
    }
    let objective_i: i32 = match objective {
        ObjectiveKind::Pnl => 0,
        ObjectiveKind::Sharpe => 1,
        ObjectiveKind::MaxDrawdown => 2,
    };
    let cfg_recompute = KernelConfig {
        device_id: cfg.device_id,
        fast_alma_offset: cfg.fast_alma_offset,
        fast_alma_sigma: cfg.fast_alma_sigma,
        slow_alma_offset: cfg.slow_alma_offset,
        slow_alma_sigma: cfg.slow_alma_sigma,
    };

    let close_f32: Vec<f32> = close.iter().map(|&x| x as f32).collect();
    let d_close = DeviceBuffer::from_slice(&close_f32).map_err(|e| e.to_string())?;

    let ma_is_close = ma_source.trim().eq_ignore_ascii_case("close");
    let alloc_ma_buf = needs_ma_prices && !ma_is_close;
    let d_ma_buf: Option<DeviceBuffer<f32>> = if alloc_ma_buf {
        let ma_f32: Vec<f32> = ma_prices
            .expect("ma_prices required when alloc_ma_buf=true")
            .iter()
            .map(|&x| x as f32)
            .collect();
        Some(DeviceBuffer::from_slice(&ma_f32).map_err(|e| e.to_string())?)
    } else {
        None
    };
    let d_ma_prices: &DeviceBuffer<f32> = d_ma_buf.as_ref().unwrap_or(&d_close);

    let d_volume: Option<DeviceBuffer<f32>> = if needs_volume {
        let vol_f32: Vec<f32> = volume
            .expect("volume required when needs_volume=true")
            .iter()
            .map(|&x| x as f32)
            .collect();
        Some(DeviceBuffer::from_slice(&vol_f32).map_err(|e| e.to_string())?)
    } else {
        None
    };
    let d_high: Option<DeviceBuffer<f32>> = if needs_high_low {
        let high_f32: Vec<f32> = high
            .expect("high required when needs_high_low=true")
            .iter()
            .map(|&x| x as f32)
            .collect();
        Some(DeviceBuffer::from_slice(&high_f32).map_err(|e| e.to_string())?)
    } else {
        None
    };
    let d_low: Option<DeviceBuffer<f32>> = if needs_high_low {
        let low_f32: Vec<f32> = low
            .expect("low required when needs_high_low=true")
            .iter()
            .map(|&x| x as f32)
            .collect();
        Some(DeviceBuffer::from_slice(&low_f32).map_err(|e| e.to_string())?)
    } else {
        None
    };

    let vram_inputs = VramMaInputs {
        prices: d_ma_prices,
        close: &d_close,
        high: d_high.as_ref(),
        low: d_low.as_ref(),
        volume: d_volume.as_ref(),
    };

    let ma = &mut rt.ma;
    let scratch = &mut rt.scratch;
    let mut fast_params_from_cfg: HashMap<String, f64> = HashMap::new();
    let mut slow_params_from_cfg: HashMap<String, f64> = HashMap::new();
    let fast_params: Option<&HashMap<String, f64>> = if fast_ma == "alma" && fast_ma_params.is_none() {
        fast_params_from_cfg.insert("offset".to_string(), cfg.fast_alma_offset);
        fast_params_from_cfg.insert("sigma".to_string(), cfg.fast_alma_sigma);
        Some(&fast_params_from_cfg)
    } else {
        fast_ma_params
    };
    let slow_params: Option<&HashMap<String, f64>> = if slow_ma == "alma" && slow_ma_params.is_none() {
        slow_params_from_cfg.insert("offset".to_string(), cfg.slow_alma_offset);
        slow_params_from_cfg.insert("sigma".to_string(), cfg.slow_alma_sigma);
        Some(&slow_params_from_cfg)
    } else {
        slow_ma_params
    };
    if fast_ma == "sma" || slow_ma == "sma" {
        ma.ensure_sma_prefix_f64(d_ma_prices, n, first_valid)?;
    }
    if fast_ma == "vwma" || slow_ma == "vwma" {
        ma.ensure_vwma_prefix_pv_vol_f64(
            d_ma_prices,
            d_volume.as_ref().ok_or_else(|| "vwma requires volume".to_string())?,
            n,
            first_valid,
        )?;
    }

    let bt_module = &rt.module;
    let bt_stream = &rt.stream;
    let use_fast_path = kernel_fast_path_enabled(fast_ma.as_str(), slow_ma.as_str());

    let compute_lr = bt_module
        .get_function("compute_log_returns_f32")
        .map_err(|e| e.to_string())?;
    let transpose = bt_module
        .get_function("transpose_row_to_tm")
        .map_err(|e| e.to_string())?;
    let backtest = bt_module
        .get_function("double_cross_backtest_tm_flex_f32")
        .map_err(|e| e.to_string())?;
    let fill_f32 = bt_module
        .get_function("fill_f32")
        .map_err(|e| e.to_string())?;
    let heatmap_update = bt_module
        .get_function("heatmap_update_scores_tm_f32")
        .map_err(|e| e.to_string())?;
    let select_topk_pairs = bt_module
        .get_function("select_topk_from_pairs_tm_f32")
        .map_err(|e| e.to_string())?;
    let select_topk_candidates = bt_module
        .get_function("select_topk_from_candidates_tm_f32")
        .map_err(|e| e.to_string())?;
    let gather_topk = bt_module
        .get_function("gather_topk_pairs_tm_f32")
        .map_err(|e| e.to_string())?;

    KernelScratch::ensure_len_f32(&mut scratch.d_lr, n)?;
    let d_lr = scratch.d_lr.as_ref().unwrap();
    unsafe {
        let mut prices_ptr = d_close.as_device_ptr().as_raw();
        let mut t_i = n as i32;
        let mut out_ptr = d_lr.as_device_ptr().as_raw();
        let block_x: u32 = 256;
        let grid_x: u32 = ((n as u32) + block_x - 1) / block_x;
        let args: &mut [*mut c_void] = &mut [
            &mut prices_ptr as *mut _ as *mut c_void,
            &mut t_i as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
        ];
        bt_stream
            .launch(&compute_lr, (grid_x, 1, 1), (block_x, 1, 1), 0, args)
            .map_err(|e| e.to_string())?;
    }

    let (pf_tile, ps_tile) = choose_tiles(
        f_total,
        s_total,
        n,
        max_pf,
        max_ps,
        fast_ma.as_str(),
        slow_ma.as_str(),
        !alloc_ma_buf,
        metrics_count,
        topk_select,
        include_all,
        heatmap_bins,
    );

    let fast_ma_cap = pf_tile
        .checked_mul(n)
        .ok_or_else(|| "pf_tile*n overflow".to_string())?;
    let slow_ma_cap = ps_tile
        .checked_mul(n)
        .ok_or_else(|| "ps_tile*n overflow".to_string())?;
    let tile_out_cap = pf_tile
        .checked_mul(ps_tile)
        .and_then(|v| v.checked_mul(metrics_count))
        .ok_or_else(|| "pf_tile*ps_tile*metrics overflow".to_string())?;

    KernelScratch::ensure_len_i32(&mut scratch.d_fast_p, pf_tile)?;
    if use_fast_path {
        scratch.d_fast_ma = None;
    } else {
        KernelScratch::ensure_len_f32(&mut scratch.d_fast_ma, fast_ma_cap)?;
    }
    KernelScratch::ensure_len_f32(&mut scratch.d_fast_ma_tm, fast_ma_cap)?;
    KernelScratch::ensure_len_i32(&mut scratch.d_slow_p, ps_tile)?;
    if use_fast_path {
        scratch.d_slow_ma = None;
    } else {
        KernelScratch::ensure_len_f32(&mut scratch.d_slow_ma, slow_ma_cap)?;
    }
    KernelScratch::ensure_len_f32(&mut scratch.d_slow_ma_tm, slow_ma_cap)?;
    KernelScratch::ensure_len_f32(&mut scratch.d_tile, tile_out_cap)?;

    let mut host_p_fast: Vec<i32> = Vec::with_capacity(pf_tile.max(1));
    let mut host_p_slow: Vec<i32> = Vec::with_capacity(ps_tile.max(1));

    let mut tile_host_buf: Vec<f32> = Vec::new();
    if include_all || export_enabled {
        tile_host_buf.resize(tile_out_cap, 0.0);
    }

    let mut top_fast_host: Vec<i32> = Vec::new();
    let mut top_slow_host: Vec<i32> = Vec::new();
    let mut top_metrics_host: Vec<f32> = Vec::new();

    if !include_all {
        let max_pairs_tile = pf_tile.saturating_mul(ps_tile).max(1);
        let blocks_max = (max_pairs_tile + TOPK_BLOCK_ITEMS - 1) / TOPK_BLOCK_ITEMS;
        let cand_len_max = blocks_max.saturating_mul(topk_select).max(topk_select);

        KernelScratch::ensure_len_i32(&mut scratch.d_cand_a, cand_len_max)?;
        KernelScratch::ensure_len_i32(&mut scratch.d_cand_b, cand_len_max)?;
        KernelScratch::ensure_len_i32(&mut scratch.d_top_fast, topk_select)?;
        KernelScratch::ensure_len_i32(&mut scratch.d_top_slow, topk_select)?;
        KernelScratch::ensure_len_f32(&mut scratch.d_top_metrics, topk_select * metrics_count)?;
        top_fast_host.resize(topk_select, 0);
        top_slow_host.resize(topk_select, 0);
        top_metrics_host.resize(topk_select * metrics_count, f32::NAN);

        if heatmap_cfg.is_some() {
            let heatmap_len = heatmap_bins.saturating_mul(heatmap_bins);
            if heatmap_len > 0 {
                KernelScratch::ensure_len_f32(&mut scratch.d_heatmap, heatmap_len)?;
                unsafe {
                    let mut out_ptr = scratch.d_heatmap.as_ref().unwrap().as_device_ptr().as_raw();
                    let mut n_i = heatmap_len as i32;
                    let mut v = f32::NEG_INFINITY;
                    let block_x: u32 = 256;
                    let grid_x: u32 = ((heatmap_len as u32) + block_x - 1) / block_x;
                    let args: &mut [*mut c_void] = &mut [
                        &mut out_ptr as *mut _ as *mut c_void,
                        &mut n_i as *mut _ as *mut c_void,
                        &mut v as *mut _ as *mut c_void,
                    ];
                    bt_stream
                        .launch(&fill_f32, (grid_x, 1, 1), (block_x, 1, 1), 0, args)
                        .map_err(|e| e.to_string())?;
                }
            }
        }
    }

    let mut flags = STRAT_ENFORCE_FAST_LT_SLOW | STRAT_SIGNED_EXPOSURE;
    if strategy.long_only {
        flags |= STRAT_LONG_ONLY;
    }
    if !strategy.allow_flip {
        flags |= STRAT_NO_FLIP;
    }
    if strategy.trade_on_next_bar {
        flags |= STRAT_TRADE_ON_NEXT_BAR;
    }
    let commission = strategy.commission as f32;
    let eps_rel = strategy.eps_rel as f32;

    let mut f_start = 0usize;
    while f_start < f_total {
        let pf = pf_tile.min(f_total - f_start);

        host_p_fast.clear();
        host_p_fast.extend(fast_periods[f_start..f_start + pf].iter().map(|&x| x as i32));
        let f_pad = host_p_fast.last().copied().unwrap_or(2);
        host_p_fast.resize(pf_tile, f_pad);
        scratch
            .d_fast_p
            .as_mut()
            .unwrap()
            .copy_from(&host_p_fast)
            .map_err(|e| e.to_string())?;
        if use_fast_path {
            ma.compute_period_ma_tm_into(
                fast_ma.as_str(),
                fast_params,
                &vram_inputs,
                n,
                first_valid,
                &host_p_fast[..pf],
                scratch.d_fast_p.as_ref().unwrap(),
                scratch.d_fast_ma_tm.as_mut().unwrap(),
            )?;
        } else {
            ma.compute_period_ma_into(
                fast_ma.as_str(),
                fast_params,
                &vram_inputs,
                n,
                first_valid,
                &host_p_fast,
                scratch.d_fast_p.as_ref().unwrap(),
                scratch.d_fast_ma.as_mut().unwrap(),
            )?;

            unsafe {
                let mut in_ptr = scratch.d_fast_ma.as_ref().unwrap().as_device_ptr().as_raw();
                let mut rows = pf as i32;
                let mut cols = n as i32;
                let mut out_ptr = scratch.d_fast_ma_tm.as_ref().unwrap().as_device_ptr().as_raw();
                let block_x: u32 = 256;
                let total = (pf * n) as u32;
                let grid_x: u32 = (total + block_x - 1) / block_x;
                let args: &mut [*mut c_void] = &mut [
                    &mut in_ptr as *mut _ as *mut c_void,
                    &mut rows as *mut _ as *mut c_void,
                    &mut cols as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
                bt_stream
                    .launch(&transpose, (grid_x, 1, 1), (block_x, 1, 1), 0, args)
                    .map_err(|e| e.to_string())?;
            }
        }

        let mut s_start = 0usize;
        while s_start < s_total {
            let ps = ps_tile.min(s_total - s_start);

            host_p_slow.clear();
            host_p_slow.extend(slow_periods[s_start..s_start + ps].iter().map(|&x| x as i32));
            let s_pad = host_p_slow.last().copied().unwrap_or(2);
            host_p_slow.resize(ps_tile, s_pad);
            scratch
                .d_slow_p
                .as_mut()
                .unwrap()
                .copy_from(&host_p_slow)
                .map_err(|e| e.to_string())?;
            if use_fast_path {
                ma.compute_period_ma_tm_into(
                    slow_ma.as_str(),
                    slow_params,
                    &vram_inputs,
                    n,
                    first_valid,
                    &host_p_slow[..ps],
                    scratch.d_slow_p.as_ref().unwrap(),
                    scratch.d_slow_ma_tm.as_mut().unwrap(),
                )?;
            } else {
                ma.compute_period_ma_into(
                    slow_ma.as_str(),
                    slow_params,
                    &vram_inputs,
                    n,
                    first_valid,
                    &host_p_slow,
                    scratch.d_slow_p.as_ref().unwrap(),
                    scratch.d_slow_ma.as_mut().unwrap(),
                )?;

                unsafe {
                    let mut in_ptr = scratch.d_slow_ma.as_ref().unwrap().as_device_ptr().as_raw();
                    let mut rows = ps as i32;
                    let mut cols = n as i32;
                    let mut out_ptr =
                        scratch.d_slow_ma_tm.as_ref().unwrap().as_device_ptr().as_raw();
                    let block_x: u32 = 256;
                    let total = (ps * n) as u32;
                    let grid_x: u32 = (total + block_x - 1) / block_x;
                    let args: &mut [*mut c_void] = &mut [
                        &mut in_ptr as *mut _ as *mut c_void,
                        &mut rows as *mut _ as *mut c_void,
                        &mut cols as *mut _ as *mut c_void,
                        &mut out_ptr as *mut _ as *mut c_void,
                    ];
                    bt_stream
                        .launch(&transpose, (grid_x, 1, 1), (block_x, 1, 1), 0, args)
                        .map_err(|e| e.to_string())?;
                }
            }

            let pairs = pf
                .checked_mul(ps)
                .ok_or_else(|| "pf*ps overflow".to_string())?;
            let out_len = pairs
                .checked_mul(metrics_count)
                .ok_or_else(|| "pairs*metrics overflow".to_string())?;

	            if out_len > tile_out_cap {
	                return Err(format!(
	                    "tile output len {out_len} exceeds allocated capacity {tile_out_cap}"
	                ));
	            }
	
	            unsafe {
	                let mut f_ma_tm = scratch.d_fast_ma_tm.as_ref().unwrap().as_device_ptr().as_raw();
	                let mut s_ma_tm = scratch.d_slow_ma_tm.as_ref().unwrap().as_device_ptr().as_raw();
                let mut lr = d_lr.as_device_ptr().as_raw();
                let mut t_i = n as i32;
                let mut pf_tile_i = pf as i32;
                let mut ps_tile_i = ps as i32;
                let mut pf_total_i = pf as i32;
                let mut ps_total_i = ps as i32;
                let mut f_off = 0i32;
                let mut s_off = 0i32;
                let mut f_per = scratch.d_fast_p.as_ref().unwrap().as_device_ptr().as_raw();
                let mut s_per = scratch.d_slow_p.as_ref().unwrap().as_device_ptr().as_raw();
                let mut fv = first_valid as i32;
                let mut commission = commission;
                let mut eps_rel = eps_rel;
                let mut flags_u = flags;
                let mut m_i = metrics_count as i32;
                let mut out_ptr = scratch.d_tile.as_ref().unwrap().as_device_ptr().as_raw();

                let block_x: u32 = 256;
                let grid_x: u32 = ((pairs as u32) + block_x - 1) / block_x;
                let args: &mut [*mut c_void] = &mut [
                    &mut f_ma_tm as *mut _ as *mut c_void,
                    &mut s_ma_tm as *mut _ as *mut c_void,
                    &mut lr as *mut _ as *mut c_void,
                    &mut t_i as *mut _ as *mut c_void,
                    &mut pf_tile_i as *mut _ as *mut c_void,
                    &mut ps_tile_i as *mut _ as *mut c_void,
                    &mut pf_total_i as *mut _ as *mut c_void,
                    &mut ps_total_i as *mut _ as *mut c_void,
                    &mut f_off as *mut _ as *mut c_void,
                    &mut s_off as *mut _ as *mut c_void,
                    &mut f_per as *mut _ as *mut c_void,
                    &mut s_per as *mut _ as *mut c_void,
                    &mut fv as *mut _ as *mut c_void,
                    &mut commission as *mut _ as *mut c_void,
                    &mut eps_rel as *mut _ as *mut c_void,
                    &mut flags_u as *mut _ as *mut c_void,
                    &mut m_i as *mut _ as *mut c_void,
                    &mut out_ptr as *mut _ as *mut c_void,
                ];
	                bt_stream
	                    .launch(&backtest, (grid_x, 1, 1), (block_x, 1, 1), 0, args)
	                    .map_err(|e| e.to_string())?;
	            }
	
	            if let (Some(h), Some((f_min, f_max, s_min, s_max))) = (scratch.d_heatmap.as_ref(), heatmap_cfg) {
	                unsafe {
	                    let mut metrics_ptr = scratch.d_tile.as_ref().unwrap().as_device_ptr().as_raw();
	                    let mut pairs_i = pairs as i32;
	                    let mut m_i = metrics_count as i32;
	                    let mut pf_i = pf as i32;
	                    let mut ps_i = ps as i32;
	                    let mut f_per = scratch.d_fast_p.as_ref().unwrap().as_device_ptr().as_raw();
	                    let mut s_per = scratch.d_slow_p.as_ref().unwrap().as_device_ptr().as_raw();
	                    let mut obj_i = objective_i;
	                    let mut bins_f = heatmap_bins as i32;
	                    let mut bins_s = heatmap_bins as i32;
	                    let mut f_min_i = f_min as i32;
	                    let mut f_max_i = f_max as i32;
	                    let mut s_min_i = s_min as i32;
	                    let mut s_max_i = s_max as i32;
	                    let mut out_ptr = h.as_device_ptr().as_raw();
	
	                    let block_x: u32 = 256;
	                    let grid_x: u32 = ((pairs as u32) + block_x - 1) / block_x;
	                    let args: &mut [*mut c_void] = &mut [
	                        &mut metrics_ptr as *mut _ as *mut c_void,
	                        &mut pairs_i as *mut _ as *mut c_void,
	                        &mut m_i as *mut _ as *mut c_void,
	                        &mut pf_i as *mut _ as *mut c_void,
	                        &mut ps_i as *mut _ as *mut c_void,
	                        &mut f_per as *mut _ as *mut c_void,
	                        &mut s_per as *mut _ as *mut c_void,
	                        &mut obj_i as *mut _ as *mut c_void,
	                        &mut bins_f as *mut _ as *mut c_void,
	                        &mut bins_s as *mut _ as *mut c_void,
	                        &mut f_min_i as *mut _ as *mut c_void,
	                        &mut f_max_i as *mut _ as *mut c_void,
	                        &mut s_min_i as *mut _ as *mut c_void,
	                        &mut s_max_i as *mut _ as *mut c_void,
	                        &mut out_ptr as *mut _ as *mut c_void,
	                    ];
	                    bt_stream
	                        .launch(&heatmap_update, (grid_x, 1, 1), (block_x, 1, 1), 0, args)
	                        .map_err(|e| e.to_string())?;
	                }
	            }
	
	            if include_all {
	                bt_stream.synchronize().map_err(|e| e.to_string())?;
	
	                scratch.d_tile
	                    .as_ref()
	                    .unwrap()
	                    .copy_to(&mut tile_host_buf)
	                    .map_err(|e| e.to_string())?;
	
	                for i in 0..pf {
	                    let fast_len = fast_periods[f_start + i];
	                    for j in 0..ps {
	                        let slow_len = slow_periods[s_start + j];
	                        if fast_len >= slow_len {
	                            continue;
	                        }
	                        let src_base = (i * ps + j) * metrics_count;
	                        let pnl = tile_host_buf[src_base + 0] as f64;
	                        let trades =
	                            if metrics_count > 1 { tile_host_buf[src_base + 1] } else { 0.0 };
	                        let max_dd =
	                            if metrics_count > 2 { tile_host_buf[src_base + 2] as f64 } else { 0.0 };
	                        let mean = if metrics_count > 3 { tile_host_buf[src_base + 3] as f64 } else { 0.0 };
	                        let std = if metrics_count > 4 { tile_host_buf[src_base + 4] as f64 } else { 0.0 };
	                        let exposure =
	                            if metrics_count > 5 { tile_host_buf[src_base + 5] as f64 } else { 0.0 };
	                        let net_exposure =
	                            if metrics_count > 6 { tile_host_buf[src_base + 6] as f64 } else { 0.0 };
	                        let sharpe = if std > 0.0 { mean / std } else { 0.0 };
	
	                        let params = DoubleMaParams {
	                            fast_len,
	                            slow_len,
	                            fast_ma_id: 0,
	                            slow_ma_id: 0,
	                        };
	                        agg.push(
	                            params,
	                            Metrics {
	                                pnl,
	                                sharpe,
	                                max_dd,
	                                trades: if trades.is_finite() && trades >= 0.0 {
	                                    trades as u32
	                                } else {
	                                    0
	                                },
	                                exposure,
	                                net_exposure,
	                            },
	                        );
	                        processed = processed.saturating_add(1);
	                        if (processed & 0x1fff) == 0 && cancel.load(Ordering::Relaxed) {
	                            return Err("cancelled".to_string());
	                        }
	                    }
	                }
	            } else {
	                let fast_chunk = &fast_periods[f_start..f_start + pf];
	                let slow_chunk = &slow_periods[s_start..s_start + ps];
	                let s_len = slow_chunk.len();
	                let mut tile_valid: usize = 0;
	                for &f in fast_chunk {
	                    let idx = slow_chunk.partition_point(|&s| s <= f);
	                    tile_valid = tile_valid.saturating_add(s_len.saturating_sub(idx));
	                }
	
	                let blocks1 = (pairs + TOPK_BLOCK_ITEMS - 1) / TOPK_BLOCK_ITEMS;
	                let cand_len1 = blocks1.saturating_mul(topk_select).max(topk_select);
	                let cand_a = scratch.d_cand_a.as_mut().unwrap();
	                let cand_b = scratch.d_cand_b.as_mut().unwrap();
	
	                unsafe {
	                    let mut metrics_ptr = scratch.d_tile.as_ref().unwrap().as_device_ptr().as_raw();
	                    let mut pairs_i = pairs as i32;
	                    let mut m_i = metrics_count as i32;
	                    let mut pf_i = pf as i32;
	                    let mut ps_i = ps as i32;
	                    let mut f_per = scratch.d_fast_p.as_ref().unwrap().as_device_ptr().as_raw();
	                    let mut s_per = scratch.d_slow_p.as_ref().unwrap().as_device_ptr().as_raw();
	                    let mut obj_i = objective_i;
	                    let mut topk_i = topk_select as i32;
	                    let mut out_ptr = cand_a.as_device_ptr().as_raw();
	
	                    let block_x: u32 = 256;
	                    let grid_x: u32 = blocks1 as u32;
	                    let args: &mut [*mut c_void] = &mut [
	                        &mut metrics_ptr as *mut _ as *mut c_void,
	                        &mut pairs_i as *mut _ as *mut c_void,
	                        &mut m_i as *mut _ as *mut c_void,
	                        &mut pf_i as *mut _ as *mut c_void,
	                        &mut ps_i as *mut _ as *mut c_void,
	                        &mut f_per as *mut _ as *mut c_void,
	                        &mut s_per as *mut _ as *mut c_void,
	                        &mut obj_i as *mut _ as *mut c_void,
	                        &mut topk_i as *mut _ as *mut c_void,
	                        &mut out_ptr as *mut _ as *mut c_void,
	                    ];
	                    bt_stream
	                        .launch(&select_topk_pairs, (grid_x, 1, 1), (block_x, 1, 1), 0, args)
	                        .map_err(|e| e.to_string())?;
	                }
	
	                let mut blocks_prev = blocks1;
	                let mut in_len = cand_len1;
	                let mut in_is_a = true;
	                while blocks_prev > 1 {
	                    let blocks_next = (in_len + TOPK_BLOCK_ITEMS - 1) / TOPK_BLOCK_ITEMS;
	                    let out_len = blocks_next.saturating_mul(topk_select).max(topk_select);
	                    unsafe {
	                        let mut metrics_ptr = scratch.d_tile.as_ref().unwrap().as_device_ptr().as_raw();
	                        let mut pairs_i = pairs as i32;
	                        let mut m_i = metrics_count as i32;
	                        let mut pf_i = pf as i32;
	                        let mut ps_i = ps as i32;
	                        let mut f_per = scratch.d_fast_p.as_ref().unwrap().as_device_ptr().as_raw();
	                        let mut s_per = scratch.d_slow_p.as_ref().unwrap().as_device_ptr().as_raw();
	                        let mut obj_i = objective_i;
	                        let mut topk_i = topk_select as i32;
	                        let mut in_ptr = if in_is_a {
	                            cand_a.as_device_ptr().as_raw()
	                        } else {
	                            cand_b.as_device_ptr().as_raw()
	                        };
	                        let mut in_len_i = in_len as i32;
	                        let mut out_ptr = if in_is_a {
	                            cand_b.as_device_ptr().as_raw()
	                        } else {
	                            cand_a.as_device_ptr().as_raw()
	                        };
	
	                        let block_x: u32 = 256;
	                        let grid_x: u32 = blocks_next as u32;
	                        let args: &mut [*mut c_void] = &mut [
	                            &mut metrics_ptr as *mut _ as *mut c_void,
	                            &mut pairs_i as *mut _ as *mut c_void,
	                            &mut m_i as *mut _ as *mut c_void,
	                            &mut pf_i as *mut _ as *mut c_void,
	                            &mut ps_i as *mut _ as *mut c_void,
	                            &mut f_per as *mut _ as *mut c_void,
	                            &mut s_per as *mut _ as *mut c_void,
	                            &mut obj_i as *mut _ as *mut c_void,
	                            &mut topk_i as *mut _ as *mut c_void,
	                            &mut in_ptr as *mut _ as *mut c_void,
	                            &mut in_len_i as *mut _ as *mut c_void,
	                            &mut out_ptr as *mut _ as *mut c_void,
	                        ];
	                        bt_stream
	                            .launch(&select_topk_candidates, (grid_x, 1, 1), (block_x, 1, 1), 0, args)
	                            .map_err(|e| e.to_string())?;
	                    }
	                    in_is_a = !in_is_a;
	                    blocks_prev = blocks_next;
	                    in_len = out_len;
	                }
	
	                unsafe {
	                    let mut metrics_ptr = scratch.d_tile.as_ref().unwrap().as_device_ptr().as_raw();
	                    let mut pairs_i = pairs as i32;
	                    let mut m_i = metrics_count as i32;
	                    let mut pf_i = pf as i32;
	                    let mut ps_i = ps as i32;
	                    let mut f_per = scratch.d_fast_p.as_ref().unwrap().as_device_ptr().as_raw();
	                    let mut s_per = scratch.d_slow_p.as_ref().unwrap().as_device_ptr().as_raw();
	                    let mut top_ptr = if in_is_a {
	                        cand_a.as_device_ptr().as_raw()
	                    } else {
	                        cand_b.as_device_ptr().as_raw()
	                    };
	                    let mut topk_i = topk_select as i32;
	                    let mut out_f = scratch.d_top_fast.as_ref().unwrap().as_device_ptr().as_raw();
	                    let mut out_s = scratch.d_top_slow.as_ref().unwrap().as_device_ptr().as_raw();
	                    let mut out_m = scratch.d_top_metrics.as_ref().unwrap().as_device_ptr().as_raw();
	
	                    let block_x: u32 = 256;
	                    let grid_x: u32 = ((topk_select as u32) + block_x - 1) / block_x;
	                    let args: &mut [*mut c_void] = &mut [
	                        &mut metrics_ptr as *mut _ as *mut c_void,
	                        &mut pairs_i as *mut _ as *mut c_void,
	                        &mut m_i as *mut _ as *mut c_void,
	                        &mut pf_i as *mut _ as *mut c_void,
	                        &mut ps_i as *mut _ as *mut c_void,
	                        &mut f_per as *mut _ as *mut c_void,
	                        &mut s_per as *mut _ as *mut c_void,
	                        &mut top_ptr as *mut _ as *mut c_void,
	                        &mut topk_i as *mut _ as *mut c_void,
	                        &mut out_f as *mut _ as *mut c_void,
	                        &mut out_s as *mut _ as *mut c_void,
	                        &mut out_m as *mut _ as *mut c_void,
	                    ];
	                    bt_stream
	                        .launch(&gather_topk, (grid_x, 1, 1), (block_x, 1, 1), 0, args)
	                        .map_err(|e| e.to_string())?;
	                }
	
	                bt_stream.synchronize().map_err(|e| e.to_string())?;

	                if let Some(w) = export_writer.as_mut() {
	                    scratch.d_tile
	                        .as_ref()
	                        .unwrap()
	                        .copy_to(&mut tile_host_buf)
	                        .map_err(|e| e.to_string())?;

	                    let mut exported = 0usize;
	                    for i in 0..pf {
	                        let fast_len = fast_periods[f_start + i];
	                        for j in 0..ps {
	                            let slow_len = slow_periods[s_start + j];
	                            if fast_len >= slow_len {
	                                continue;
	                            }

	                            let src_base = (i * ps + j) * metrics_count;
	                            if src_base + metrics_count > tile_host_buf.len() {
	                                continue;
	                            }
	                            let pnl = tile_host_buf[src_base + 0] as f64;
	                            let trades =
	                                if metrics_count > 1 { tile_host_buf[src_base + 1] } else { 0.0 };
	                            let max_dd =
	                                if metrics_count > 2 { tile_host_buf[src_base + 2] as f64 } else { 0.0 };
	                            let mean = if metrics_count > 3 { tile_host_buf[src_base + 3] as f64 } else { 0.0 };
	                            let std = if metrics_count > 4 { tile_host_buf[src_base + 4] as f64 } else { 0.0 };
	                            let exposure =
	                                if metrics_count > 5 { tile_host_buf[src_base + 5] as f64 } else { 0.0 };
	                            let net_exposure =
	                                if metrics_count > 6 { tile_host_buf[src_base + 6] as f64 } else { 0.0 };
	                            let sharpe = if std > 0.0 { mean / std } else { 0.0 };
	                            let trades_u = if trades.is_finite() && trades >= 0.0 {
	                                trades as u32
	                            } else {
	                                0
	                            };

	                            let score = match objective {
	                                ObjectiveKind::Pnl => pnl,
	                                ObjectiveKind::Sharpe => sharpe,
	                                ObjectiveKind::MaxDrawdown => -max_dd,
	                            };

	                            writeln!(
	                                w,
	                                "{},{},{},{},{},{},{},{},{},{},{}",
	                                fast_ma.as_str(),
	                                slow_ma.as_str(),
	                                fast_len,
	                                slow_len,
	                                pnl,
	                                sharpe,
	                                max_dd,
	                                trades_u,
	                                exposure,
	                                net_exposure,
	                                score
	                            )
	                            .map_err(|e| e.to_string())?;

	                            exported = exported.saturating_add(1);
	                            if (exported & 0x1fff) == 0 && cancel.load(Ordering::Relaxed) {
	                                return Err("cancelled".to_string());
	                            }
	                        }
	                    }
	                }
	
	                scratch.d_top_fast
	                    .as_ref()
	                    .unwrap()
	                    .copy_to(&mut top_fast_host)
	                    .map_err(|e| e.to_string())?;
	                scratch.d_top_slow
	                    .as_ref()
	                    .unwrap()
	                    .copy_to(&mut top_slow_host)
	                    .map_err(|e| e.to_string())?;
	                scratch.d_top_metrics
	                    .as_ref()
	                    .unwrap()
	                    .copy_to(&mut top_metrics_host)
	                    .map_err(|e| e.to_string())?;
	
	                for r in 0..topk_select {
	                    let fast_len_i = top_fast_host[r];
	                    let slow_len_i = top_slow_host[r];
	                    if fast_len_i <= 0 || slow_len_i <= 0 {
	                        continue;
	                    }
	                    let fast_len = fast_len_i as u16;
	                    let slow_len = slow_len_i as u16;
	                    if fast_len >= slow_len {
	                        continue;
	                    }
	
	                    let base = r.saturating_mul(metrics_count);
	                    if base >= top_metrics_host.len() {
	                        continue;
	                    }
	                    let pnl = top_metrics_host[base + 0] as f64;
	                    let trades = if metrics_count > 1 {
	                        top_metrics_host[base + 1]
	                    } else {
	                        0.0
	                    };
	                    let max_dd = if metrics_count > 2 {
	                        top_metrics_host[base + 2] as f64
	                    } else {
	                        0.0
	                    };
	                    let mean = if metrics_count > 3 {
	                        top_metrics_host[base + 3] as f64
	                    } else {
	                        0.0
	                    };
	                    let std = if metrics_count > 4 {
	                        top_metrics_host[base + 4] as f64
	                    } else {
	                        0.0
	                    };
	                    let exposure = if metrics_count > 5 {
	                        top_metrics_host[base + 5] as f64
	                    } else {
	                        0.0
	                    };
	                    let net_exposure = if metrics_count > 6 {
	                        top_metrics_host[base + 6] as f64
	                    } else {
	                        0.0
	                    };
	                    let sharpe = if std > 0.0 { mean / std } else { 0.0 };
	
	                    let params = DoubleMaParams {
	                        fast_len,
	                        slow_len,
	                        fast_ma_id: 0,
	                        slow_ma_id: 0,
	                    };
	                    agg.push(
	                        params,
	                        Metrics {
	                            pnl,
	                            sharpe,
	                            max_dd,
	                            trades: if trades.is_finite() && trades >= 0.0 {
	                                trades as u32
	                            } else {
	                                0
	                            },
	                            exposure,
	                            net_exposure,
	                        },
	                    );
	                }
	
	                processed = processed.saturating_add(tile_valid);
	                if cancel.load(Ordering::Relaxed) {
	                    return Err("cancelled".to_string());
	                }
	            }
	
	            emit_progress(progress, processed, total_pairs, "gpu-kernel");
	
	            s_start += ps;
	        }

	        f_start += pf;
	    }

	    let heatmap_out: Option<OptimizationHeatmap> = if !include_all {
	        if let (Some(h), Some((f_min, f_max, s_min, s_max))) = (scratch.d_heatmap.as_ref(), heatmap_cfg) {
	            let heatmap_len = heatmap_bins.saturating_mul(heatmap_bins);
	            if heatmap_len == 0 {
	                None
	            } else {
	                let mut host = vec![f32::NEG_INFINITY; heatmap_len];
	                h.copy_to(&mut host).map_err(|e| e.to_string())?;
	                let values: Vec<Option<f64>> = host
	                    .into_iter()
	                    .map(|v| if v.is_finite() && v != f32::NEG_INFINITY { Some(v as f64) } else { None })
	                    .collect();
	                Some(OptimizationHeatmap {
	                    bins_fast: heatmap_bins,
	                    bins_slow: heatmap_bins,
	                    fast_min: f_min,
	                    fast_max: f_max,
	                    slow_min: s_min,
	                    slow_max: s_max,
	                    values,
	                })
	            }
	        } else {
	            None
	        }
	    } else {
	        None
	    };
	
	    let mut result = agg
	        .finalize()
	        .ok_or_else(|| "grid search produced no result".to_string())?;

	    if !include_all {
	        result.num_combos = total_pairs;
	        if heatmap_out.is_some() {
	            result.heatmap = heatmap_out;
	        }
	    }
	
	    let skip_recompute = matches!(std::env::var("VECTORBT_KERNEL_SKIP_RECOMPUTE"), Ok(ref v) if v == "1" || v.eq_ignore_ascii_case("true"));
	    if fast_pass && metrics_count < METRICS_COUNT && !skip_recompute {
	        if cancel.load(Ordering::Relaxed) {
	            return Err("cancelled".to_string());
        }

        let mut fast_needed: Vec<u16> = Vec::with_capacity(result.top.len() + 1);
        let mut slow_needed: Vec<u16> = Vec::with_capacity(result.top.len() + 1);
        fast_needed.push(result.best_params.fast_len);
        slow_needed.push(result.best_params.slow_len);
        for (p, _) in result.top.iter() {
            fast_needed.push(p.fast_len);
            slow_needed.push(p.slow_len);
        }
        fast_needed.sort_unstable();
        fast_needed.dedup();
        slow_needed.sort_unstable();
        slow_needed.dedup();

        let total_pairs_subset = fast_needed.len().saturating_mul(slow_needed.len());
        let full = optimize_double_ma_gpu_kernel_impl(
            rt,
            candles,
            &fast_needed,
            &slow_needed,
            fast_ma_type,
            slow_ma_type,
            ma_source,
            fast_ma_params,
            slow_ma_params,
            strategy,
            objective,
            0,
            true,
            cfg_recompute,
            None,
            total_pairs_subset,
            0,
            cancel,
            None,
        )?;
        let all = full.all.ok_or_else(|| "internal error: expected include_all=true".to_string())?;

        let mut map: HashMap<(u16, u16), Metrics> = HashMap::with_capacity(all.len());
        for (p, m) in all {
            map.insert((p.fast_len, p.slow_len), m);
        }

        result.best_metrics = map
            .get(&(result.best_params.fast_len, result.best_params.slow_len))
            .cloned()
            .ok_or_else(|| "best metrics missing from recompute pass".to_string())?;

        for (p, m) in result.top.iter_mut() {
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".to_string());
            }
            if let Some(full_m) = map.get(&(p.fast_len, p.slow_len)) {
                *m = full_m.clone();
            }
        }
    }

    if let Some(w) = export_writer.as_mut() {
        w.flush().map_err(|e| e.to_string())?;
    }

    if !kernel_persist_buffers_enabled() {
        rt.ma.clear_cached_constants();
        rt.scratch.clear();
    }

    Ok(result)
}
