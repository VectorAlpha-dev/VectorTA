# SIMD Scalar-Selection Overrides Tracker

Purpose: track progress on forcing **scalar** selection for **single-series outputs** when `Kernel::Auto` is used, for indicators where AVX2/AVX512 is currently slower or a stub.

Constraints (repo policy):
- Do not change unit test reference values.
- Do not run `cargo fmt`/rustfmt automatically.

---

## Goal

For the indicators listed below:
- Ensure `Kernel::Auto` resolves to `Kernel::Scalar` for **single-series** APIs.
- Keep SIMD code in-tree (do not delete kernels), but **short-circuit runtime selection** to scalar for now.
- Do **not** change batch selection behavior (batch performance currently unknown).

Non-goals (for this task):
- Rewriting SIMD kernels for speed.
- Changing batch kernels (unless you explicitly approve per-indicator).
- Changing CUDA behavior.

---

## Implementation rule (this task)

Per indicator module:
1. Identify the single-series kernel resolution point (usually: `if kernel == Auto { detect_best_kernel() } else { kernel }`).
2. Change only the `Auto` branch to resolve to `Kernel::Scalar`.
3. Leave explicit user requests alone (`Kernel::Avx2`/`Kernel::Avx512` still works if passed explicitly).
4. Do not touch any batch selection (`detect_best_batch_kernel()` etc).

Validation per indicator (as we go):
- Run its focused lib tests.
- Run the single-series Criterion IDs for scalar vs AVX2/AVX512 and record the results below.

---

## Benchmarking Notes

- Use `cargo bench --features nightly-avx --bench indicator_benchmark -- --list` to discover IDs.
- Recommended quick confirmation run: add Criterion args `-- --sample-size 10 --warm-up-time 1 --measurement-time 2`.
- Record results as `scalar <= avx2` and `scalar <= avx512` (or paste times).

## Confirmed Scope (from you)

- `Kernel::Auto` should always resolve to `Kernel::Scalar` for the listed indicators’ **single-series** path.
- Explicit `Kernel::{Scalar,Avx2,Avx512}` selection remains available.
- Batch selection remains unchanged.

---

## Tracker

Legend:
- `A` = audited (where `Auto` resolves for single-series)
- `S` = `Auto => Scalar` implemented for single-series
- `T` = tests run (focused)
- `B` = benches run (scalar vs avx2 vs avx512 recorded)

### `src/indicators/*.rs`

| Indicator | File | A | S | T | B | Benchmark Notes |
|---|---|---:|---:|---:|---:|---|
| alligator | `src/indicators/alligator.rs` | [ ] | [ ] | [ ] | [ ] | multi-output |
| alphatrend | `src/indicators/alphatrend.rs` | [ ] | [ ] | [ ] | [ ] |  |
| apo | `src/indicators/apo.rs` | [ ] | [ ] | [ ] | [ ] |  |
| aroon | `src/indicators/aroon.rs` | [ ] | [ ] | [ ] | [ ] | multi-output |
| atr | `src/indicators/atr.rs` | [ ] | [ ] | [ ] | [ ] |  |
| avsl | `src/indicators/avsl.rs` | [ ] | [ ] | [ ] | [ ] |  |
| bollinger_bands | `src/indicators/bollinger_bands.rs` | [ ] | [ ] | [ ] | [ ] | multi-output |
| bollinger_bands_width | `src/indicators/bollinger_bands_width.rs` | [ ] | [ ] | [ ] | [ ] |  |
| cci_cycle | `src/indicators/cci_cycle.rs` | [ ] | [ ] | [ ] | [ ] |  |
| cfo | `src/indicators/cfo.rs` | [ ] | [ ] | [ ] | [ ] |  |
| cg | `src/indicators/cg.rs` | [ ] | [ ] | [ ] | [ ] |  |
| chandelier_exit | `src/indicators/chandelier_exit.rs` | [ ] | [ ] | [ ] | [ ] | multi-output |
| cksp | `src/indicators/cksp.rs` | [ ] | [ ] | [ ] | [ ] | multi-output |
| cmo | `src/indicators/cmo.rs` | [ ] | [ ] | [ ] | [ ] |  |
| correlation_cycle | `src/indicators/correlation_cycle.rs` | [ ] | [ ] | [ ] | [ ] |  |
| cvi | `src/indicators/cvi.rs` | [ ] | [ ] | [ ] | [ ] |  |
| damiani_volatmeter | `src/indicators/damiani_volatmeter.rs` | [ ] | [ ] | [ ] | [ ] |  |
| donchian | `src/indicators/donchian.rs` | [ ] | [ ] | [ ] | [ ] | multi-output |
| emv | `src/indicators/emv.rs` | [ ] | [ ] | [ ] | [ ] |  |
| er | `src/indicators/er.rs` | [ ] | [ ] | [ ] | [ ] |  |
| eri | `src/indicators/eri.rs` | [ ] | [ ] | [ ] | [ ] | multi-output |
| fvg_trailing_stop | `src/indicators/fvg_trailing_stop.rs` | [ ] | [ ] | [ ] | [ ] | multi-output |
| gatorosc | `src/indicators/gatorosc.rs` | [ ] | [ ] | [ ] | [ ] | multi-output |
| halftrend | `src/indicators/halftrend.rs` | [ ] | [ ] | [ ] | [ ] | multi-output |
| ift_rsi | `src/indicators/ift_rsi.rs` | [ ] | [ ] | [ ] | [ ] |  |
| jma (non-MA variant?) | `src/indicators/moving_averages/jma.rs` | [ ] | [ ] | [ ] | [ ] | see MA section |
| kaufmanstop | `src/indicators/kaufmanstop.rs` | [ ] | [ ] | [ ] | [ ] |  |
| keltner | `src/indicators/keltner.rs` | [ ] | [ ] | [ ] | [ ] | multi-output |
| kurtosis | `src/indicators/kurtosis.rs` | [ ] | [ ] | [ ] | [ ] |  |
| kvo | `src/indicators/kvo.rs` | [ ] | [ ] | [ ] | [ ] |  |
| linearreg_angle | `src/indicators/linearreg_angle.rs` | [ ] | [ ] | [ ] | [ ] |  |
| linearreg_intercept | `src/indicators/linearreg_intercept.rs` | [ ] | [ ] | [ ] | [ ] |  |
| linearreg_slope | `src/indicators/linearreg_slope.rs` | [ ] | [ ] | [ ] | [ ] |  |
| lpc | `src/indicators/lpc.rs` | [ ] | [ ] | [ ] | [ ] |  |
| lrsi | `src/indicators/lrsi.rs` | [ ] | [ ] | [ ] | [ ] |  |
| mab | `src/indicators/mab.rs` | [ ] | [ ] | [ ] | [ ] |  |
| macd | `src/indicators/macd.rs` | [ ] | [ ] | [ ] | [ ] | multi-output |
| macz | `src/indicators/macz.rs` | [ ] | [ ] | [ ] | [ ] |  |
| mama | `src/indicators/moving_averages/mama.rs` | [ ] | [ ] | [ ] | [ ] | see MA section |
| medium_ad | `src/indicators/medium_ad.rs` | [ ] | [ ] | [ ] | [ ] |  |
| mod_god_mode | `src/indicators/mod_god_mode.rs` | [ ] | [ ] | [ ] | [ ] |  |
| mom | `src/indicators/mom.rs` | [ ] | [ ] | [ ] | [ ] |  |
| msw | `src/indicators/msw.rs` | [ ] | [ ] | [ ] | [ ] | multi-output |
| mfi? (not in your list) | `src/indicators/mfi.rs` | [ ] | [ ] | [ ] | [ ] | confirm scope |
| natr | `src/indicators/natr.rs` | [ ] | [ ] | [ ] | [ ] |  |
| net_myrsi | `src/indicators/net_myrsi.rs` | [ ] | [ ] | [ ] | [ ] |  |
| nvi | `src/indicators/nvi.rs` | [ ] | [ ] | [ ] | [ ] |  |
| obv | `src/indicators/obv.rs` | [ ] | [ ] | [ ] | [ ] |  |
| ott | `src/indicators/ott.rs` | [ ] | [ ] | [ ] | [ ] |  |
| percentile_nearest_rank | `src/indicators/percentile_nearest_rank.rs` | [ ] | [ ] | [ ] | [ ] |  |
| pma | `src/indicators/pma.rs` | [ ] | [ ] | [ ] | [ ] |  |
| ppo | `src/indicators/ppo.rs` | [ ] | [ ] | [ ] | [ ] |  |
| pvi | `src/indicators/pvi.rs` | [ ] | [ ] | [ ] | [ ] |  |
| qstick | `src/indicators/qstick.rs` | [ ] | [ ] | [ ] | [ ] |  |
| range_filter | `src/indicators/range_filter.rs` | [ ] | [ ] | [ ] | [ ] |  |
| rocr | `src/indicators/rocr.rs` | [ ] | [ ] | [ ] | [ ] |  |
| rsi | `src/indicators/rsi.rs` | [ ] | [ ] | [ ] | [ ] |  |
| rsx | `src/indicators/rsx.rs` | [ ] | [ ] | [ ] | [ ] |  |
| rvi | `src/indicators/rvi.rs` | [ ] | [ ] | [ ] | [ ] |  |
| sar | `src/indicators/sar.rs` | [ ] | [ ] | [ ] | [ ] |  |
| srsi | `src/indicators/srsi.rs` | [ ] | [ ] | [ ] | [ ] |  |
| stc | `src/indicators/stc.rs` | [ ] | [ ] | [ ] | [ ] |  |
| supertrend | `src/indicators/supertrend.rs` | [ ] | [ ] | [ ] | [ ] | multi-output |
| ultosc | `src/indicators/ultosc.rs` | [ ] | [ ] | [ ] | [ ] |  |
| var | `src/indicators/var.rs` | [ ] | [ ] | [ ] | [ ] |  |
| vlma | `src/indicators/vlma.rs` | [ ] | [ ] | [ ] | [ ] |  |
| vpci | `src/indicators/vpci.rs` | [ ] | [ ] | [ ] | [ ] |  |
| vpt | `src/indicators/vpt.rs` | [ ] | [ ] | [ ] | [ ] |  |
| wclprice | `src/indicators/wclprice.rs` | [ ] | [ ] | [ ] | [ ] |  |
| wto | `src/indicators/wto.rs` | [ ] | [ ] | [ ] | [ ] | multi-output? |
| zscore | `src/indicators/zscore.rs` | [ ] | [ ] | [ ] | [ ] |  |

### `src/indicators/moving_averages/*.rs`

| Indicator | File | A | S | T | B | Benchmark Notes |
|---|---|---:|---:|---:|---:|---|
| buff_averages | `src/indicators/moving_averages/buff_averages.rs` | [ ] | [ ] | [ ] | [ ] |  |
| dma | `src/indicators/moving_averages/dma.rs` | [ ] | [ ] | [ ] | [ ] |  |
| edcf | `src/indicators/moving_averages/edcf.rs` | [ ] | [ ] | [ ] | [ ] |  |
| ehlers_ecema | `src/indicators/moving_averages/ehlers_ecema.rs` | [ ] | [ ] | [ ] | [ ] | “ecema” in your list |
| ehlers_pma | `src/indicators/moving_averages/ehlers_pma.rs` | [ ] | [ ] | [ ] | [ ] |  |
| ema | `src/indicators/moving_averages/ema.rs` | [ ] | [ ] | [ ] | [ ] |  |
| gaussian | `src/indicators/moving_averages/gaussian.rs` | [ ] | [ ] | [ ] | [ ] |  |
| highpass | `src/indicators/moving_averages/highpass.rs` | [ ] | [ ] | [ ] | [ ] |  |
| highpass_2_pole | `src/indicators/moving_averages/highpass_2_pole.rs` | [ ] | [ ] | [ ] | [ ] |  |
| hwma | `src/indicators/moving_averages/hwma.rs` | [ ] | [ ] | [ ] | [ ] |  |
| jma | `src/indicators/moving_averages/jma.rs` | [ ] | [ ] | [ ] | [ ] |  |
| jsa | `src/indicators/moving_averages/jsa.rs` | [ ] | [ ] | [ ] | [ ] |  |
| kama | `src/indicators/moving_averages/kama.rs` | [ ] | [ ] | [ ] | [ ] |  |
| linreg | `src/indicators/moving_averages/linreg.rs` | [ ] | [ ] | [ ] | [ ] |  |
| maaq | `src/indicators/moving_averages/maaq.rs` | [ ] | [ ] | [ ] | [ ] |  |
| mama | `src/indicators/moving_averages/mama.rs` | [ ] | [ ] | [ ] | [ ] |  |
| mwdx | `src/indicators/moving_averages/mwdx.rs` | [ ] | [ ] | [ ] | [ ] |  |
| nama | `src/indicators/moving_averages/nama.rs` | [ ] | [ ] | [ ] | [ ] |  |
| net_myrsi? (not MA) | `src/indicators/net_myrsi.rs` | [ ] | [ ] | [ ] | [ ] | confirm scope |
| pma (MA vs indicator) | `src/indicators/pma.rs` | [ ] | [ ] | [ ] | [ ] | confirm which “pma” you meant |
| reflex | `src/indicators/moving_averages/reflex.rs` | [ ] | [ ] | [ ] | [ ] |  |
| supersmoother | `src/indicators/moving_averages/supersmoother.rs` | [ ] | [ ] | [ ] | [ ] |  |
| supersmoother_3_pole | `src/indicators/moving_averages/supersmoother_3_pole.rs` | [ ] | [ ] | [ ] | [ ] |  |
| swma | `src/indicators/moving_averages/swma.rs` | [ ] | [ ] | [ ] | [ ] |  |
| tema | `src/indicators/moving_averages/tema.rs` | [ ] | [ ] | [ ] | [ ] |  |
| tilson | `src/indicators/moving_averages/tilson.rs` | [ ] | [ ] | [ ] | [ ] |  |
| tradjema | `src/indicators/moving_averages/tradjema.rs` | [ ] | [ ] | [ ] | [ ] |  |
| trima | `src/indicators/moving_averages/trima.rs` | [ ] | [ ] | [ ] | [ ] |  |
| uma | `src/indicators/moving_averages/uma.rs` | [ ] | [ ] | [ ] | [ ] |  |
| volume_adjusted_ma | `src/indicators/moving_averages/volume_adjusted_ma.rs` | [ ] | [ ] | [ ] | [ ] |  |
| vwma | `src/indicators/moving_averages/vwma.rs` | [ ] | [ ] | [ ] | [ ] |  |
| wilders | `src/indicators/moving_averages/wilders.rs` | [ ] | [ ] | [ ] | [ ] |  |
| wma | `src/indicators/moving_averages/wma.rs` | [ ] | [ ] | [ ] | [ ] |  |
| zlema | `src/indicators/moving_averages/zlema.rs` | [ ] | [ ] | [ ] | [ ] |  |
