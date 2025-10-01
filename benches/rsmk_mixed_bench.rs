use anyhow::Result;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use my_project::indicators::moving_averages::ma::{ma, MaData};
use my_project::indicators::rsmk::{rsmk_with_kernel, RsmkInput, RsmkParams};
use std::f64::consts::PI;

fn bench_one(c: &mut Criterion, len: usize) -> Result<()> {
    let mut group = c.benchmark_group("rsmk_mixed");
    group.throughput(criterion::Throughput::Elements(len as u64));

    // Generate synthetic, positive series of length `len`
    let mut main_buf = vec![f64::NAN; len];
    let mut comp_buf = vec![f64::NAN; len];
    for i in 0..len {
        let x = i as f64;
        // simple smooth function with trend; strictly positive
        let v = 10_000.0 + (x * 0.001).sin() * 50.0 + 0.05 * (x / 1000.0);
        main_buf[i] = v;
        // small offset to avoid exact equality but keep positivity
        comp_buf[i] = v * (1.0 + 0.0003 * (x * PI / 8192.0).sin());
    }
    let main = &main_buf[..];
    let compare = &comp_buf[..];

    // Params: default periods, mixed MA types
    let lookback = 90usize;
    let period = 3usize;
    let signal_period = 20usize;

    // Fused path via rsmk_with_kernel
    let input = RsmkInput::from_slices(
        main,
        compare,
        RsmkParams {
            lookback: Some(lookback),
            period: Some(period),
            signal_period: Some(signal_period),
            matype: Some("ema".to_string()),
            signal_matype: Some("sma".to_string()),
        },
    );

    group.bench_with_input(BenchmarkId::new("fused_ema_sma", len), &len, |b, _| {
        b.iter(|| {
            let out = rsmk_with_kernel(black_box(&input), my_project::utilities::enums::Kernel::Auto)
                .expect("rsmk fused");
            black_box(out);
        });
    });

    // Generic path: replicate old fallback (ma over momentum, then ma over indicator)
    group.bench_with_input(BenchmarkId::new("generic_ema_sma", len), &len, |b, _| {
        b.iter(|| {
            // log ratio
            let mut lr = vec![f64::NAN; len];
            for i in 0..len {
                let m = main[i];
                let c0 = compare[i];
                lr[i] = if m.is_nan() || c0.is_nan() || c0 == 0.0 { f64::NAN } else { (m / c0).ln() };
            }
            let first = lr.iter().position(|x| !x.is_nan()).unwrap_or(len);
            if first == len { return; }

            // momentum
            let mut mom = vec![f64::NAN; len];
            for i in (first + lookback)..len {
                let a = lr[i];
                let b = lr[i - lookback];
                mom[i] = if a.is_nan() || b.is_nan() { f64::NAN } else { a - b };
            }

            // indicator EMA over momentum
            let mut indicator = ma("ema", MaData::Slice(&mom), period).unwrap();
            for v in &mut indicator { *v *= 100.0; }
            // signal SMA over indicator
            let _signal = ma("sma", MaData::Slice(&indicator), signal_period).unwrap();
        });
    });

    group.finish();
    Ok(())
}

fn rsmk_mixed_benchmarks(c: &mut Criterion) {
    // Try two sizes representative of realistic workloads
    for &len in &[10_000usize, 100_000usize] {
        bench_one(c, len).expect("bench_one");
    }
}

criterion_group!(benches, rsmk_mixed_benchmarks);
criterion_main!(benches);
