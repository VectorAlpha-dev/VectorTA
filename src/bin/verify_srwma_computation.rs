use my_project::indicators::moving_averages::srwma::{srwma_with_kernel, SrwmaInput, SrwmaParams};
use my_project::utilities::data_loader::read_candles_from_csv;
use my_project::utilities::enums::Kernel;
use std::time::Instant;

fn main() {
	println!("Verifying SRWMA computation in Rust...\n");

	// Load the same CSV file
	let candles = read_candles_from_csv("src/data/1MillionCandles.csv").expect("Failed to load CSV");

	let sizes = [1000, 10000, 100000];
	let period = 14;

	for &size in &sizes {
		let data = &candles.close[..size.min(candles.close.len())];
		println!("\nData size: {}", size);
		println!("First 5 values: {:?}", &data[..5.min(data.len())]);
		println!("Last 5 values: {:?}", &data[data.len().saturating_sub(5)..]);

		let params = SrwmaParams { period: Some(period) };
		let input = SrwmaInput::from_slice(data, params);

		// Warmup
		for _ in 0..10 {
			let _ = srwma_with_kernel(&input, Kernel::Scalar);
		}

		// Time the computation
		let start = Instant::now();
		let result = srwma_with_kernel(&input, Kernel::Scalar).unwrap();
		let duration = start.elapsed();

		println!("\nComputation time: {:.3} ms", duration.as_secs_f64() * 1000.0);
		println!("Result length: {}", result.values.len());

		// Count NaN values
		let nan_count = result.values.iter().filter(|&&x| x.is_nan()).count();
		let first_non_nan = result.values.iter().position(|&x| !x.is_nan());

		println!("NaN count (warmup): {}", nan_count);
		println!("First non-NaN index: {:?}", first_non_nan);
		println!("Expected warmup: {}", period + 1);

		if let Some(idx) = first_non_nan {
			let first_values: Vec<String> = result.values[idx..idx + 5.min(result.values.len() - idx)]
				.iter()
				.map(|&v| format!("{:.6}", v))
				.collect();
			println!("First 5 non-NaN values: {:?}", first_values);

			let last_values: Vec<String> = result.values[result.values.len().saturating_sub(5)..]
				.iter()
				.map(|&v| format!("{:.6}", v))
				.collect();
			println!("Last 5 values: {:?}", last_values);
		}
	}

	// Test different kernels on 100k data
	println!("\n\nTesting different kernels on 100k data:");
	let data = &candles.close[..100000.min(candles.close.len())];
	let params = SrwmaParams { period: Some(period) };
	let input = SrwmaInput::from_slice(data, params);

	for kernel in [Kernel::Scalar, Kernel::Avx2, Kernel::Avx512, Kernel::Auto] {
		// Warmup
		for _ in 0..10 {
			let _ = srwma_with_kernel(&input, kernel);
		}

		let mut times = Vec::new();
		for _ in 0..100 {
			let start = Instant::now();
			let _ = srwma_with_kernel(&input, kernel).unwrap();
			times.push(start.elapsed().as_secs_f64() * 1000.0);
		}

		times.sort_by(|a, b| a.partial_cmp(b).unwrap());
		let median = times[times.len() / 2];
		let mean = times.iter().sum::<f64>() / times.len() as f64;

		println!("\nKernel {:?}:", kernel);
		println!("  Median: {:.3} ms", median);
		println!("  Mean: {:.3} ms", mean);
		println!("  Min: {:.3} ms", times[0]);
		println!("  Max: {:.3} ms", times[times.len() - 1]);
	}

	println!("\nDone!");
}
