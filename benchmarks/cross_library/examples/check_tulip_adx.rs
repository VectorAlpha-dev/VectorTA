use cross_library_benchmark::tulip;
use cross_library_benchmark::utils::CandleData;
use std::path::Path;
use std::time::Instant;

fn main() {
    let path = Path::new("../../src/data/4kCandles.csv");
    let data = CandleData::from_csv(path).expect("failed to load data");
    let size = data.len();


    let inputs: Vec<&[f64]> = vec![&data.high, &data.low, &data.close];
    let options: Vec<f64> = vec![14.0];
    let mut out = vec![0.0; size];
    let mut outs: Vec<&mut [f64]> = vec![&mut out[..]];

    unsafe {
        match tulip::get_start_index("adx", &options) {
            Ok(start) => println!("start index for adx = {}", start),
            Err(e) => {
                eprintln!("get_start_index error: {}", e);
                return;
            }
        }

        let t0 = Instant::now();
        let res = tulip::call_indicator("adx", size, &inputs, &options, &mut outs);
        let dt = t0.elapsed();
        println!("call_indicator returned: {:?} in {:?}", res, dt);

        let s: f64 = out.iter().take(10).sum();
        println!("first10 sum = {}", s);
    }
}

