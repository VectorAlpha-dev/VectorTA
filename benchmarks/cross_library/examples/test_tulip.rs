use cross_library_benchmark::tulip;

fn main() {
    
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let mut output = vec![0.0; data.len()];

    unsafe {
        let result = tulip::call_indicator(
            "sma",
            data.len(),
            &[&data],
            &[3.0], 
            &mut [&mut output],
        );

        match result {
            Ok(()) => {
                println!("Tulip SMA calculation successful!");
                println!("Input:  {:?}", data);
                println!("Output: {:?}", output);
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
    }
}