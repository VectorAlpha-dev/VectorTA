#![feature(stdarch_x86_avx512)]
use std::arch::x86_64::*;

unsafe fn horiz_sum_old(z: __m512d) -> f64 {
    let hi = _mm512_extractf64x4_pd(z, 1);
    let lo = _mm512_castpd512_pd256(z);
    let red = _mm256_add_pd(hi, lo);
    let red = _mm256_hadd_pd(red, red);
    _mm_cvtsd_f64(_mm256_castpd256_pd128(red))
}

unsafe fn horiz_sum_new(z: __m512d) -> f64 {
    // Extract upper and lower 256-bit halves
    let hi = _mm512_extractf64x4_pd(z, 1);
    let lo = _mm512_castpd512_pd256(z);
    // Add them together (now have 4 sums)
    let sum256 = _mm256_add_pd(hi, lo);
    // Horizontal add to get 2 sums
    let sum128 = _mm256_hadd_pd(sum256, sum256);
    // Extract high and low 128-bit parts and add
    let hi128 = _mm256_extractf128_pd(sum128, 1);
    let lo128 = _mm256_castpd256_pd128(sum128);
    let final_sum = _mm_add_pd(hi128, lo128);
    // Extract the final scalar result
    _mm_cvtsd_f64(final_sum)
}

fn main() {
    if !is_x86_feature_detected!("avx512f") {
        println!("AVX512 not supported");
        return;
    }
    
    unsafe {
        // Test vector [1, 2, 3, 4, 5, 6, 7, 8]
        let test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let vec = _mm512_loadu_pd(test_data.as_ptr());
        
        let old_sum = horiz_sum_old(vec);
        let new_sum = horiz_sum_new(vec);
        let expected = 1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0;
        
        println!("Test vector: {:?}", test_data);
        println!("Old horiz_sum: {}", old_sum);
        println!("New horiz_sum: {}", new_sum);
        println!("Expected:      {}", expected);
        println!("Old correct? {}", (old_sum - expected).abs() < 1e-10);
        println!("New correct? {}", (new_sum - expected).abs() < 1e-10);
    }
}