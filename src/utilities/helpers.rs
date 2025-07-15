use crate::utilities::enums::Kernel;
use std::arch::is_x86_feature_detected;
use std::sync::OnceLock;
use std::{mem::MaybeUninit, ptr, slice};
use aligned_vec::{AVec};

static BEST_SINGLE: OnceLock<Kernel> = OnceLock::new();
static BEST_BATCH: OnceLock<Kernel> = OnceLock::new();

#[inline(always)]
pub fn detect_best_kernel() -> Kernel {
    *BEST_SINGLE.get_or_init(|| {
        
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("fma") {
                return Kernel::Avx512;
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return Kernel::Avx2;
            }
        }
        
        Kernel::Scalar
    })
}

#[inline(always)]
pub fn detect_best_batch_kernel() -> Kernel {
    *BEST_BATCH.get_or_init(|| {
        
        match detect_best_kernel() {
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx512 => Kernel::Avx512Batch,
            #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
            Kernel::Avx2 => Kernel::Avx2Batch,
            _ => Kernel::ScalarBatch,
        }
    })
}

#[macro_export]
macro_rules! skip_if_unsupported {
    ($kernel:expr, $test_name:expr) => {{
        use $crate::utilities::enums::Kernel;
        use std::arch::is_x86_feature_detected;

        

        #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
        {
            if matches!(
                $kernel,
                Kernel::Avx2
                    | Kernel::Avx2Batch
                    | Kernel::Avx512
                    | Kernel::Avx512Batch
            ) {
                eprintln!(
                    "[{}] skipped {:?} – compiled without `nightly-avx`",
                    $test_name, $kernel
                );
                return Ok(());
            }
        }

        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        {
            let need: (&'static str, fn() -> bool) = match $kernel {
                Kernel::Avx512 | Kernel::Avx512Batch => (
                    "AVX-512F + FMA",
                    || is_x86_feature_detected!("avx512f")
                    && is_x86_feature_detected!("fma"),
                ),
                Kernel::Avx2 | Kernel::Avx2Batch => (
                    "AVX2 + FMA",
                    || is_x86_feature_detected!("avx2")
                    && is_x86_feature_detected!("fma"),
                ),
                _ => ("", || true),
            };

            if !(need.1)() {
                eprintln!(
                    "[{}] skipped {:?} - CPU lacks {}",
                    $test_name, $kernel, need.0
                );
                return Ok(());
            }
        }
    }};
}
#[inline(always)]
pub fn alloc_with_nan_prefix(len: usize, warm: usize) -> Vec<f64> {
    use std::mem::{self, MaybeUninit};

    let warm = warm.min(len);

    // Allocate with capacity
    let mut buf: Vec<MaybeUninit<f64>> = Vec::with_capacity(len);
    
    // In release mode, just set length after filling warmup
    #[cfg(not(debug_assertions))]
    {
        unsafe { buf.set_len(len); }
        for i in 0..warm {
            buf[i].write(f64::from_bits(0x7ff8_0000_0000_0000));
        }
    }
    
    // In debug mode, initialize all values to avoid any UB window
    #[cfg(debug_assertions)]
    {
        // Extend to full length with poison values
        for _ in 0..warm {
            buf.push(MaybeUninit::new(f64::from_bits(0x7ff8_0000_0000_0000)));
        }
        for _ in warm..len {
            buf.push(MaybeUninit::new(f64::from_bits(0x11111111_11111111)));
        }
    }

    // Convert to Vec<f64>
    let ptr = buf.as_mut_ptr() as *mut f64;
    let cap = buf.capacity();
    mem::forget(buf);
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}



#[inline]
pub fn init_matrix_prefixes(
    buf: &mut [MaybeUninit<f64>],
    cols: usize,
    warm_prefixes: &[usize],
) {
    assert!(cols != 0 && buf.len() % cols == 0,
            "`buf` length must be a multiple of `cols`");
    let rows = buf.len() / cols;
    assert_eq!(rows, warm_prefixes.len(),
        "`warm_prefixes` length must equal number of rows");

    // DEBUG ONLY: First poison ALL cells to catch uninitialized reads
    #[cfg(debug_assertions)]
    {
        // Use different poison value 0x22222222_22222222
        // This appears as 3.5873e-221 when interpreted as f64
        for cell in buf.iter_mut() {
            cell.write(f64::from_bits(0x22222222_22222222));
        }
    }

    // Write NaN values to warm prefixes
    buf.chunks_exact_mut(cols)
        .zip(warm_prefixes)
        .for_each(|(row, &warm)| {
            assert!(warm <= cols, "warm prefix exceeds row width");
            for cell in &mut row[..warm] {
                cell.write(f64::from_bits(0x7ff8_0000_0000_0000));
            }
        });
}

/// Allocate `rows × cols` uninitialised `f64` without UB or silent overflow.
#[inline]
pub fn make_uninit_matrix(rows: usize, cols: usize)
    -> Vec<MaybeUninit<f64>>
{
    let total = rows.checked_mul(cols)
        .expect("rows * cols overflowed usize");

    // try_reserve_exact lets us bail out gracefully instead of aborting/BSOD
    let mut v: Vec<MaybeUninit<f64>> = Vec::new();
    v.try_reserve_exact(total)
        .expect("OOM in make_uninit_matrix");

    // In release mode, just set length - MaybeUninit is safe for uninitialized memory
    #[cfg(not(debug_assertions))]
    {
        unsafe { v.set_len(total); }
    }
    
    // DEBUG ONLY: poison all cells with a third distinct pattern
    #[cfg(debug_assertions)]
    {
        // Push poison values to avoid UB window
        for _ in 0..total {
            v.push(MaybeUninit::new(f64::from_bits(0x33333333_33333333)));
        }
    }
    v
}