use crate::utilities::enums::Kernel;
use std::arch::is_x86_feature_detected;
use std::sync::OnceLock;
use std::{mem::MaybeUninit, ptr, slice};
use aligned_vec::{AVec};

static BEST_SINGLE: OnceLock<Kernel> = OnceLock::new();
static BEST_BATCH: OnceLock<Kernel> = OnceLock::new();

#[inline(always)]
pub fn detect_best_kernel() -> Kernel {
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    {
        *BEST_SINGLE.get_or_init(|| {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("fma") {
                return Kernel::Avx512;
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return Kernel::Avx2;
            }
            Kernel::Scalar
        })
    }

    #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
    {
        Kernel::Scalar
    }
}

#[inline(always)]
pub fn detect_best_batch_kernel() -> Kernel {
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    {
        *BEST_BATCH.get_or_init(|| match detect_best_kernel() {
            Kernel::Avx512 => Kernel::Avx512Batch,
            Kernel::Avx2 => Kernel::Avx2Batch,
            _ => Kernel::ScalarBatch,
        })
    }

    #[cfg(not(all(feature = "nightly-avx", target_arch = "x86_64")))]
    {
        Kernel::ScalarBatch
    }
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
    use std::{mem::MaybeUninit, ptr};
    let mut v: Vec<MaybeUninit<f64>> = Vec::with_capacity(len);
    unsafe { v.set_len(len); }
    let prefix_bytes = (warm * std::mem::size_of::<f64>()) as isize;
    let dst = v.as_mut_ptr().cast::<u8>();
    unsafe { ptr::write_bytes(dst, 0xFF, prefix_bytes as usize); }
    let p   = v.as_mut_ptr() as *mut f64;
    let cap = v.capacity();
    std::mem::forget(v);
    unsafe { Vec::from_raw_parts(p, len, cap) }
}


#[inline]
pub fn init_matrix_prefixes(
    buf: &mut [MaybeUninit<f64>],
    cols: usize,
    warm_prefixes: &[usize],
) {
    assert!(
        cols != 0 && buf.len() % cols == 0,
        "`buf` length must be a multiple of `cols`"
    );
    let rows = buf.len() / cols;
    assert_eq!(
        rows,
        warm_prefixes.len(),
        "`warm_prefixes` length must equal number of rows"
    );

    for (row_idx, &warm) in warm_prefixes.iter().enumerate() {
        debug_assert!(
            warm <= cols,
            "prefix length ({warm}) exceeds number of columns ({cols})"
        );

        let start = row_idx * cols;
        for cell in &mut buf[start..start + warm] {
            cell.write(f64::NAN);
        }
    }
}

/// ---------------------------------------------------------------------------
/// 3.  Allocate `rows × cols` uninitialised elements with overflow checking.
/// ---------------------------------------------------------------------------
#[inline]
pub fn make_uninit_matrix(rows: usize, cols: usize) -> Vec<MaybeUninit<f64>> {
    let total = rows
        .checked_mul(cols)
        .expect("rows * cols overflowed usize");
    let mut v = Vec::<MaybeUninit<f64>>::with_capacity(total);
    unsafe { v.set_len(total) };
    v
}