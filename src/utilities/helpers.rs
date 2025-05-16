use std::arch::is_x86_feature_detected;
use std::sync::OnceLock;
use crate::utilities::enums::Kernel;

static BEST_SINGLE : OnceLock<Kernel> = OnceLock::new();
static BEST_BATCH  : OnceLock<Kernel> = OnceLock::new();

#[inline(always)]
pub fn detect_best_kernel() -> Kernel {
    *BEST_SINGLE.get_or_init(|| {
        if cfg!(target_arch = "x86_64") {
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
    *BEST_BATCH.get_or_init(|| match detect_best_kernel() {
        Kernel::Avx512 => Kernel::Avx512Batch,
        Kernel::Avx2   => Kernel::Avx2Batch,
        Kernel::Scalar => Kernel::ScalarBatch,
        _              => Kernel::ScalarBatch,
    })
}


#[inline]
pub fn skip_if_unsupported(kernel: Kernel, test_name: &str) {
    use std::arch::is_x86_feature_detected;

    let (need_avx2, need_avx512) = match kernel {
        Kernel::Avx2 | Kernel::Avx2Batch     => (true,  false),
        Kernel::Avx512 | Kernel::Avx512Batch => (false, true),
        _                                    => (false, false),
    };

    let fma_ok = is_x86_feature_detected!("fma");

    if need_avx512 {
        if !(is_x86_feature_detected!("avx512f") && fma_ok) {
            eprintln!(
                "[{}] Skipping {:?} test – CPU lacks AVX-512F+FMA",
                test_name, kernel
            );
            std::process::exit(0);
        }
    } else if need_avx2 {
        if !(is_x86_feature_detected!("avx2") && fma_ok) {
            eprintln!(
                "[{}] Skipping {:?} test – CPU lacks AVX2+FMA",
                test_name, kernel
            );
            std::process::exit(0);
        }
    }
}
