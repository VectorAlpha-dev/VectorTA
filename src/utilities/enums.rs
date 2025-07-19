#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Kernel {
	Auto,
	Scalar,
	Avx2,
	Avx512,
	ScalarBatch,
	Avx2Batch,
	Avx512Batch,
}

impl Default for Kernel {
	fn default() -> Self {
		Kernel::Auto
	}
}

impl Kernel {
	#[inline(always)]
	pub const fn is_batch(self) -> bool {
		matches!(self, Kernel::ScalarBatch | Kernel::Avx2Batch | Kernel::Avx512Batch)
	}
}
