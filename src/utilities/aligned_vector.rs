use std::{
	alloc::{alloc, dealloc, Layout},
	ptr::NonNull,
};

pub struct AlignedVec {
	ptr: NonNull<f64>,
	len: usize,
	cap: usize,
}

impl AlignedVec {
	#[inline]
	pub fn with_capacity(cap: usize) -> Self {
		assert!(cap > 0);
		let layout = Layout::from_size_align(cap * 8, 64).expect("layout");
		let raw = unsafe { alloc(layout) } as *mut f64;
		if raw.is_null() {
			std::alloc::handle_alloc_error(layout);
		}
		Self {
			ptr: unsafe { NonNull::new_unchecked(raw) },
			len: cap,
			cap,
		}
	}

	#[inline]
	pub fn as_slice(&self) -> &[f64] {
		unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
	}

	#[inline]
	pub fn as_mut_slice(&mut self) -> &mut [f64] {
		unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
	}

	#[inline]
	pub fn as_ptr(&self) -> *const f64 {
		self.ptr.as_ptr()
	}
}

impl Drop for AlignedVec {
	fn drop(&mut self) {
		let layout = Layout::from_size_align(self.cap * 8, 64).unwrap();
		unsafe { dealloc(self.ptr.as_ptr() as *mut u8, layout) };
	}
}

impl AsRef<[f64]> for AlignedVec {
	#[inline]
	fn as_ref(&self) -> &[f64] {
		self.as_slice()
	}
}
