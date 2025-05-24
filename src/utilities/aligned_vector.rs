use std::{
    alloc::{alloc, alloc_zeroed, dealloc, Layout},
    marker::PhantomData,
    mem::{align_of, size_of, MaybeUninit},
    ops::{Deref, DerefMut, Index, IndexMut},
    ptr::{self, NonNull},
    slice,
};

#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alignment {
    Sse = 16,
    Avx = 32,
    Avx512 = 64,
    Custom(usize),
}

impl Alignment {
    #[inline]
    pub const fn as_bytes(&self) -> usize {
        match self {
            Alignment::Sse => 16,
            Alignment::Avx => 32,
            Alignment::Avx512 => 64,
            Alignment::Custom(n) => *n,
        }
    }

    #[inline]
    pub const fn is_valid(&self) -> bool {
        let bytes = self.as_bytes();
        bytes > 0 && (bytes & (bytes - 1)) == 0
    }
}

pub struct AlignedVec<T = f64> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
    align: Alignment,
    _phantom: PhantomData<T>,
}

impl<T> AlignedVec<T> {
    #[inline]
    pub fn with_capacity_aligned(cap: usize, align: Alignment) -> Self {
        assert!(cap > 0, "Capacity must be greater than 0");
        assert!(align.is_valid(), "Alignment must be a power of 2");
        assert!(
            align.as_bytes() >= align_of::<T>(),
            "Alignment must be at least {}",
            align_of::<T>()
        );

        let layout = Layout::from_size_align(cap * size_of::<T>(), align.as_bytes())
            .expect("Invalid layout");

        let raw = unsafe { alloc(layout) } as *mut T;
        if raw.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        Self {
            ptr: unsafe { NonNull::new_unchecked(raw) },
            len: 0,
            cap,
            align,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self::with_capacity_aligned(cap, Alignment::Avx512)
    }

    #[inline]
    pub fn zeroed(cap: usize) -> Self {
        Self::zeroed_aligned(cap, Alignment::Avx512)
    }

    pub fn zeroed_aligned(cap: usize, align: Alignment) -> Self {
        assert!(cap > 0, "Capacity must be greater than 0");
        assert!(align.is_valid(), "Alignment must be a power of 2");

        let layout = Layout::from_size_align(cap * size_of::<T>(), align.as_bytes())
            .expect("Invalid layout");

        let raw = unsafe { alloc_zeroed(layout) } as *mut T;
        if raw.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        Self {
            ptr: unsafe { NonNull::new_unchecked(raw) },
            len: cap,
            cap,
            align,
            _phantom: PhantomData,
        }
    }

    pub fn from_slice(slice: &[T]) -> Self
    where
        T: Copy,
    {
        Self::from_slice_aligned(slice, Alignment::Avx512)
    }

    pub fn from_slice_aligned(slice: &[T], align: Alignment) -> Self
    where
        T: Copy,
    {
        let mut vec = Self::with_capacity_aligned(slice.len(), align);
        vec.extend_from_slice(slice);
        vec
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    #[inline]
    pub fn alignment(&self) -> Alignment {
        self.align
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn is_aligned_to(&self, align: usize) -> bool {
        is_aligned(self.ptr.as_ptr(), align)
    }

    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn push(&mut self, value: T) {
        assert!(self.len < self.cap, "Capacity exceeded");
        unsafe {
            self.ptr.as_ptr().add(self.len).write(value);
        }
        self.len += 1;
    }

    pub fn extend_from_slice(&mut self, slice: &[T])
    where
        T: Copy,
    {
        let new_len = self.len + slice.len();
        assert!(new_len <= self.cap, "Capacity exceeded");

        unsafe {
            ptr::copy_nonoverlapping(slice.as_ptr(), self.ptr.as_ptr().add(self.len), slice.len());
        }
        self.len = new_len;
    }

    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        assert!(new_len <= self.cap, "Cannot resize beyond capacity");

        if new_len > self.len {
            for i in self.len..new_len {
                unsafe {
                    self.ptr.as_ptr().add(i).write(value.clone());
                }
            }
        }
        self.len = new_len;
    }

    #[inline]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.cap);
        self.len = new_len;
    }

    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.as_slice().to_vec()
    }
}

impl AlignedVec<f64> {
    pub fn filled(cap: usize, value: f64) -> Self {
        Self::filled_aligned(cap, value, Alignment::Avx512)
    }

    pub fn filled_aligned(cap: usize, value: f64, align: Alignment) -> Self {
        let mut vec = Self::with_capacity_aligned(cap, align);
        vec.resize(cap, value);
        vec
    }

    pub fn nan(cap: usize) -> Self {
        Self::filled(cap, f64::NAN)
    }

    pub fn fill_range(&mut self, range: std::ops::Range<usize>, value: f64) {
        assert!(range.end <= self.len);
        for i in range {
            self[i] = value;
        }
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if self.cap > 0 {
            for i in 0..self.len {
                unsafe {
                    ptr::drop_in_place(self.ptr.as_ptr().add(i));
                }
            }

            let layout = Layout::from_size_align(self.cap * size_of::<T>(), self.align.as_bytes())
                .expect("Invalid layout");
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

impl<T> Index<usize> for AlignedVec<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<T> IndexMut<usize> for AlignedVec<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}

impl<T> Deref for AlignedVec<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for AlignedVec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T> AsRef<[T]> for AlignedVec<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsMut<[T]> for AlignedVec<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> IntoIterator for AlignedVec<T> {
    type Item = T;
    type IntoIter = AlignedVecIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        let ptr = self.ptr;
        let len = self.len;
        let cap = self.cap;
        let align = self.align;

        std::mem::forget(self);

        AlignedVecIntoIter {
            ptr,
            len,
            cap,
            align,
            pos: 0,
            _phantom: PhantomData,
        }
    }
}

pub struct AlignedVecIntoIter<T> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
    align: Alignment,
    pos: usize,
    _phantom: PhantomData<T>,
}

impl<T> Iterator for AlignedVecIntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.len {
            let value = unsafe { self.ptr.as_ptr().add(self.pos).read() };
            self.pos += 1;
            Some(value)
        } else {
            None
        }
    }
}

impl<T> Drop for AlignedVecIntoIter<T> {
    fn drop(&mut self) {
        for i in self.pos..self.len {
            unsafe {
                ptr::drop_in_place(self.ptr.as_ptr().add(i));
            }
        }

        if self.cap > 0 {
            let layout = Layout::from_size_align(self.cap * size_of::<T>(), self.align.as_bytes())
                .expect("Invalid layout");
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

impl<T: Copy> Clone for AlignedVec<T> {
    fn clone(&self) -> Self {
        let mut new_vec = Self::with_capacity_aligned(self.cap, self.align);
        new_vec.extend_from_slice(self.as_slice());
        new_vec
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for AlignedVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedVec")
            .field("len", &self.len)
            .field("cap", &self.cap)
            .field("align", &self.align.as_bytes())
            .field("data", &self.as_slice())
            .finish()
    }
}

#[inline]
pub fn is_aligned<T>(ptr: *const T, align: usize) -> bool {
    debug_assert!(align.is_power_of_two(), "Alignment must be power of 2");
    (ptr as usize) & (align - 1) == 0
}

#[inline]
pub fn is_avx2_aligned<T>(slice: &[T]) -> bool {
    is_aligned(slice.as_ptr(), 32)
}

#[inline]
pub fn is_avx512_aligned<T>(slice: &[T]) -> bool {
    is_aligned(slice.as_ptr(), 64)
}

#[inline]
pub const fn align_up(size: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (size + align - 1) & !(align - 1)
}

#[inline]
pub const fn align_down(size: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    size & !(align - 1)
}

#[inline]
pub const fn padding_for(size: usize, align: usize) -> usize {
    align_up(size, align) - size
}

pub fn copy_to_aligned<T: Copy>(data: &[T]) -> AlignedVec<T> {
    AlignedVec::from_slice(data)
}

pub fn copy_to_aligned_with<T: Copy>(data: &[T], align: Alignment) -> AlignedVec<T> {
    AlignedVec::from_slice_aligned(data, align)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment() {
        let vec = AlignedVec::<f64>::with_capacity(10);
        assert!(is_avx512_aligned(vec.as_slice()));
        assert!(vec.is_aligned_to(64));

        let vec_avx = AlignedVec::<f64>::with_capacity_aligned(10, Alignment::Avx);
        assert!(is_avx2_aligned(vec_avx.as_slice()));
        assert!(vec_avx.is_aligned_to(32));
    }

    #[test]
    fn test_operations() {
        let mut vec = AlignedVec::zeroed(10);
        assert_eq!(vec.len(), 10);
        assert!(vec.iter().all(|&x| x == 0.0));

        vec[0] = 1.0;
        vec.push(2.0);
        assert_eq!(vec.len(), 11);

        let slice = &[3.0, 4.0, 5.0];
        vec.extend_from_slice(slice);
        assert_eq!(vec.len(), 14);
    }

    #[test]
    fn test_utility_functions() {
        assert_eq!(align_up(10, 8), 16);
        assert_eq!(align_up(16, 8), 16);
        assert_eq!(align_down(10, 8), 8);
        assert_eq!(padding_for(10, 8), 6);
    }
}
