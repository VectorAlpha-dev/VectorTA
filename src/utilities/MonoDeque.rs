#[derive(Clone)]
pub struct MonoDeque {
    pub buf:  Vec<usize>,
    pub head: usize,
    pub tail: usize,
    pub mask: usize,
}

impl MonoDeque {
    pub fn with_capacity(cap: usize) -> Self {
        let size = cap.next_power_of_two();
        Self { buf: vec![0; size], head: 0, tail: 0, mask: size - 1 }
    }
    #[inline(always)]
    pub fn is_empty(&self) -> bool { self.head == self.tail }
    #[inline(always)]
    pub fn len(&self) -> usize { self.tail.wrapping_sub(self.head) & self.mask }
    #[inline(always)]
    pub fn front(&self) -> usize { self.buf[self.head] }
    #[inline(always)]
    pub fn pop_front(&mut self) { self.head = (self.head + 1) & self.mask; }
    #[inline(always)]
    pub fn pop_back(&mut self)  { self.tail = (self.tail.wrapping_sub(1)) & self.mask; }

    #[inline(always)]
    pub fn push_max(&mut self, idx: usize, src: &[f64]) {
        while !self.is_empty() && src[idx] >= src[self.buf[(self.tail.wrapping_sub(1)) & self.mask]] {
            self.pop_back();
        }
        self.buf[self.tail] = idx;
        self.tail = (self.tail + 1) & self.mask;
    }
    #[inline(always)]
    pub fn push_min(&mut self, idx: usize, src: &[f64]) {
        while !self.is_empty() && src[idx] <= src[self.buf[(self.tail.wrapping_sub(1)) & self.mask]] {
            self.pop_back();
        }
        self.buf[self.tail] = idx;
        self.tail = (self.tail + 1) & self.mask;
    }
    #[inline(always)]
    pub fn expire(&mut self, oldest_allowed: usize) {
        while !self.is_empty() && self.front() < oldest_allowed {
            self.pop_front();
        }
    }
}