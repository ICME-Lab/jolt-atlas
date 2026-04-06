//! Runtime-sized sparse row representation

/// Runtime-sized sparse row for binary MSM
///
/// Uses u16 indices when n ≤ 65536, otherwise u32.
#[derive(Clone, Debug)]
pub struct SmallRow {
    idx_u16: Option<Box<[u16]>>,
    idx_u32: Option<Box<[u32]>>,
}

impl SmallRow {
    #[inline]
    pub fn from_u16(mut v: Vec<u16>) -> Self {
        v.shrink_to_fit();
        Self {
            idx_u16: Some(v.into_boxed_slice()),
            idx_u32: None,
        }
    }

    #[inline]
    pub fn from_u32(mut v: Vec<u32>) -> Self {
        v.shrink_to_fit();
        Self {
            idx_u16: None,
            idx_u32: Some(v.into_boxed_slice()),
        }
    }

    #[inline(always)]
    pub fn as_u16_slice(&self) -> &[u16] {
        self.idx_u16.as_deref().expect("expected u16 storage")
    }

    #[inline(always)]
    pub fn as_u32_slice(&self) -> &[u32] {
        self.idx_u32.as_deref().expect("expected u32 storage")
    }

    #[inline(always)]
    pub fn is_u16(&self) -> bool {
        self.idx_u16.is_some()
    }

    #[inline]
    pub fn iter_usize(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        if let Some(s) = &self.idx_u16 {
            return SmallRowIter::U16(s.iter());
        }
        let s = self.idx_u32.as_ref().expect("row not initialized");
        SmallRowIter::U32(s.iter())
    }

    #[inline]
    pub fn len(&self) -> usize {
        if let Some(s) = &self.idx_u16 {
            return s.len();
        }
        self.idx_u32.as_ref().expect("row not initialized").len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub enum SmallRowIter<'a> {
    U16(core::slice::Iter<'a, u16>),
    U32(core::slice::Iter<'a, u32>),
}

impl<'a> Iterator for SmallRowIter<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            SmallRowIter::U16(iter) => iter.next().map(|&x| x as usize),
            SmallRowIter::U32(iter) => iter.next().map(|&x| x as usize),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            SmallRowIter::U16(iter) => iter.size_hint(),
            SmallRowIter::U32(iter) => iter.size_hint(),
        }
    }
}

impl<'a> ExactSizeIterator for SmallRowIter<'a> {
    #[inline]
    fn len(&self) -> usize {
        match self {
            SmallRowIter::U16(iter) => iter.len(),
            SmallRowIter::U32(iter) => iter.len(),
        }
    }
}
