pub fn drop_in_background_thread<T>(data: T)
where
    T: Send + 'static,
{
    // h/t https://abrams.cc/rust-dropping-things-in-another-thread
    rayon::spawn(move || drop(data));
}

/// A vector of `size` zeros, allocated zeroed in one call instead of written element by element.
///
/// Sound for types whose zero value is the all-zero bit pattern (field elements in Montgomery or
/// canonical form, integers); zero-sized types and empty vectors take the safe path.
#[expect(
    clippy::unwrap_used,
    reason = "Layout::array only fails on overflow, which callers' size arithmetic already rules out"
)]
pub fn unsafe_allocate_zero_vec<T: Sized + num_traits::Zero>(size: usize) -> Vec<T> {
    if size == 0 || std::mem::size_of::<T>() == 0 {
        return std::iter::repeat_with(T::zero).take(size).collect();
    }

    unsafe {
        let layout = std::alloc::Layout::array::<T>(size).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout).cast::<T>();
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        #[expect(clippy::same_length_and_capacity)]
        Vec::from_raw_parts(ptr, size, size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::fp128::Fp128 as Fr;
    use crate::field::Ring;
    use num_traits::Zero;

    #[test]
    fn zero_vec_field_elements() {
        let table: Vec<Fr> = unsafe_allocate_zero_vec(1 << 10);
        assert_eq!(table, vec![Fr::from_u64(0); 1 << 10]);
        assert!(unsafe_allocate_zero_vec::<Fr>(0).is_empty());
    }

    #[test]
    fn zero_vec_zero_sized_type() {
        #[derive(Clone, Copy, Debug, PartialEq)]
        struct Zst;
        impl std::ops::Add for Zst {
            type Output = Self;
            fn add(self, _: Self) -> Self {
                Self
            }
        }
        impl Zero for Zst {
            fn zero() -> Self {
                Self
            }
            fn is_zero(&self) -> bool {
                true
            }
        }
        let v: Vec<Zst> = unsafe_allocate_zero_vec(8);
        assert_eq!(v.len(), 8);
    }

    #[test]
    fn zero_vec_padded_type() {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq)]
        struct PaddedZero {
            byte: u8,
            word: u32,
        }
        impl std::ops::Add for PaddedZero {
            type Output = Self;
            fn add(self, _: Self) -> Self {
                self
            }
        }
        impl Zero for PaddedZero {
            fn zero() -> Self {
                Self { byte: 0, word: 0 }
            }
            fn is_zero(&self) -> bool {
                self.byte == 0 && self.word == 0
            }
        }
        let values: Vec<PaddedZero> = unsafe_allocate_zero_vec(4);
        assert!(values.iter().all(Zero::is_zero));
    }
}
