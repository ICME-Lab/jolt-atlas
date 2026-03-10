use ark_ff::biginteger::S64;

pub trait Math {
    fn pow2(self) -> usize;
    fn log_2(self) -> usize;
    fn ceil_log_2(self) -> usize;
}

impl Math for usize {
    #[inline]
    fn pow2(self) -> usize {
        let base: usize = 2;
        base.pow(self as u32)
    }

    fn log_2(self) -> usize {
        assert_ne!(self, 0);
        assert!(
            self.is_power_of_two(),
            "log_2 is only defined for power-of-two values, got {self}"
        );
        (1usize.leading_zeros() - self.leading_zeros()) as usize
    }

    fn ceil_log_2(self) -> usize {
        assert_ne!(self, 0);
        self.next_power_of_two().log_2()
    }
}

#[inline(always)]
pub fn s64_from_diff_u64s(a: u64, b: u64) -> S64 {
    if a < b {
        S64::new([b - a], false)
    } else {
        S64::new([a - b], true)
    }
}
