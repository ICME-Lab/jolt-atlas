pub trait Math {
    fn pow2(self) -> usize;
    fn log_2(self) -> usize;
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
}
