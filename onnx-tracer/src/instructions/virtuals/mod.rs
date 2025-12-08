#![allow(non_camel_case_types)]
mod virtual_advice;
mod virtual_assert_eq;
mod virtual_assert_valid_div0;
mod virtual_assert_valid_signed_remainder;
mod virtual_move;
mod virtual_pow2;
mod virtual_saturating_sum;
mod virtual_shift_right_bitmask;
mod virtual_sra;

pub use {
    virtual_advice::VirtualAdvice, virtual_assert_eq::VirtualAssertEq,
    virtual_assert_valid_div0::VirtualAssertValidDiv0,
    virtual_assert_valid_signed_remainder::VirtualAssertValidSignedRemainder,
    virtual_move::VirtualMove, virtual_pow2::VirtualPow2,
    virtual_saturating_sum::VirtualSaturatingSum,
    virtual_shift_right_bitmask::VirtualShiftRightBitmask, virtual_sra::VirtualSra,
};
