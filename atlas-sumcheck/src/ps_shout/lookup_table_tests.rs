//! Read-raf sumchecks over the lookup tables of `atlas-lookup-tables`.

use super::unary::tests::test_read_raf_sumcheck as unary_read_raf;
use atlas_lookup_tables::{
    clamp::{ActivationClampTable, ClampTable, SaturationTable, SoftmaxClampTable, LARGE_XLEN},
    relu::ReluTable,
    right_shift::RightShiftTable,
    unsigned_abs::UnsignedAbsTable,
};
use common::consts::XLEN;

#[test]
fn clamp_read_raf() {
    unary_read_raf::<ClampTable<XLEN>, XLEN>();
    unary_read_raf::<SaturationTable, LARGE_XLEN>();
    unary_read_raf::<ActivationClampTable<XLEN>, XLEN>();
    unary_read_raf::<SoftmaxClampTable<XLEN>, XLEN>();
}

#[test]
fn relu_read_raf() {
    unary_read_raf::<ReluTable<32>, 32>();
}

#[test]
fn right_shift_read_raf() {
    unary_read_raf::<RightShiftTable<32>, 32>();
}

#[test]
fn unsigned_abs_read_raf() {
    unary_read_raf::<UnsignedAbsTable<32>, 32>();
}
