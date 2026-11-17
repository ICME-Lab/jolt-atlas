//! The prover's side of Sin and Cos ([`jolt_atlas_verifier::onnx_proof::ops::trig`]): the
//! period reduction's claim, the downscaled remainder and the downscale lookup's witness.

pub use jolt_atlas_verifier::onnx_proof::ops::trig::*;

use crate::node_view::ComputationNode;
use crate::onnx_proof::op_lookups::LookupWitness;
use crate::onnx_proof::trace_view::TraceView;
use crate::utils::opening_access::AccOpeningAccessor;
use atlas_algebra::{field::JoltField, utils::lookup_bits::LookupBits};
use atlas_sumcheck::opening_proof::OpeningAccumulator;
use atlas_trace::{Tensor, Trace};
use common::{
    consts::{TRIG_DOWNSCALE_BITS, TRIG_PERIOD_MODULUS, XLEN},
    CommittedPoly, VirtualPoly,
};

/// The claim of a [`trig_period`]: the input at the reduced output point is `P·q + rem` over
/// the quotient and remainder claims held there.
pub fn trig_period_claims<F: JoltField>(
    node: &ComputationNode,
    accumulator: &dyn OpeningAccumulator<F>,
) -> Vec<F> {
    let expect = "the prover holds every claim it reads";
    let accessor = AccOpeningAccessor::new(accumulator, node);
    let quotient = accessor
        .get_advice(CommittedPoly::TrigPeriodQuotient)
        .expect(expect)
        .1;
    let remainder = accessor
        .get_advice(VirtualPoly::TrigPeriodRemainder)
        .expect(expect)
        .1;
    vec![F::from_i32(TRIG_PERIOD_MODULUS as i32) * quotient + remainder]
}

/// A trig node's downscaled period remainder, `(input mod P) >> TRIG_DOWNSCALE_BITS`: the
/// addresses its table is read at.
pub fn downscaled(node: &ComputationNode, trace: &Trace) -> Tensor<i32> {
    TraceView::trig_period_advice(trace, node)
        .remainder
        .map(|v| v >> TRIG_DOWNSCALE_BITS)
}

impl LookupWitness for TrigDownscaleOperands {
    /// The period remainder the trace captured.
    fn witness(&self, node: &ComputationNode, trace: &Trace) -> Tensor<i64> {
        TraceView::trig_period_advice(trace, node)
            .remainder
            .map(|v| v as i64)
    }

    fn lookup_bits(&self, witness: &Tensor<i64>) -> Vec<LookupBits> {
        witness
            .iter()
            .map(|&v| LookupBits::new(v as u64, XLEN))
            .collect()
    }
}
