use crate::node_view::ComputationNode;
use crate::onnx_proof::op_lookups::{DefaultLookupOperands, LookupWitness};
use crate::utils::compute_lookup_indices_from_operands;
use atlas_algebra::utils::lookup_bits::LookupBits;
use atlas_trace::op::Clamp;
use atlas_trace::{Tensor, Trace};
pub use jolt_atlas_verifier::onnx_proof::ops::clamp::*;

crate::impl_plan_op_prover!(Clamp);

impl LookupWitness for SymmetricClampOperands {
    fn witness(&self, node: &ComputationNode, trace: &Trace) -> Tensor<i64> {
        LookupWitness::witness(&DefaultLookupOperands, node, trace)
    }

    fn lookup_bits(&self, witness: &Tensor<i64>) -> Vec<LookupBits> {
        let operand = witness.map(|v| v as i32);
        compute_lookup_indices_from_operands(&[&operand], false)
    }
}
