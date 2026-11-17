//! The smooth ReLUs: GELU in its erf and tanh forms, and SiLU.
//!
//! Each is `f(x) = x·g(x)` with `f(x) − f(−x) = x`, so `f(x) − ReLU(x)` is even, and it
//! vanishes outside a bounded range. With `H = f − ReLU` tabulated over the clamped input:
//!
//! ```text
//! out = ReLU(x) + H[clamp(x)]
//! ```
//!
//! The proof is the clamped-activation protocol ([`super::activation_clamped`]) with the
//! small table holding `H` and reading `out − ReLU(x)`, plus one more read of the clamp's
//! 32-bit address: `ReLU(x)` from the ReLU table. The clamp and the ReLU reads share the
//! committed address chunks, so they see one `x`.
//!
//! **S ⊆ R.** The one-hot checks make the 32-bit address an integer `x` and the table address
//! an integer; the clamp read gives `clamped = Clamp[x]` and the ReLU read `relu = ReLU[x]` for
//! that same `x`; both reads' operand claim ties `x` to the node's operand at the reduced
//! output point; the small-table read gives `out − relu = H[clamped]` there. So
//! `out = ReLU(x) + H[clamp(x)]` on every element.

use crate::node_view::ComputationNode;
use crate::onnx_proof::lookup::{small_table, SmallTableDef};
use crate::onnx_proof::ops::OpLayout;
use crate::onnx_proof::relation::{OpExpr, RelationOpening};
use crate::onnx_proof::{
    deferred_lookups::DeferredBatch,
    op_lookups::{DefaultLookupOperands, LookupOperandsTrait, LookupRaPolys},
    ops::activation_clamped::{clamped_activation_committed_polynomials, ActivationClampOperands},
    plan::{OpLookup, OpOpen, OpPlan, OpRelation, Plan, Step},
};
use crate::utils::opening_access::OpeningIdBuilder;
use atlas_algebra::{
    field::JoltField,
    poly::point::{OpeningPoint, BIG_ENDIAN},
    utils::errors::ProofVerifyError,
};
use atlas_claims::opening;
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_trace::{
    op::{SmoothRelu, SmoothReluForm},
    semantics::tables::TableKind,
    trace::ModelExecutionIO,
    Model,
};
use common::{
    consts::{ACTIVATION_TABLE_VARS, MODEL_SCALE, XLEN},
    CommittedPoly, VirtualPoly,
};

/// The address variables of `form`'s correction table.
pub fn table_vars(form: SmoothReluForm) -> usize {
    match form {
        SmoothReluForm::GeluErf | SmoothReluForm::GeluTanh => ACTIVATION_TABLE_VARS,
        SmoothReluForm::Silu => ACTIVATION_TABLE_VARS + 1,
    }
}

/// The opening of `ReLU(x)` at the node's reduced output point.
pub fn relu_opening_id(node: &ComputationNode) -> OpeningId {
    OpeningIdBuilder::new(node).advice(VirtualPoly::SmoothReluRelu)
}

/// `form`'s correction table `H = f − ReLU`, read at the clamped input, whose opening the
/// read's setup appends; the read's value is the output less `ReLU(x)`.
pub fn table_def(form: SmoothReluForm) -> SmallTableDef {
    SmallTableDef {
        kind: match form {
            SmoothReluForm::GeluErf => TableKind::GeluErf,
            SmoothReluForm::GeluTanh => TableKind::GeluTanh,
            SmoothReluForm::Silu => TableKind::Silu,
        },
        vars: table_vars(form),
        signed: true,
        ra: VirtualPoly::ActivationSmallRa,
        ra_chunk: CommittedPoly::ActivationSmallRaD,
        address: VirtualPoly::ActivationClampedOutput,
        opens_address: true,
        value: output_less_relu,
    }
}

/// The value a correction table's read equals: the node's reduced output less `ReLU(x)`.
fn output_less_relu(node: &ComputationNode) -> OpExpr {
    opening(RelationOpening::ReducedNodeOutput(node.idx))
        - opening(RelationOpening::Id(relu_opening_id(node)))
}

/// Lookup operands of a smooth ReLU's `ReLU` read: the clamp lookup's operand and committed
/// address, with `ReLU(x)` as the value read.
#[derive(Default)]
pub struct SmoothReluOperands;

impl LookupOperandsTrait for SmoothReluOperands {
    const LOG_K: usize = XLEN;

    fn rv_claim<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Result<F, ProofVerifyError> {
        Ok(accumulator
            .get_virtual_polynomial_opening(relu_opening_id(node))?
            .1)
    }

    fn witness_opening_id(node: &ComputationNode) -> OpeningId {
        DefaultLookupOperands::witness_opening_id(node)
    }

    fn r_cycle<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Result<OpeningPoint<BIG_ENDIAN, F>, ProofVerifyError> {
        DefaultLookupOperands::r_cycle(node, accumulator)
    }

    fn r_cycle_source(node_idx: usize) -> OpeningId {
        DefaultLookupOperands::r_cycle_source(node_idx)
    }
}

impl LookupRaPolys for SmoothReluOperands {
    fn ra_virtual_poly(node_idx: usize) -> VirtualPoly {
        ActivationClampOperands::ra_virtual_poly(node_idx)
    }

    fn ra_committed_poly(node_idx: usize, d: usize) -> CommittedPoly {
        ActivationClampOperands::ra_committed_poly(node_idx, d)
    }
}

impl OpLayout for SmoothRelu {
    fn get_committed_polynomials(
        &self,
        node: &ComputationNode,
    ) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
        Ok(clamped_activation_committed_polynomials(
            node,
            table_def(self.form),
        ))
    }
}

/// The plan: the evaluation reduction; `ReLU(x)` opened at the reduced output point; the
/// correction table's read, the clamp read and the ReLU read, queued for the post-loop lookup
/// batch; the table address's and the 32-bit address's one-hot checks, queued for the one-hot
/// batch.
impl OpPlan for SmoothRelu {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        _model: &Model,
        _io: &ModelExecutionIO,
    ) -> Result<Plan<F>, ProofVerifyError> {
        if node.scale as usize != MODEL_SCALE {
            return Err(ProofVerifyError::InvalidOpeningProof(format!(
                "SmoothRelu verifying is only calibrated for MODEL_SCALE={MODEL_SCALE} (got {})",
                node.scale
            )));
        }
        let clamp = match self.form {
            SmoothReluForm::GeluErf | SmoothReluForm::GeluTanh => OpLookup::ActivationClamp,
            SmoothReluForm::Silu => OpLookup::SiluClamp,
        };
        let defer = |batch, relation| Step::Defer { batch, relation };
        Ok(Plan(vec![
            Step::Reduce,
            Step::Open(OpOpen::SmoothReluRelu(node.clone())),
            defer(DeferredBatch::Lookups, OpRelation::Op(small_table(node)?)),
            defer(
                DeferredBatch::OneHot,
                OpRelation::SmallTableOneHot(node.clone()),
            ),
            defer(
                DeferredBatch::Lookups,
                OpRelation::Lookup(clamp, node.clone()),
            ),
            defer(
                DeferredBatch::Lookups,
                OpRelation::Lookup(OpLookup::SmoothRelu, node.clone()),
            ),
            defer(
                DeferredBatch::OneHot,
                OpRelation::LookupOneHot(OpLookup::ActivationClamp, node.clone()),
            ),
        ]))
    }
}
