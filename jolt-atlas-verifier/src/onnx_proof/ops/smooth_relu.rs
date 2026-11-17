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
use crate::onnx_proof::neural_teleport::small_table::SmallTable;
use crate::onnx_proof::neural_teleport::utils::materialize_signed_activation_table;
use crate::onnx_proof::ops::OpLayout;
use crate::onnx_proof::relation::{RelationExpr, RelationOpening};
use crate::onnx_proof::{
    deferred_lookups::DeferredBatch,
    op_lookups::{DefaultLookupOperands, LookupOperandsTrait, LookupRaPolys},
    ops::activation_clamped::{
        clamped_activation_committed_polynomials, clamped_opening_id, ActivationClampOperands,
    },
    plan::{OpActivation, OpLookup, OpOpen, OpPlan, OpRelation, Plan, Step},
};
use crate::utils::opening_access::OpeningIdBuilder;
use atlas_algebra::field::Ring;
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

macro_rules! correction_table {
    ($(#[$doc:meta])* $name:ident, $form:ident) => {
        $(#[$doc])*
        pub struct $name;

        impl SmallTable for $name {
            const VARS: usize = match SmoothReluForm::$form {
                SmoothReluForm::Silu => ACTIVATION_TABLE_VARS + 1,
                _ => ACTIVATION_TABLE_VARS,
            };
            const RA: fn(usize) -> VirtualPoly = VirtualPoly::ActivationSmallRa;
            const SIGNED: bool = true;
            fn materialize() -> Result<Vec<i32>, ProofVerifyError> {
                materialize_signed_activation_table(
                    Self::VARS,
                    1,
                    MODEL_SCALE as u8,
                    TableKind::$form,
                )
            }
            fn address(node: &ComputationNode) -> OpeningId {
                clamped_opening_id(node.idx)
            }
            /// The correction is the output less `ReLU(x)`.
            fn value<G: Ring>(node: &ComputationNode) -> RelationExpr<G> {
                opening::<G, _, _, _>(RelationOpening::ReducedNodeOutput(node.idx))
                    - opening(RelationOpening::Id(relu_opening_id(node)))
            }
        }
    };
}

correction_table!(
    /// The erf-form GELU's correction to `ReLU`.
    GeluErfTable,
    GeluErf
);
correction_table!(
    /// The tanh-form GELU's correction to `ReLU`.
    GeluTanhTable,
    GeluTanh
);
correction_table!(
    /// SiLU's correction to `ReLU`, over a domain twice as wide.
    SiluTable,
    Silu
);

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
        Ok(clamped_activation_committed_polynomials(node))
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
        let (table, clamp) = match self.form {
            SmoothReluForm::GeluErf => (OpActivation::GeluErf, OpLookup::ActivationClamp),
            SmoothReluForm::GeluTanh => (OpActivation::GeluTanh, OpLookup::ActivationClamp),
            SmoothReluForm::Silu => (OpActivation::Silu, OpLookup::SiluClamp),
        };
        let defer = |batch, relation| Step::Defer { batch, relation };
        Ok(Plan(vec![
            Step::Reduce,
            Step::Open(OpOpen::SmoothReluRelu(node.clone())),
            defer(
                DeferredBatch::Lookups,
                OpRelation::SmallActivation(table, node.clone()),
            ),
            defer(
                DeferredBatch::OneHot,
                OpRelation::SmallActivationOneHot(node.clone()),
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
