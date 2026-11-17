//! Sin and Cos.
//!
//! The input is reduced by the period `P` into a quotient and a remainder
//! (`input = P·q + rem`, [`trig_period`]); the remainder is downscaled by a right shift
//! (`downscaled = rem >> TRIG_DOWNSCALE_BITS`, a [`RightShiftTable`](atlas_lookup_tables::right_shift::RightShiftTable)
//! read); and the output is the trig table read at the downscaled remainder
//! ([`small_table`](crate::onnx_proof::lookup::small_table)), whose entries carry the
//! downscale's rescale back to `2^MODEL_SCALE` precision.

use crate::node_view::ComputationNode;
use crate::onnx_proof::deferred_lookups::DeferredBatch;
use crate::onnx_proof::lookup::{output_value, small_table, SmallTableDef};
use crate::onnx_proof::ops::OpLayout;
use crate::onnx_proof::relation::{OpRel, OpRelKind, RelationOpening};
use crate::onnx_proof::{
    op_lookups::{LookupOperandsTrait, LookupRaPolys, OpLookupEncoding},
    plan::{OpLookup, OpOpen, OpPlan, OpRelation, Plan, Step},
};
use crate::utils::opening_access::{AccOpeningAccessor, OpeningIdBuilder, Target};
use atlas_algebra::{
    field::JoltField,
    poly::point::{OpeningPoint, BIG_ENDIAN},
    utils::errors::ProofVerifyError,
};
use atlas_claims::{constant, opening, Int};
use atlas_sumcheck::{
    config::{OneHotConfig, OneHotParams},
    opening_proof::{OpeningAccumulator, OpeningId, SumcheckId},
    shout::RaOneHotEncoding,
};
use atlas_trace::{
    op::{Cos, Sin},
    semantics::tables::TableKind,
    trace::ModelExecutionIO,
    Model,
};
use common::consts::{MODEL_SCALE, TRIG_DOWNSCALE_BITS, TRIG_PERIOD_MODULUS, XLEN};
use common::{CommittedPoly, VirtualPoly};

/// Address variables of the Sin and Cos tables, after downscaling.
///
/// The un-downscaled table size is the next power of two above the period, so every
/// remainder in `[0, TRIG_PERIOD_MODULUS)` is a table address; downscaling then shrinks that
/// by `TRIG_DOWNSCALE_BITS`.
pub const TRIG_TABLE_VARS: usize = (TRIG_PERIOD_MODULUS as usize).next_power_of_two().ilog2()
    as usize
    - TRIG_DOWNSCALE_BITS as usize;

/// The sine table, read at a Sin node's downscaled remainder.
pub fn sin_table() -> SmallTableDef {
    table_def(TableKind::Sin, VirtualPoly::SinRa, CommittedPoly::SinRaD)
}

/// The cosine table, read at a Cos node's downscaled remainder.
pub fn cos_table() -> SmallTableDef {
    table_def(TableKind::Cos, VirtualPoly::CosRa, CommittedPoly::CosRaD)
}

/// A trig table read at the downscaled remainder, which an earlier step of the plan opens.
fn table_def(
    kind: TableKind,
    ra: fn(usize) -> VirtualPoly,
    ra_chunk: fn(usize, usize) -> CommittedPoly,
) -> SmallTableDef {
    SmallTableDef {
        kind,
        vars: TRIG_TABLE_VARS,
        signed: false,
        ra,
        ra_chunk,
        address: VirtualPoly::TrigDownscaled,
        opens_address: false,
        value: output_value,
    }
}

/// The committed polynomials of a trig node reading the table `def`: the period quotient, the
/// downscale's address chunks and the table's.
fn committed_polynomials(node: &ComputationNode, def: SmallTableDef) -> Vec<CommittedPoly> {
    let table_d =
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), def.vars).instruction_d;
    let downscale_d = OpLookupEncoding::<TrigDownscaleOperands>::new(node)
        .one_hot_params()
        .instruction_d;
    let mut polys = vec![CommittedPoly::TrigPeriodQuotient(node.idx)];
    polys.extend((0..downscale_d).map(|i| CommittedPoly::TrigDownscaleRaD(node.idx, i)));
    polys.extend((0..table_d).map(|i| (def.ra_chunk)(node.idx, i)));
    polys
}

/// The plan of a trig node `name`: the evaluation reduction; the period quotient and the
/// downscaled remainder opened at the reduced output point; the downscale and table reads and
/// their one-hot checks, queued for the post-loop batches; then the period reduction
/// `input = P·q + rem` in the clear, once `q` and `rem` both hold claims at the reduced output
/// point.
fn trig_plan<F: JoltField>(
    node: &ComputationNode,
    name: &str,
) -> Result<Plan<F>, ProofVerifyError> {
    if node.scale != MODEL_SCALE as u8 {
        return Err(ProofVerifyError::InvalidOpeningProof(format!(
            "{name} verifying is only calibrated for MODEL_SCALE={MODEL_SCALE} (got {})",
            node.scale
        )));
    }
    Ok(Plan(vec![
        Step::Reduce,
        Step::Open(OpOpen::TrigPeriodQuotient(node.clone())),
        Step::Open(OpOpen::TrigDownscaled(node.clone())),
        Step::Defer {
            batch: DeferredBatch::Lookups,
            relation: OpRelation::Lookup(OpLookup::TrigDownscale, node.clone()),
        },
        Step::Defer {
            batch: DeferredBatch::Lookups,
            relation: OpRelation::Op(small_table(node)?),
        },
        Step::Defer {
            batch: DeferredBatch::OneHot,
            relation: OpRelation::LookupOneHot(OpLookup::TrigDownscale, node.clone()),
        },
        Step::Defer {
            batch: DeferredBatch::OneHot,
            relation: OpRelation::SmallTableOneHot(node.clone()),
        },
        Step::Tie {
            relation: trig_period(node),
        },
    ]))
}

impl OpLayout for Sin {
    fn get_committed_polynomials(
        &self,
        node: &ComputationNode,
    ) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
        Ok(committed_polynomials(node, sin_table()))
    }
}

impl OpPlan for Sin {
    fn plan<F: JoltField>(
        &self,
        node: &ComputationNode,
        _model: &Model,
        _io: &ModelExecutionIO,
    ) -> Result<Plan<F>, ProofVerifyError> {
        trig_plan(node, "Sin")
    }
}

impl OpLayout for Cos {
    fn get_committed_polynomials(
        &self,
        node: &ComputationNode,
    ) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
        Ok(committed_polynomials(node, cos_table()))
    }
}

impl OpPlan for Cos {
    fn plan<F: JoltField>(
        &self,
        node: &ComputationNode,
        _model: &Model,
        _io: &ModelExecutionIO,
    ) -> Result<Plan<F>, ProofVerifyError> {
        trig_plan(node, "Cos")
    }
}

/// The period reduction of a trig node as a tie at the reduced output point `r`: the node's
/// input, opened there, is `P·q + rem` over the quotient and remainder claims already bound
/// there.
///
/// ```text
/// input claim   P·q(r) + rem(r)        (the committed quotient, the remainder)
/// output claim  input(r)               (the operand, opened at r)
/// ```
///
/// **S ⊆ R.** `atlas_trace` reduces the input by the period `P` into a quotient and a
/// remainder with `input = P·q + rem` at every element, so the identity holds between the
/// three extensions at every point. The remainder's range is the downscale lookup's, outside
/// the relation.
pub fn trig_period<F: JoltField>(node: &ComputationNode) -> OpRel<F> {
    let ids = OpeningIdBuilder::new(node);
    let input = ids.nodeio(Target::Input(0));
    OpRel {
        degree: 1,
        input: constant(Int(i128::from(TRIG_PERIOD_MODULUS as i32)))
            * opening(RelationOpening::Id(
                ids.advice(CommittedPoly::TrigPeriodQuotient),
            ))
            + opening(RelationOpening::Id(
                ids.advice(VirtualPoly::TrigPeriodRemainder),
            )),
        output: opening(RelationOpening::Id(input)),
        produced: vec![input],
        ..OpRel::base(OpRelKind::TrigPeriod, node)
    }
}

/// Lookup operands of the downscale: the read's operand is the period remainder
/// (`VirtualPoly::TrigPeriodRemainder`, whose claim the period reduction reads), its value the
/// downscaled remainder, its one-hot addresses `VirtualPoly::TrigDownscaleRa` over the
/// committed `CommittedPoly::TrigDownscaleRaD` chunks. Shared by Cos and Sin.
#[derive(Default, Clone)]
pub struct TrigDownscaleOperands;

impl LookupOperandsTrait for TrigDownscaleOperands {
    const LOG_K: usize = XLEN;

    fn rv_claim<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Result<F, ProofVerifyError> {
        Ok(AccOpeningAccessor::new(accumulator, node)
            .get_advice(VirtualPoly::TrigDownscaled)?
            .1)
    }

    fn witness_opening_id(node: &ComputationNode) -> OpeningId {
        OpeningId::new(
            VirtualPoly::TrigPeriodRemainder(node.idx),
            SumcheckId::NodeExecution(node.idx),
        )
    }

    fn r_cycle<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Result<OpeningPoint<BIG_ENDIAN, F>, ProofVerifyError> {
        Ok(AccOpeningAccessor::new(accumulator, node)
            .get_reduced_opening()?
            .0)
    }

    fn r_cycle_source(node_idx: usize) -> OpeningId {
        OpeningId::new(
            VirtualPoly::NodeOutput(node_idx),
            SumcheckId::NodeExecution(node_idx),
        )
    }
}

impl LookupRaPolys for TrigDownscaleOperands {
    fn ra_virtual_poly(node_idx: usize) -> VirtualPoly {
        VirtualPoly::TrigDownscaleRa(node_idx)
    }

    fn ra_committed_poly(node_idx: usize, d: usize) -> CommittedPoly {
        CommittedPoly::TrigDownscaleRaD(node_idx, d)
    }
}
