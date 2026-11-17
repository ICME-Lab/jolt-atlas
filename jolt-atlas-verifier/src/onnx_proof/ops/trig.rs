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
use crate::onnx_proof::lookup::{
    output_value, small_table, PrefixSuffixTable, SmallTableDef, TableRead,
};
use crate::onnx_proof::ops::OpLayout;
use crate::onnx_proof::plan::{OpOpen, OpPlan, OpRelation, Plan, Step};
use crate::onnx_proof::relation::{OpRel, OpRelKind, RelationOpening};
use crate::utils::opening_access::{OpeningIdBuilder, Target};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_claims::{constant, opening, Int};
use atlas_trace::{
    op::{Cos, Sin},
    semantics::tables::TableKind,
    trace::ModelExecutionIO,
    Model,
};
use common::consts::{MODEL_SCALE, TRIG_DOWNSCALE_BITS, TRIG_PERIOD_MODULUS};
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
    let mut polys = vec![CommittedPoly::TrigPeriodQuotient(node.idx)];
    polys.extend(downscale_read(node).ra.committed_polys());
    polys.extend(def.ra_encoding(node).committed_polys());
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
    let downscale = downscale_read(node);
    let downscale_ra = downscale.ra.clone();
    Ok(Plan(vec![
        Step::Reduce,
        Step::Open(OpOpen::TrigPeriodQuotient(node.clone())),
        Step::Open(OpOpen::TrigDownscaled(node.clone())),
        Step::Defer {
            batch: DeferredBatch::Lookups,
            relation: OpRelation::TableRead(downscale),
        },
        Step::Defer {
            batch: DeferredBatch::Lookups,
            relation: OpRelation::Op(small_table(node)?),
        },
        Step::Defer {
            batch: DeferredBatch::OneHot,
            relation: OpRelation::OneHot(downscale_ra),
        },
        Step::Defer {
            batch: DeferredBatch::OneHot,
            relation: OpRelation::OneHot(SmallTableDef::for_node(node)?.ra_encoding(node)),
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

/// The downscale's read: the downscaled remainder is `RightShift[rem]` at the reduced output
/// point. Its operand is the period remainder, whose claim the period reduction reads.
pub fn downscale_read(node: &ComputationNode) -> TableRead {
    let ids = OpeningIdBuilder::new(node);
    TableRead::new(
        node,
        PrefixSuffixTable::RightShift,
        RelationOpening::Id(ids.advice(VirtualPoly::TrigDownscaled)),
        ids.advice(VirtualPoly::TrigPeriodRemainder),
        RelationOpening::ReducedNodeOutput(node.idx),
        ids.nodeio(Target::Current),
        VirtualPoly::TrigDownscaleRa,
        CommittedPoly::TrigDownscaleRaD,
    )
}
