//! The clamped activations Erf, Sigmoid and Tanh.
//!
//! Each op fuses two lookup stages into one node:
//! 1. **Execution**: `output = Table[clamped]` over the activation's small table
//!    ([`small_table`]), where `clamped` is an internal advice claim (gamma-batched with the
//!    identity polynomial so the one-hot `ra` address is soundly tied to its value).
//! 2. **ActivationClamp**: proves that same `clamped` claim equals
//!    `ActivationClampTable[raw_input]` (natively symmetric clamp lookup), which soundly
//!    ties `clamped` back to the real input rather than just asserting it.
//!
//! Both stages are proven in the model-wide sumchecks ([`clamped_activation_plan`]): the two
//! reads join the lookups sumcheck, their one-hot (`ra`/`hw`/`bool`) checks the one-hot sumcheck.

use crate::node_view::ComputationNode;
use crate::onnx_proof::lookup::{
    output_value, small_table, PrefixSuffixTable, SmallTableDef, TableRead,
};
use crate::onnx_proof::relation::RelationOpening;
use crate::onnx_proof::schedule::{OpPlan, OpRelation, ScheduleBuilder};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_sumcheck::opening_proof::{OpeningId, SumcheckId};
use atlas_trace::{
    op::{Erf, Sigmoid, Tanh},
    semantics::tables::TableKind,
    Model,
};
use common::{consts::ACTIVATION_TABLE_VARS, CommittedPoly, VirtualPoly};

/// Degree of the small-table execution sumcheck.
pub const EXEC_DEGREE_BOUND: usize = 2;

/// Opening of the clamped input the activation table is read at.
pub fn clamped_opening_id(node_idx: usize) -> OpeningId {
    OpeningId::new(
        VirtualPoly::ActivationClampedOutput(node_idx),
        SumcheckId::NodeExecution(node_idx),
    )
}

/// A clamped activation's small table: `Table[i] = activation(i)` for two's-complement `i`
/// over [`ACTIVATION_TABLE_VARS`] bits, read at the clamped input, whose opening the read's
/// setup appends.
pub fn table_def(kind: TableKind) -> SmallTableDef {
    SmallTableDef {
        kind,
        vars: ACTIVATION_TABLE_VARS,
        signed: true,
        ra: VirtualPoly::ActivationSmallRa,
        ra_chunk: CommittedPoly::ActivationSmallRaD,
        address: VirtualPoly::ActivationClampedOutput,
        opens_address: true,
        value: output_value,
    }
}

/// The input clamp's read: the clamped input is `table[operand]` at the reduced output point,
/// `table` the clamp of the activation's width ([`PrefixSuffixTable::ActivationClamp`], or
/// SiLU's wider [`PrefixSuffixTable::SiluClamp`]). It ties `clamped` back to the real input.
pub fn clamp_read(node: &ComputationNode, table: PrefixSuffixTable) -> TableRead {
    TableRead::of_operand(
        node,
        table,
        RelationOpening::Id(clamped_opening_id(node.idx)),
        VirtualPoly::ActivationClampRa,
        CommittedPoly::ActivationClampRaD,
    )
}

/// The plan of a clamped activation: the evaluation reduction, then the two stages, proven in
/// the lookups sumcheck, and their one-hot checks, proven in the one-hot sumcheck. `Err` if the
/// node's operator reads no small table.
pub fn clamped_activation_plan<F: JoltField>(
    node: &ComputationNode,
    schedule: &mut ScheduleBuilder<F>,
) -> Result<(), ProofVerifyError> {
    let clamp = clamp_read(node, PrefixSuffixTable::ActivationClamp);
    let clamp_ra = clamp.ra.clone();
    let def = SmallTableDef::for_node(node)?;
    schedule.reduce();
    schedule.lookup(OpRelation::Op(small_table(node, def)));
    schedule.one_hot(OpRelation::OneHot(def.ra_encoding(node)));
    schedule.lookup(OpRelation::TableRead(clamp));
    schedule.one_hot(OpRelation::OneHot(clamp_ra));
    Ok(())
}

impl OpPlan for Erf {
    fn plan<F: JoltField>(
        &self,
        node: &ComputationNode,
        _model: &Model,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        clamped_activation_plan(node, schedule)
    }
}

impl OpPlan for Sigmoid {
    fn plan<F: JoltField>(
        &self,
        node: &ComputationNode,
        _model: &Model,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        clamped_activation_plan(node, schedule)
    }
}

impl OpPlan for Tanh {
    fn plan<F: JoltField>(
        &self,
        node: &ComputationNode,
        _model: &Model,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        clamped_activation_plan(node, schedule)
    }
}
