//! The clamped activations Erf, Sigmoid and Tanh.
//!
//! Each op fuses two lookup stages into one node:
//! 1. **Execution**: `output = Table[clamped]` over the activation's small table
//!    ([`small_table`]), where `clamped` is an internal advice claim (gamma-batched with the
//!    identity polynomial so the one-hot `ra` address is soundly tied to its value).
//! 2. **ActivationClamp**: proves that same `clamped` claim equals
//!    `ActivationClampTable[raw_input]` (natively symmetric clamp lookup, same shape as
//!    `Clamp`), which soundly ties `clamped` back to the real input rather than just
//!    asserting it.
//!
//! Both stages are deferred to the post-loop batches ([`clamped_activation_plan`]): the two
//! reads join the lookup batch, their one-hot (`ra`/`hw`/`bool`) checks the one-hot batch.

use crate::node_view::ComputationNode;
use crate::onnx_proof::lookup::{output_value, small_table, SmallTableDef};
use crate::onnx_proof::ops::OpLayout;
use crate::onnx_proof::{
    deferred_lookups::DeferredBatch,
    op_lookups::{DefaultLookupOperands, LookupOperandsTrait, LookupRaPolys, OpLookupEncoding},
    plan::{OpLookup, OpPlan, OpRelation, Plan, Step},
};
use atlas_algebra::{
    field::JoltField,
    poly::point::{OpeningPoint, BIG_ENDIAN},
    utils::errors::ProofVerifyError,
};
use atlas_sumcheck::{
    config::{OneHotConfig, OneHotParams},
    opening_proof::{OpeningAccumulator, OpeningId, SumcheckId},
    shout::RaOneHotEncoding,
};
use atlas_trace::{
    op::{Erf, Sigmoid, Tanh},
    semantics::tables::TableKind,
    trace::ModelExecutionIO,
    Model,
};
use common::{
    consts::{ACTIVATION_TABLE_VARS, XLEN},
    CommittedPoly, VirtualPoly,
};

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

// ---------------------------------------------------------------------------
// Stage: ActivationClamp -- prefix-suffix lookup proving the `ActivationClampedOutput`
// advice equals `ActivationClampTable[raw_input]`, ties `clamped` back to the real input.
// ---------------------------------------------------------------------------

/// Lookup operands of the activation's input clamp.
#[derive(Default)]
pub struct ActivationClampOperands;

impl LookupOperandsTrait for ActivationClampOperands {
    const LOG_K: usize = XLEN;

    fn rv_claim<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Result<F, ProofVerifyError> {
        Ok(accumulator
            .get_virtual_polynomial_opening(clamped_opening_id(node.idx))?
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

impl LookupRaPolys for ActivationClampOperands {
    fn ra_virtual_poly(node_idx: usize) -> VirtualPoly {
        VirtualPoly::ActivationClampRa(node_idx)
    }

    fn ra_committed_poly(node_idx: usize, d: usize) -> CommittedPoly {
        CommittedPoly::ActivationClampRaD(node_idx, d)
    }
}

/// The plan of a clamped activation: the evaluation reduction, then the two stages queued for
/// the post-loop lookup batch and their one-hot checks for the one-hot batch. `Err` if the
/// node's operator reads no small table.
pub fn clamped_activation_plan<F: JoltField>(
    node: &ComputationNode,
) -> Result<Plan<F>, ProofVerifyError> {
    Ok(Plan(vec![
        Step::Reduce,
        Step::Defer {
            batch: DeferredBatch::Lookups,
            relation: OpRelation::Op(small_table(node)?),
        },
        Step::Defer {
            batch: DeferredBatch::OneHot,
            relation: OpRelation::SmallTableOneHot(node.clone()),
        },
        Step::Defer {
            batch: DeferredBatch::Lookups,
            relation: OpRelation::Lookup(OpLookup::ActivationClamp, node.clone()),
        },
        Step::Defer {
            batch: DeferredBatch::OneHot,
            relation: OpRelation::LookupOneHot(OpLookup::ActivationClamp, node.clone()),
        },
    ]))
}

/// Committed polynomials for a node reading the small table `def` at its clamped input: both
/// stages' one-hot chunks.
pub fn clamped_activation_committed_polynomials(
    node: &ComputationNode,
    def: SmallTableDef,
) -> Vec<CommittedPoly> {
    let clamp_encoding: OpLookupEncoding<ActivationClampOperands> = OpLookupEncoding::new(node);
    let clamp_d = clamp_encoding.one_hot_params().instruction_d;
    let small_d =
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), def.vars).instruction_d;
    let mut polys = vec![];
    polys.extend((0..clamp_d).map(|i| CommittedPoly::ActivationClampRaD(node.idx, i)));
    polys.extend((0..small_d).map(|i| (def.ra_chunk)(node.idx, i)));
    polys
}

impl OpLayout for Erf {
    fn get_committed_polynomials(
        &self,
        node: &ComputationNode,
    ) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
        Ok(clamped_activation_committed_polynomials(
            node,
            table_def(TableKind::Erf),
        ))
    }
}

impl OpPlan for Erf {
    fn plan<F: JoltField>(
        &self,
        node: &ComputationNode,
        _model: &Model,
        _io: &ModelExecutionIO,
    ) -> Result<Plan<F>, ProofVerifyError> {
        clamped_activation_plan(node)
    }
}

impl OpLayout for Sigmoid {
    fn get_committed_polynomials(
        &self,
        node: &ComputationNode,
    ) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
        Ok(clamped_activation_committed_polynomials(
            node,
            table_def(TableKind::Sigmoid),
        ))
    }
}

impl OpPlan for Sigmoid {
    fn plan<F: JoltField>(
        &self,
        node: &ComputationNode,
        _model: &Model,
        _io: &ModelExecutionIO,
    ) -> Result<Plan<F>, ProofVerifyError> {
        clamped_activation_plan(node)
    }
}

impl OpLayout for Tanh {
    fn get_committed_polynomials(
        &self,
        node: &ComputationNode,
    ) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
        Ok(clamped_activation_committed_polynomials(
            node,
            table_def(TableKind::Tanh),
        ))
    }
}

impl OpPlan for Tanh {
    fn plan<F: JoltField>(
        &self,
        node: &ComputationNode,
        _model: &Model,
        _io: &ModelExecutionIO,
    ) -> Result<Plan<F>, ProofVerifyError> {
        clamped_activation_plan(node)
    }
}
