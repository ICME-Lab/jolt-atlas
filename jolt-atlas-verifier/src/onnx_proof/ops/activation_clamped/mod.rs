//! Shared plumbing for the clamped Erf/Sigmoid/Tanh variants.
//!
//! Each op fuses two lookup stages into one node:
//! 1. **Execution**: `output = SmallTable[clamped]`, where `clamped` is an internal
//!    advice claim (gamma-batched with the identity polynomial so the one-hot `ra`
//!    address is soundly tied to its value).
//! 2. **ActivationClamp**: proves that same `clamped` claim equals
//!    `ActivationClampTable[raw_input]` (natively symmetric clamp lookup, same shape as
//!    `Clamp`), which soundly ties `clamped` back to the real input rather than just
//!    asserting it.
//!
//! Both stages proceed sumcheck-then-sumcheck (Execution, then ActivationClamp), with
//! their one-hot (`ra`/`hw`/`bool`) checks batched together afterwards into a single
//! `BatchedSumcheck` call.

use atlas_trace::{Tensor, Trace};

use crate::node_view::ComputationNode;
use crate::onnx_proof::neural_teleport::small_table::{SmallTable, SmallTableRelation};
use crate::onnx_proof::relation::{RelationParams, RelationVerifier};
use crate::{
    onnx_proof::{
        deferred_lookups::{DeferredBatch, DeferredOneHot},
        op_lookups::{
            DefaultLookupOperands, LookupOperandsTrait, OpLookupEncoding, OpLookupProvider,
        },
        Verifier,
    },
    utils::compute_lookup_indices_from_operands,
};
use atlas_algebra::{
    field::JoltField,
    poly::opening_point::{OpeningPoint, BIG_ENDIAN},
    utils::{errors::ProofVerifyError, lookup_bits::LookupBits},
};
use atlas_lookup_tables::clamp::ActivationClampTable;
use atlas_sumcheck::{
    config::{OneHotConfig, OneHotParams},
    opening_proof::{OpeningAccumulator, OpeningId, SumcheckId, VerifierOpeningAccumulator},
    shout::RaOneHotEncoding,
};
use atlas_transcript::Transcript;
use common::{
    consts::{ACTIVATION_TABLE_VARS, XLEN},
    CommittedPoly, VirtualPoly,
};
use jolt_claims::SymbolicSumcheck;

/// Degree of the small-table execution sumcheck.
pub const EXEC_DEGREE_BOUND: usize = 2;

/// A clamped-activation's small lookup table: `Table[i] = activation(i)`, for
/// two's-complement `i` over [`ACTIVATION_TABLE_BOUND`] bits.
pub trait SmallActivationTable: Send + Sync + 'static {
    /// Materializes the full table (all `2^ACTIVATION_TABLE_BOUND` entries).
    fn materialize() -> Vec<i32>;
}

/// Opening of the clamped input the activation table is read at.
pub fn clamped_opening_id(node_idx: usize) -> OpeningId {
    OpeningId::new(
        VirtualPoly::ActivationClampedOutput(node_idx),
        SumcheckId::NodeExecution(node_idx),
    )
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
    ) -> F {
        accumulator
            .get_virtual_polynomial_opening(clamped_opening_id(node.idx))
            .1
    }

    fn ra_virtual_poly(node_idx: usize) -> VirtualPoly {
        VirtualPoly::ActivationClampRa(node_idx)
    }

    fn ra_committed_poly(node_idx: usize, d: usize) -> CommittedPoly {
        CommittedPoly::ActivationClampRaD(node_idx, d)
    }

    fn witness_opening_id(node: &ComputationNode) -> OpeningId {
        DefaultLookupOperands::witness_opening_id(node)
    }

    fn r_cycle<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        DefaultLookupOperands::r_cycle(node, accumulator)
    }

    fn r_cycle_source(node_idx: usize) -> OpeningId {
        DefaultLookupOperands::r_cycle_source(node_idx)
    }

    fn witness(&self, node: &ComputationNode, trace: &Trace) -> Tensor<i64> {
        DefaultLookupOperands.witness(node, trace)
    }

    fn lookup_bits(&self, witness: &Tensor<i64>) -> Vec<LookupBits> {
        let operand = witness.map(|v| v as i32);
        compute_lookup_indices_from_operands(&[&operand], false)
    }
}

// ---------------------------------------------------------------------------
// Stage: Execution -- small dense-table lookup, gamma-batched with an identity
// polynomial.
// ---------------------------------------------------------------------------

impl<Table: SmallActivationTable> SmallTable for Table {
    const VARS: usize = ACTIVATION_TABLE_VARS;
    const RA: fn(usize) -> VirtualPoly = VirtualPoly::ActivationSmallRa;
    const SIGNED: bool = true;
    fn materialize() -> Vec<i32> {
        <Table as SmallActivationTable>::materialize()
    }
    fn address(node: &ComputationNode) -> OpeningId {
        clamped_opening_id(node.idx)
    }
}

/// The execution check's relation, set up: `γ` drawn, the reduced output point read. Both sides
/// then append the clamped input's opening at that point before the sumcheck.
pub fn small_table_params<F: JoltField, Table: SmallActivationTable>(
    node: &ComputationNode,
    accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) -> RelationParams<F, SmallTableRelation<Table>> {
    RelationParams::new(
        SmallTableRelation::new(node.clone()),
        accumulator,
        transcript,
    )
}

fn small_table_verifier<F: JoltField, Table: SmallActivationTable>(
    node: &ComputationNode,
    accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) -> RelationVerifier<F, SmallTableRelation<Table>> {
    let (relation, setup) =
        small_table_params::<F, Table>(node, accumulator, transcript).into_parts();
    accumulator.append_virtual(
        transcript,
        clamped_opening_id(node.idx),
        OpeningPoint::new(setup.incoming_point.clone()),
    );
    RelationVerifier::at(relation, setup)
}

/// One-hot encoding of the small table's read addresses.
pub struct SmallTableRaEncoding {
    /// The node whose reads are encoded.
    pub node_idx: usize,
}

impl RaOneHotEncoding for SmallTableRaEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPoly {
        CommittedPoly::ActivationSmallRaD(self.node_idx, d)
    }

    fn r_cycle_source(&self) -> OpeningId {
        clamped_opening_id(self.node_idx)
    }

    fn ra_source(&self) -> OpeningId {
        OpeningId::new(
            VirtualPoly::ActivationSmallRa(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn log_k(&self) -> usize {
        ACTIVATION_TABLE_VARS
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), ACTIVATION_TABLE_VARS)
    }
}

/// Verifier counterpart of [`prove_clamped_activation`].
pub fn verify_clamped_activation<F, T, Table>(
    node: &ComputationNode,
    verifier: &mut Verifier<'_, F, T>,
) -> Result<(), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
    Table: SmallActivationTable,
{
    // Stage 1: Execution (verified in the deferred `ActivationExec` batch).
    let exec_verifier =
        small_table_verifier::<F, Table>(node, &mut verifier.accumulator, &mut verifier.transcript);
    verifier.defer(DeferredBatch::ActivationExec, Box::new(exec_verifier));

    // Stage 2: ActivationClamp (deferred `ActivationClamp` batch).
    let clamp_provider: OpLookupProvider<ActivationClampOperands> =
        OpLookupProvider::new(node.clone());
    let clamp_verifier_sumcheck = clamp_provider
        .read_raf_verify::<F, T, ActivationClampTable<XLEN>, XLEN>(
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );
    verifier.defer(
        DeferredBatch::ActivationClamp,
        Box::new(clamp_verifier_sumcheck),
    );

    // Stage 3: one-hot checks, verified after their batches.
    verifier
        .deferred_onehots
        .push(DeferredOneHot::ActivationSmall(node.idx));
    verifier
        .deferred_onehots
        .push(DeferredOneHot::ActivationClamp(node.clone()));

    Ok(())
}

/// Committed polynomials for a clamped-activation node (both stages' one-hot RaD chunks).
pub fn clamped_activation_committed_polynomials(node: &ComputationNode) -> Vec<CommittedPoly> {
    let clamp_encoding: OpLookupEncoding<ActivationClampOperands> = OpLookupEncoding::new(node);
    let small_encoding = SmallTableRaEncoding { node_idx: node.idx };
    let clamp_d = clamp_encoding.one_hot_params().instruction_d;
    let small_d = small_encoding.one_hot_params().instruction_d;
    let mut polys = vec![];
    polys.extend((0..clamp_d).map(|i| CommittedPoly::ActivationClampRaD(node.idx, i)));
    polys.extend((0..small_d).map(|i| CommittedPoly::ActivationSmallRaD(node.idx, i)));
    polys
}
