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
use crate::{
    onnx_proof::deferred_lookups::{DeferredBatch, DeferredOneHot},
    onnx_proof::{
        op_lookups::{
            DefaultLookupOperands, LookupOperandsTrait, OpLookupEncoding, OpLookupProvider,
        },
        Verifier,
    },
    utils::{compute_lookup_indices_from_operands, opening_access::AccOpeningAccessor},
};
use atlas_algebra::{
    field::{IntoOpening, JoltField},
    poly::{
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_point::{OpeningPoint, BIG_ENDIAN, LITTLE_ENDIAN},
        signed_identity_poly::SignedIdentityPoly,
    },
    utils::{errors::ProofVerifyError, lookup_bits::LookupBits},
};
use atlas_lookup_tables::clamp::ActivationClampTable;
use atlas_sumcheck::{
    config::{OneHotConfig, OneHotParams},
    opening_proof::{OpeningAccumulator, OpeningId, SumcheckId, VerifierOpeningAccumulator},
    shout::RaOneHotEncoding,
    sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
};
use atlas_transcript::Transcript;
use common::{
    consts::{ACTIVATION_TABLE_VARS, XLEN},
    CommittedPoly, VirtualPoly,
};
use std::marker::PhantomData;

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

/// Parameters of the small-table execution sumcheck.
#[derive(Clone)]
pub struct SmallTableParams<F: JoltField, Table> {
    /// Batching challenge.
    pub gamma: F,
    /// The node output's reduced opening point.
    pub r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    /// The node being proven.
    pub computation_node: ComputationNode,
    _table: PhantomData<Table>,
}

impl<F: JoltField, Table: SmallActivationTable> SmallTableParams<F, Table> {
    /// Read the node's reduced output opening and draw the batching challenge.
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let gamma = transcript.challenge_scalar();
        let (r_node_output, _) = accumulator.get_node_output_opening(computation_node.idx);
        Self {
            gamma,
            r_node_output,
            computation_node,
            _table: PhantomData,
        }
    }
}

impl<F: JoltField, Table: SmallActivationTable> SumcheckInstanceParams<F>
    for SmallTableParams<F, Table>
{
    fn degree(&self) -> usize {
        EXEC_DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, rv_claim) = accumulator.get_node_output_opening(self.computation_node.idx);
        let (_, clamped_claim) = accumulator
            .get_virtual_polynomial_opening(clamped_opening_id(self.computation_node.idx));
        rv_claim + self.gamma * clamped_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        ACTIVATION_TABLE_VARS
    }
}

struct SmallTableVerifier<F: JoltField, Table> {
    params: SmallTableParams<F, Table>,
    table: MultilinearPolynomial<F>,
}

impl<F: JoltField, Table: SmallActivationTable> SmallTableVerifier<F, Table> {
    /// Read the node's reduced output opening and draw the batching challenge.
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = SmallTableParams::new(computation_node, accumulator, transcript);
        accumulator.append_virtual(
            transcript,
            clamped_opening_id(params.computation_node.idx),
            params.r_node_output.clone(),
        );
        let table = MultilinearPolynomial::from(Table::materialize());
        Self { params, table }
    }
}

impl<F: JoltField, T: Transcript, Table: SmallActivationTable> SumcheckInstanceVerifier<F, T>
    for SmallTableVerifier<F, Table>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.params.computation_node);
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());

        let ra_claim = accessor.get_advice(VirtualPoly::ActivationSmallRa).1;
        let table_claim = self.table.evaluate(&opening_point.r);
        let int_eval = SignedIdentityPoly::new(ACTIVATION_TABLE_VARS).evaluate(&opening_point.r);

        ra_claim * (table_claim + self.params.gamma * int_eval)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let r = [
            opening_point.r.as_slice(),
            self.params.r_node_output.r.as_slice(),
        ]
        .concat();
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .into_provider(transcript, OpeningPoint::new(r));
        provider.append_advice(VirtualPoly::ActivationSmallRa);
    }
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
    let exec_verifier = SmallTableVerifier::<F, Table>::new(
        node.clone(),
        &mut verifier.accumulator,
        &mut verifier.transcript,
    );
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
