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

use crate::{
    onnx_proof::deferred_lookups::{DeferredBatch, DeferredOneHot},
    onnx_proof::{
        neural_teleport::{n_bits_to_usize, utils::compute_ra_evals_nbits_2comp},
        op_lookups::{
            DefaultLookupOperands, LookupOperandsTrait, OpLookupEncoding, OpLookupProvider,
        },
        ProofId, Prover, Verifier,
    },
    utils::{compute_lookup_indices_from_operands, opening_access::AccOpeningAccessor},
};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    tensor::{ops::nonlinearities, Tensor},
};
use common::{
    consts::{ACTIVATION_BOUND, ACTIVATION_TABLE_VARS, XLEN},
    parallel::par_enabled,
    CommittedPoly, VirtualPoly,
};
#[cfg(feature = "zk")]
use joltworks::subprotocols::blindfold::{
    InputClaimConstraint, OutputClaimConstraint, ProductTerm, ValueSource,
};
use joltworks::{
    config::{OneHotConfig, OneHotParams},
    field::{IntoOpening, JoltField},
    lookup_tables::clamp::ActivationClampTable,
    poly::{
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        signed_identity_poly::SignedIdentityPoly,
        unipoly::UniPoly,
    },
    subprotocols::{
        shout::RaOneHotEncoding,
        sumcheck::SumcheckInstanceProof,
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, lookup_bits::LookupBits},
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use std::marker::PhantomData;

const EXEC_DEGREE_BOUND: usize = 2;

/// A clamped-activation's small lookup table: `Table[i] = activation(i)`, for
/// two's-complement `i` over [`ACTIVATION_TABLE_BOUND`] bits.
pub trait SmallActivationTable: Send + Sync + 'static {
    /// Materializes the full table (all `2^ACTIVATION_TABLE_BOUND` entries).
    fn materialize() -> Vec<i32>;
}

fn clamped_opening_id(node_idx: usize) -> OpeningId {
    OpeningId::new(
        VirtualPoly::ActivationClampedOutput(node_idx),
        SumcheckId::NodeExecution(node_idx),
    )
}

// ---------------------------------------------------------------------------
// Stage: ActivationClamp -- prefix-suffix lookup proving the `ActivationClampedOutput`
// advice equals `ActivationClampTable[raw_input]`, ties `clamped` back to the real input.
// ---------------------------------------------------------------------------

#[derive(Default)]
pub(crate) struct ActivationClampOperands;

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

#[derive(Clone)]
struct SmallTableParams<F: JoltField, Table> {
    gamma: F,
    r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    computation_node: ComputationNode,
    _table: PhantomData<Table>,
}

impl<F: JoltField, Table: SmallActivationTable> SmallTableParams<F, Table> {
    fn new(
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

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::default()
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(
        &self,
        _accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        Vec::new()
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        use crate::utils::opening_access::OpeningIdBuilder;
        let builder = OpeningIdBuilder::new(&self.computation_node);
        let ra_id = builder.advice(VirtualPoly::ActivationSmallRa);
        Some(OutputClaimConstraint::sum_of_products(vec![
            ProductTerm::scaled(ValueSource::Challenge(0), vec![ValueSource::Opening(ra_id)]),
        ]))
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        let opening_point = self.normalize_opening_point(&sumcheck_challenges.into_opening());
        let table = MultilinearPolynomial::from(Table::materialize());
        let table_claim = table.evaluate(&opening_point.r);
        let int_eval = SignedIdentityPoly::new(ACTIVATION_TABLE_VARS).evaluate(&opening_point.r);
        vec![table_claim + self.gamma * int_eval]
    }
}

struct SmallTableProver<F: JoltField, Table> {
    params: SmallTableParams<F, Table>,
    table: MultilinearPolynomial<F>,
    input_onehot: MultilinearPolynomial<F>,
    identity: SignedIdentityPoly<F>,
    /// Lookup indices for this stage's one-hot checks, so the caller can reuse them
    /// without recomputing from the trace.
    lookup_indices: Vec<usize>,
}

impl<F: JoltField, Table: SmallActivationTable> SmallTableProver<F, Table> {
    /// Computes the clamped tensor and registers its advice claim at `r_node_output`;
    /// the `ActivationClamp` stage then ties that claim back to `raw_input`.
    fn initialize(
        params: SmallTableParams<F, Table>,
        trace: &Trace,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let LayerData { operands, .. } = Trace::layer_data(trace, &params.computation_node);
        let raw_input = operands[0];
        let clamped_tensor = nonlinearities::clamp(raw_input, ACTIVATION_BOUND);
        let clamped_tensor_padded = clamped_tensor.padded_next_power_of_two();

        let clamped_claim = MultilinearPolynomial::from(
            clamped_tensor_padded
                .data()
                .iter()
                .map(|&v| v as i64)
                .collect::<Vec<_>>(),
        )
        .evaluate(&params.r_node_output.r);
        accumulator.append_virtual(
            transcript,
            clamped_opening_id(params.computation_node.idx),
            params.r_node_output.clone(),
            clamped_claim,
        );

        let table = MultilinearPolynomial::from(Table::materialize());
        let input_onehot: Vec<F> = compute_ra_evals_nbits_2comp(
            &params.r_node_output.r,
            &clamped_tensor,
            ACTIVATION_TABLE_VARS,
        );
        let input_onehot = MultilinearPolynomial::from(input_onehot);
        assert_eq!(input_onehot.len(), table.len());
        let identity = SignedIdentityPoly::new(ACTIVATION_TABLE_VARS);

        let lookup_indices: Vec<usize> = clamped_tensor_padded
            .par_iter()
            .with_min_len(par_enabled())
            .map(|&x| n_bits_to_usize(x, ACTIVATION_TABLE_VARS))
            .collect();

        Self {
            params,
            table,
            input_onehot,
            identity,
            lookup_indices,
        }
    }
}

impl<F: JoltField, T: Transcript, Table: SmallActivationTable> SumcheckInstanceProver<F, T>
    for SmallTableProver<F, Table>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self {
            input_onehot,
            table,
            identity,
            ..
        } = self;

        let univariate_poly_evals: [F; 2] = (0..input_onehot.len() / 2)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|i| {
                let ra_evals =
                    input_onehot.sumcheck_evals(i, EXEC_DEGREE_BOUND, BindingOrder::LowToHigh);
                let table_evals =
                    table.sumcheck_evals(i, EXEC_DEGREE_BOUND, BindingOrder::LowToHigh);
                let id_evals =
                    identity.sumcheck_evals(i, EXEC_DEGREE_BOUND, BindingOrder::LowToHigh);

                [
                    ra_evals[0] * (table_evals[0] + id_evals[0] * self.params.gamma),
                    ra_evals[1] * (table_evals[1] + id_evals[1] * self.params.gamma),
                ]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        UniPoly::from_evals_and_hint(previous_claim, &univariate_poly_evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.input_onehot
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.table.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.identity.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
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
        provider.append_advice(
            VirtualPoly::ActivationSmallRa,
            self.input_onehot.final_claim(),
        );
    }
}

struct SmallTableVerifier<F: JoltField, Table> {
    params: SmallTableParams<F, Table>,
    table: MultilinearPolynomial<F>,
}

impl<F: JoltField, Table: SmallActivationTable> SmallTableVerifier<F, Table> {
    fn new(
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

pub(crate) struct SmallTableRaEncoding {
    pub(crate) node_idx: usize,
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

// ---------------------------------------------------------------------------
// Public entry points, called by the thin per-op `OperatorProofTrait` impls.
// ---------------------------------------------------------------------------

/// Full (Execution + ActivationClamp) proof for a clamped-activation node: Execution
/// establishes the clamped advice claim, ActivationClamp ties it back to the raw
/// input, then both stages' one-hot checks are batched into one `BatchedSumcheck`.
pub fn prove_clamped_activation<F, T, Table>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)>
where
    F: JoltField,
    T: Transcript,
    Table: SmallActivationTable,
{
    // Stage 1: Execution -- output = SmallTable[clamped]. Built here (its
    // challenge is drawn now, its `clamped` advice claim appended now), proven
    // after the node loop in the `ActivationExec` batch.
    let params = SmallTableParams::<F, Table>::new(
        node.clone(),
        &prover.accumulator,
        &mut prover.transcript,
    );
    let exec_sumcheck = SmallTableProver::<F, Table>::initialize(
        params,
        &prover.trace,
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    let small_indices = exec_sumcheck.lookup_indices.clone();
    prover.defer(DeferredBatch::ActivationExec, Box::new(exec_sumcheck));

    // Stage 2: ActivationClamp -- clamped = ActivationClampTable[raw_input].
    // Same: built now, proven in the `ActivationClamp` batch.
    let clamp_provider: OpLookupProvider<ActivationClampOperands> =
        OpLookupProvider::new(node.clone());
    let (clamp_sumcheck, clamp_lookup_indices) = clamp_provider
        .read_raf_prove::<F, T, ActivationClampTable<XLEN>, XLEN>(
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
    prover.defer(DeferredBatch::ActivationClamp, Box::new(clamp_sumcheck));

    // Stage 3: both stages' one-hot checks, built and proven after their batches.
    prover
        .deferred_onehots
        .push((DeferredOneHot::ActivationSmall(node.idx), small_indices));
    prover.deferred_onehots.push((
        DeferredOneHot::ActivationClamp(node.clone()),
        clamp_lookup_indices,
    ));

    Vec::new()
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
