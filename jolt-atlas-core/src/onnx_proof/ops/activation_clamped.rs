//! Shared plumbing for the clamped Erf/Sigmoid/Tanh variants.
//!
//! Each op fuses two proof stages into a single ONNX-graph node, proven in the usual
//! reverse-topological node order:
//!
//! 1. **Execution** (small dense-table lookup): proves `output = SmallTable[clamped]`,
//!    where `output` is this node's real, already-established output claim, and
//!    `clamped` is an internal advice claim (gamma-batched with the identity
//!    polynomial, exactly as `neural_teleport`'s Tanh does today, so the one-hot `ra`
//!    polynomial's encoded address is soundly tied to `clamped`'s claimed value).
//! 2. **ActivationClamp** (cheap prefix-suffix lookup): proves that the *same* `clamped`
//!    claim from stage 1 equals `ActivationClampTable[raw_input]` (offset-trick, same
//!    shape as the `Clamp` op), and in doing so produces a new claim about `raw_input`
//!    -- exactly the same claim-chaining every other node's operand already goes
//!    through. This is what closes the soundness gap `neural_teleport`'s Tanh has
//!    today (see the `TODO` on `DummyClampedTanhInput`): here `clamped` is *proven*
//!    equal to the real clamp of the input, not just asserted.
//!
//! Both stages evaluate their tensors at this node's own output point (`r_node_output`,
//! from the standard per-node eval-reduction) since clamping is a plain per-element
//! operation -- no `tau`-division/remainder machinery (and thus no `EvalShift`-style
//! point-shifting) is needed.

use crate::{
    onnx_proof::{
        neural_teleport::{n_bits_to_usize, utils::compute_ra_evals_nbits_2comp},
        op_lookups::{DefaultLookupOperands, LookupOperandsTrait, OpLookupEncoding, OpLookupProvider},
        ProofId, ProofType, Prover, Verifier,
    },
    utils::{compute_lookup_indices_from_operands, opening_access::AccOpeningAccessor},
};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    tensor::{ops::nonlinearities, Tensor},
};
use common::{
    consts::{ACTIVATION_BOUND, ACTIVATION_TABLE_BOUND, XLEN},
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
        shout::{self, RaOneHotEncoding},
        sumcheck::{BatchedSumcheck, Sumcheck, SumcheckInstanceProof},
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

/// Offset mapping the symmetric clamp range `[-2^ACTIVATION_BOUND, 2^ACTIVATION_BOUND - 1]`
/// onto `ActivationClampTable`'s floor-at-0 domain (same trick as `SymmetricClampOperands`).
const OFFSET: i32 = 1 << ACTIVATION_BOUND;
const EXEC_DEGREE_BOUND: usize = 2;

/// Materializes a clamped-activation's small lookup table: `Table[i] = activation(i)`
/// at model scale [`common::consts::SCALE_12`], for two's-complement `i` over
/// [`ACTIVATION_TABLE_BOUND`] bits.
pub trait SmallActivationTable: Send + Sync {
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
// Stage: ActivationClamp -- cheap prefix-suffix lookup via `OpLookupProvider`,
// proving the shared `ActivationClampedOutput` advice equals
// `ActivationClampTable[raw_input]` (offset-trick, same shape as `Clamp`'s
// `SymmetricClampOperands`). Its `rv_claim` is this internal advice claim rather
// than the node's own real output, via `LookupOperandsTrait::rv_claim`.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct ActivationClampOperands;

impl LookupOperandsTrait for ActivationClampOperands {
    const LOG_K: usize = XLEN;

    fn rv_claim<F: JoltField>(node: &ComputationNode, accumulator: &dyn OpeningAccumulator<F>) -> F {
        accumulator
            .get_virtual_polynomial_opening(clamped_opening_id(node.idx))
            .1
    }

    fn transform_operand_claims<F: JoltField>(&self, claims: Vec<F>) -> (F, F) {
        (claims[0], claims[1] + F::from_u64(OFFSET as u64))
    }

    fn transform_output_claim<F: JoltField>(&self, claim: F) -> F {
        claim + F::from_u64(OFFSET as u64)
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

    fn witness(&self, node: &ComputationNode, trace: &Trace) -> Tensor<i64> {
        DefaultLookupOperands.witness(node, trace)
    }

    fn lookup_bits(witness: &Tensor<i64>) -> Vec<LookupBits> {
        let operand = witness.map(|v| v as i32 + OFFSET);
        compute_lookup_indices_from_operands(&[&operand], false)
    }
}

/// Proves the (already-registered, by the Execution stage) clamped advice claim
/// equals `ActivationClampTable[raw_input]`.
fn prove_activation_clamp<F, T>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)>
where
    F: JoltField,
    T: Transcript,
{
    let mut results = Vec::new();

    let provider: OpLookupProvider<ActivationClampOperands> = OpLookupProvider::new(node.clone());
    let (mut exec_sumcheck, lookup_indices) = provider
        .read_raf_prove::<F, T, ActivationClampTable<XLEN>, XLEN>(
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
    let (exec_proof, _) = Sumcheck::prove(
        &mut exec_sumcheck,
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    results.push((ProofId(node.idx, ProofType::NeuralTeleport), exec_proof));

    let encoding = provider.encoding();
    let [ra_prover, hw_prover, bool_prover] = shout::ra_onehot_provers(
        &encoding,
        &lookup_indices,
        &prover.accumulator,
        &mut prover.transcript,
    );
    let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
        vec![ra_prover, hw_prover, bool_prover];
    let (ra_one_hot_proof, _) = BatchedSumcheck::prove(
        instances.iter_mut().map(|v| &mut **v as _).collect(),
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    results.push((ProofId(node.idx, ProofType::RaHammingWeight), ra_one_hot_proof));

    results
}

/// Verifier counterpart of [`prove_activation_clamp`]: verifies the clamp lookup
/// (against the advice claim the Execution stage already registered) + its one-hot checks.
fn verify_activation_clamp<F, T>(
    node: &ComputationNode,
    verifier: &mut Verifier<'_, F, T>,
) -> Result<(), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
{
    let provider: OpLookupProvider<ActivationClampOperands> = OpLookupProvider::new(node.clone());
    let verifier_sumcheck = provider.read_raf_verify::<F, T, ActivationClampTable<XLEN>, XLEN>(
        &mut verifier.accumulator,
        &mut verifier.transcript,
    );
    let exec_proof = verifier
        .proofs
        .get(&ProofId(node.idx, ProofType::NeuralTeleport))
        .ok_or(ProofVerifyError::MissingProof(node.idx))?;
    Sumcheck::verify(
        exec_proof,
        &verifier_sumcheck,
        &mut verifier.accumulator,
        &mut verifier.transcript,
    )?;

    let encoding = provider.encoding();
    let [ra_verifier, hw_verifier, bool_verifier] =
        shout::ra_onehot_verifiers(&encoding, &verifier.accumulator, &mut verifier.transcript);
    let ra_one_hot_proof = verifier
        .proofs
        .get(&ProofId(node.idx, ProofType::RaHammingWeight))
        .ok_or(ProofVerifyError::MissingProof(node.idx))?;
    BatchedSumcheck::verify(
        ra_one_hot_proof,
        vec![&*ra_verifier, &*hw_verifier, &*bool_verifier],
        &mut verifier.accumulator,
        &mut verifier.transcript,
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Stage: Execution -- small dense-table lookup (gamma-batched with an identity
// polynomial, mirroring `neural_teleport`'s Tanh execution stage).
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
        let (_, clamped_claim) =
            accumulator.get_virtual_polynomial_opening(clamped_opening_id(self.computation_node.idx));
        rv_claim + self.gamma * clamped_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        ACTIVATION_TABLE_BOUND
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
        let int_eval = SignedIdentityPoly::new(ACTIVATION_TABLE_BOUND).evaluate(&opening_point.r);
        vec![table_claim + self.gamma * int_eval]
    }
}

struct SmallTableProver<F: JoltField, Table> {
    params: SmallTableParams<F, Table>,
    table: MultilinearPolynomial<F>,
    input_onehot: MultilinearPolynomial<F>,
    identity: SignedIdentityPoly<F>,
    /// The clamped tensor, kept around so the caller can reuse it (e.g. to build the
    /// `RaOneHotChecks` lookup indices) without recomputing it from the trace again.
    clamped_tensor: Tensor<i32>,
}

impl<F: JoltField, Table: SmallActivationTable> SmallTableProver<F, Table> {
    /// Computes the clamped tensor (mirroring `atlas-onnx-tracer`'s `erf.rs`-style
    /// execution logic) and registers its advice claim at `r_node_output`
    /// (`VirtualPoly::ActivationClampedOutput`) -- the `ActivationClamp` stage, proven
    /// afterwards, soundly ties this same claim back to `raw_input`.
    fn initialize(
        params: SmallTableParams<F, Table>,
        trace: &Trace,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let LayerData { operands, .. } = Trace::layer_data(trace, &params.computation_node);
        let raw_input = operands[0];
        let clamped_tensor = nonlinearities::clamp(raw_input, ACTIVATION_BOUND);

        let clamped_claim = MultilinearPolynomial::from(
            clamped_tensor
                .padded_next_power_of_two()
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
            ACTIVATION_TABLE_BOUND,
        );
        let input_onehot = MultilinearPolynomial::from(input_onehot);
        assert_eq!(input_onehot.len(), table.len());
        let identity = SignedIdentityPoly::new(ACTIVATION_TABLE_BOUND);
        Self {
            params,
            table,
            input_onehot,
            identity,
            clamped_tensor,
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
                let id_evals = identity.sumcheck_evals(i, EXEC_DEGREE_BOUND, BindingOrder::LowToHigh);

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
        provider.append_advice(VirtualPoly::ActivationSmallRa, self.input_onehot.final_claim());
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
        let int_eval = SignedIdentityPoly::new(ACTIVATION_TABLE_BOUND).evaluate(&opening_point.r);

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

struct SmallTableRaEncoding {
    node_idx: usize,
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
        ACTIVATION_TABLE_BOUND
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), ACTIVATION_TABLE_BOUND)
    }
}

// ---------------------------------------------------------------------------
// Public entry points, called by the thin per-op `OperatorProofTrait` impls.
// ---------------------------------------------------------------------------

/// Full (Execution + ActivationClamp) proof for a clamped-activation node, proven in
/// that order: Execution establishes the clamped advice claim (from this node's own,
/// already-known output claim), which ActivationClamp then soundly ties back to the
/// raw input -- the reverse-topological node order's usual "claim about my input"
/// hand-off, just internal to this one fused node.
pub fn prove_clamped_activation<F, T, Table>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)>
where
    F: JoltField,
    T: Transcript,
    Table: SmallActivationTable,
{
    let mut results = Vec::new();

    let params =
        SmallTableParams::<F, Table>::new(node.clone(), &prover.accumulator, &mut prover.transcript);
    let mut exec_sumcheck = SmallTableProver::<F, Table>::initialize(
        params,
        &prover.trace,
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    let (exec_proof, _) = Sumcheck::prove(
        &mut exec_sumcheck,
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    results.push((ProofId(node.idx, ProofType::Execution), exec_proof));

    let lookup_indices: Vec<usize> = exec_sumcheck
        .clamped_tensor
        .padded_next_power_of_two()
        .par_iter()
        .with_min_len(par_enabled())
        .map(|&x| n_bits_to_usize(x, ACTIVATION_TABLE_BOUND))
        .collect();
    let encoding = SmallTableRaEncoding { node_idx: node.idx };
    let [ra_prover, hw_prover, bool_prover] = shout::ra_onehot_provers(
        &encoding,
        &lookup_indices,
        &prover.accumulator,
        &mut prover.transcript,
    );
    let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
        vec![ra_prover, hw_prover, bool_prover];
    let (ra_one_hot_proof, _) = BatchedSumcheck::prove(
        instances.iter_mut().map(|v| &mut **v as _).collect(),
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    results.push((
        ProofId(node.idx, ProofType::RaOneHotChecks),
        ra_one_hot_proof,
    ));

    results.extend(prove_activation_clamp(node, prover));

    results
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
    let exec_proof = verifier
        .proofs
        .get(&ProofId(node.idx, ProofType::Execution))
        .ok_or(ProofVerifyError::MissingProof(node.idx))?;
    let exec_verifier = SmallTableVerifier::<F, Table>::new(
        node.clone(),
        &mut verifier.accumulator,
        &mut verifier.transcript,
    );
    Sumcheck::verify(
        exec_proof,
        &exec_verifier,
        &mut verifier.accumulator,
        &mut verifier.transcript,
    )?;

    let encoding = SmallTableRaEncoding { node_idx: node.idx };
    let [ra_verifier, hw_verifier, bool_verifier] =
        shout::ra_onehot_verifiers(&encoding, &verifier.accumulator, &mut verifier.transcript);
    let ra_one_hot_proof = verifier
        .proofs
        .get(&ProofId(node.idx, ProofType::RaOneHotChecks))
        .ok_or(ProofVerifyError::MissingProof(node.idx))?;
    BatchedSumcheck::verify(
        ra_one_hot_proof,
        vec![&*ra_verifier, &*hw_verifier, &*bool_verifier],
        &mut verifier.accumulator,
        &mut verifier.transcript,
    )?;

    verify_activation_clamp(node, verifier)?;

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
