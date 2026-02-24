use crate::onnx_proof::{ops::OperatorProofTrait, ProofId, Prover, Verifier};
use atlas_onnx_tracer::{
    model::ComputationGraph,
    node::ComputationNode,
    ops::Erf,
};
use common::{CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    field::JoltField,
    poly::{
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck::SumcheckInstanceProof,
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Erf {
    fn prove(
        &self,
        _node: &ComputationNode,
        _prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        todo!("Implement Erf proving pipeline")
    }

    fn verify(
        &self,
        _node: &ComputationNode,
        _verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        todo!("Implement Erf verification pipeline")
    }

    fn get_committed_polynomials(&self, _node: &ComputationNode) -> Vec<CommittedPolynomial> {
        todo!("Implement Erf committed polynomial list")
    }
}

const DEGREE_BOUND: usize = 2; // TODO

/// Parameters for proving error function (erf) activation operations.
///
/// Mirrors the parameter layout used by `TanhParams` so both lookup-style
/// activations follow the same transcript/challenge flow.
#[derive(Clone)]
pub struct ErfParams<F: JoltField> {
    /// Folding challenge used to combine multiple checks into one claim.
    pub gamma: F,
    /// Opening point sampled from the output claim of this node.
    pub r_node_output: Vec<F::Challenge>,
    /// Computation node currently being proven.
    pub computation_node: ComputationNode,
    /// Operator parameters (e.g. fixed-point scale for erf).
    pub op: Erf,
    /// Phantom marker for the field type.
    pub _marker: core::marker::PhantomData<F>,
}

impl<F: JoltField> ErfParams<F> {
    /// Build erf parameters from the current accumulator/transcript state.
    ///
    /// This samples a fresh folding challenge and reuses the node-output opening
    /// point already present in the accumulator, matching the tanh flow.
    pub fn new(
        computation_node: ComputationNode,
        _graph: &ComputationGraph,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
        op: Erf,
    ) -> Self {
        let gamma = transcript.challenge_scalar();
        let r_node_output = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(computation_node.idx),
                SumcheckId::Execution,
            )
            .0
            .r;

        Self {
            gamma,
            r_node_output,
            computation_node,
            op,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ErfParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let rv_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;

        let quotient_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::TeleportQuotient(self.computation_node.idx),
                SumcheckId::Raf,
            )
            .1;

        rv_claim + self.gamma * quotient_claim
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.op.log_table
    }
}

pub struct ErfProver<F: JoltField> {
    pub params: ErfParams<F>,
}

impl<F: JoltField> ErfProver<F> {
    pub fn initialize(
        _params: ErfParams<F>,
        _accumulator: &mut ProverOpeningAccumulator<F>,
        _transcript: &mut impl Transcript,
    ) -> Self {
        todo!("Initialize Erf prover state")
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ErfProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UniPoly<F> {
        todo!("Compute Erf prover univariate message")
    }

    fn ingest_challenge(&mut self, _r_j: F::Challenge, _round: usize) {
        todo!("Bind Erf prover polynomials to verifier challenge")
    }

    fn cache_openings(
        &self,
        _accumulator: &mut ProverOpeningAccumulator<F>,
        _transcript: &mut T,
        _sumcheck_challenges: &[F::Challenge],
    ) {
        todo!("Cache Erf prover virtual openings")
    }
}

pub struct ErfVerifier<F: JoltField> {
    pub params: ErfParams<F>,
}

impl<F: JoltField> ErfVerifier<F> {
    pub fn new(
        _computation_node: ComputationNode,
        _accumulator: &mut VerifierOpeningAccumulator<F>,
        _transcript: &mut impl Transcript,
    ) -> Self {
        todo!("Initialize Erf verifier state")
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for ErfVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        _accumulator: &VerifierOpeningAccumulator<F>,
        _sumcheck_challenges: &[F::Challenge],
    ) -> F {
        todo!("Compute expected Erf output claim")
    }

    fn cache_openings(
        &self,
        _accumulator: &mut VerifierOpeningAccumulator<F>,
        _transcript: &mut T,
        _sumcheck_challenges: &[F::Challenge],
    ) {
        todo!("Cache Erf verifier virtual openings")
    }
}
