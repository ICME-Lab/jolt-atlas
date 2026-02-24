use crate::onnx_proof::{ops::OperatorProofTrait, ProofId, Prover, Verifier};
use atlas_onnx_tracer::{node::ComputationNode, ops::Erf};
use common::CommittedPolynomial;
use joltworks::{
    field::JoltField,
    poly::{
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator,
            VerifierOpeningAccumulator, BIG_ENDIAN,
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

#[derive(Clone)]
pub struct ErfParams<F: JoltField> {
    pub computation_node: ComputationNode,
    pub _marker: core::marker::PhantomData<F>,
}

impl<F: JoltField> ErfParams<F> {
    pub fn new(_computation_node: ComputationNode, _accumulator: &dyn OpeningAccumulator<F>) -> Self {
        todo!("Initialize Erf params from accumulator challenges")
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ErfParams<F> {
    fn degree(&self) -> usize {
        todo!("Define sumcheck degree for Erf")
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        todo!("Define folded input claim for Erf")
    }

    fn normalize_opening_point(&self, _challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        todo!("Define opening-point normalization for Erf")
    }

    fn num_rounds(&self) -> usize {
        todo!("Define number of sumcheck rounds for Erf")
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
