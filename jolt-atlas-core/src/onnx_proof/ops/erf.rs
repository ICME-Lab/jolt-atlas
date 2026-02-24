use crate::onnx_proof::{ProofId, Prover, Verifier, ops::{OperatorProofTrait, tanh::compute_ra_evals}};
use atlas_onnx_tracer::{
    model::{
        trace::{LayerData, Trace},
        ComputationGraph,
    },
    node::ComputationNode,
    tensor::Tensor,
    ops::Erf,
};
use common::{CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        teleport_id_poly::TeleportIdPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck::SumcheckInstanceProof,
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, thread::unsafe_allocate_zero_vec},
};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator},
    slice::ParallelSlice,
};
use crate::onnx_proof::neural_teleport::{division::compute_division, erf::ErfTable, n_bits_to_usize};

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


/// Prover state for erf activation sumcheck protocol.
///
/// This implements a Read-Raf sumcheck for Erf lookup, asserting that each output[i] = ErfTable[input[i]]
/// where input[i] = Ra[k] * Int[k] with Ra as one-hot encoding and Int as custom identity polynomial.
pub struct ErfProver<F: JoltField> {
    pub params: ErfParams<F>,
    pub erf_table: MultilinearPolynomial<F>,
    pub input_onehot: MultilinearPolynomial<F>,
    pub identity: TeleportIdPolynomial<F>,
}

impl<F: JoltField> ErfProver<F> {
  /// Initialize the prover with trace data, parameters, accumulator, and transcript.
    pub fn initialize(
        trace: &Trace,
        params: ErfParams<F>,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let LayerData { operands, .. } = Trace::layer_data(trace, &params.computation_node);
        let input = operands[0];

        let (quotient_tensor, _remainder) = compute_division(input, params.op.tau);

        // Ensure input is within expected range for table size: 2^(log_table_size - 1) <= input < 2^(log_table_size - 1)
        // Inputs outside this range will error
        // TODO: Pass these input in a clamping lookup table, since anyway tanh(±∞) = ±1, so we only need to handle a limited input range.
        assert!(quotient_tensor.iter().all(|&x| {
            let lower_bound = -(1 << (params.op.log_table - 1));
            let upper_bound = (1 << (params.op.log_table - 1)) - 1;
            x >= lower_bound && x <= upper_bound
        }));

        // Create and materialize the erf lookup table (reduced size)
        let erf_table = ErfTable::new(params.op.log_table);
        let erf_table = MultilinearPolynomial::from(erf_table.materialize());

        // Use the compute_ra_evals in tanh.rs
        let input_onehot: Vec<F> =
            compute_ra_evals(&params.r_node_output, &quotient_tensor, params.op.log_table);

        // TODO(ClankPan): Follow up on the TODOs in tanh.rs.
        let quotient_claim = MultilinearPolynomial::from(quotient_tensor.into_container_data())
            .evaluate(&params.r_node_output);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::TeleportQuotient(params.computation_node.idx),
            SumcheckId::Raf,
            params.r_node_output.clone().into(),
            quotient_claim,
        );

        let input_onehot = MultilinearPolynomial::from(input_onehot);
        assert_eq!(input_onehot.len(), erf_table.len());
        let identity = TeleportIdPolynomial::new(params.op.log_table);

        Self {
            params,
            erf_table,
            input_onehot,
            identity,
        }
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

// fn compute_ra_evals<F>(r: &[F::Challenge], input: &Tensor<i32>, log_table_size: usize) -> Vec<F>
// where
//     F: JoltField,
// {
//     let E = EqPolynomial::evals(r);
//     let num_threads = rayon::current_num_threads();
//     let chunk_size = input.len().div_ceil(num_threads);

//     let table_size = 1 << log_table_size;
//     let input_usize = input
//         .par_iter()
//         .map(|&x| n_bits_to_usize(x, log_table_size))
//         .collect::<Vec<usize>>();

//     let partial_results: Vec<Vec<F>> = input_usize
//         .par_chunks(chunk_size)
//         .enumerate()
//         .map(|(chunk_idx, chunk)| {
//             let mut local_ra = unsafe_allocate_zero_vec::<F>(table_size);
//             let base_idx = chunk_idx * chunk_size;
//             chunk.iter().enumerate().for_each(|(local_j, &k)| {
//                 let global_j = base_idx + local_j;
//                 local_ra[k] += E[global_j];
//             });
//             local_ra
//         })
//         .collect();

//     let mut ra = unsafe_allocate_zero_vec::<F>(table_size);
//     for partial in partial_results {
//         ra.par_iter_mut()
//             .zip(partial.par_iter())
//             .for_each(|(dest, &src)| *dest += src);
//     }
//     ra
// }

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
