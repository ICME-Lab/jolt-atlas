//! Softmax verification for softmax(axes = -1).
//!
//! This module verifies the softmax computation over the last dimension, which corresponds to
//! ∏(prev_dims) standard softmax operations (i.e., one per feature vector).
//!
//! # Verification Approach
//!
//! For each feature vector, we verify the standard softmax computation in stages:
//!
//! ## 1. Division / Normalization
//! We first verify the division step of softmax, as the division output yields the softmax output.
//! The sum and max checks are batched together with division:
//! - The sum and max tensors are scalars
//! - They do not need to be virtualized, since evaluating them at any point always yields the same value
//!
//! This allows them to be batched with div for efficiency.
//!
//! ## 2. Exponentiation
//! Exponentiation is verified using a small LUT, following the approach used in zkGPT
//! (except we use shout instead of lasso).
//!
//! ## 3. Operand Consistency
//! We constrain the softmax operand to be derived from the exponentiation raf claim by verifier checking:
//! ```text
//! softmax_operand_claim = (-raf_claim) + max_logit
//! ```
//! This also implies that the max value is greater than or equal to the other operand values,
//! so no additional max-comparison constraint is required.
//!
//! ## 4. Linking to Main Operand Claim
//! Finally, we link each per-feature softmax operand claim back to the main operand claim.

use crate::onnx_proof::{
    ops::{
        softmax_axes::softmax::{
            exponentiation::{self, SoftmaxExpRaEncoding},
            max,
            scalar_div::{DivParams, DivProver, DivVerifier},
            sum::{SumParams, SumProver, SumVerifier},
            SoftmaxIndex,
        },
        OperatorProofTrait, Prover, Verifier,
    },
    ProofId, ProofType,
};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::SoftmaxAxes,
    tensor::{
        ops::nonlinearities::softmax_fixed_128,
        Tensor,
    },
};
use common::{CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{
            OpeningAccumulator, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
        },
    },
    subprotocols::{
        shout::{self, RaOneHotEncoding},
        sumcheck::{BatchedSumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math, thread::drop_in_background_thread},
};

pub mod softmax;

#[derive(Clone)]
pub struct SoftmaxLastAxisConfig {
    num_feature_vectors: usize,
    feature_vector_size: usize,
}

impl SoftmaxLastAxisConfig {
    pub fn new(computation_node: &ComputationNode) -> Self {
        let (&feature_vector_size, dims) = computation_node
            .output_dims
            .split_last()
            .expect("Softmax output dims should not be empty");
        Self {
            feature_vector_size,
            num_feature_vectors: dims.iter().product(),
        }
    }
}

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for SoftmaxAxes {
    #[tracing::instrument(skip_all, name = "SoftmaxAxes::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let params = SoftmaxAxesParams::<F>::new(node.clone(), &prover.accumulator);
        let softmax_prover = SoftmaxAxesProver::initialize(&prover.trace, params);
        softmax_prover.prove(&mut prover.accumulator, &mut prover.transcript)
    }

    #[tracing::instrument(skip_all, name = "SoftmaxAxes::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let softmax_verifier = SoftmaxAxesVerifier::new(node.clone(), &verifier.accumulator);
        let div_sum_max_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::SoftmaxDivSumMax))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let exponentiation_read_raf_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::SoftmaxExponentiationReadRaf))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let exponentiation_ra_onehot_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::SoftmaxExponentiationRaOneHot))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        softmax_verifier.verify(
            div_sum_max_proof,
            exponentiation_read_raf_proof,
            exponentiation_ra_onehot_proof,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )
    }

    fn get_committed_polynomials(
        &self,
        node: &ComputationNode,
    ) -> Vec<common::CommittedPolynomial> {
        let mut polys = vec![];
        let config = SoftmaxLastAxisConfig::new(node);
        let encoding = SoftmaxExpRaEncoding {
            softmax_index: SoftmaxIndex {
                node_idx: node.idx,
                feature_idx: 0,
            },
        };
        let d = encoding.one_hot_params().instruction_d;
        for feature_idx in 0..config.num_feature_vectors {
            polys.push(CommittedPolynomial::SoftmaxRemainder(node.idx, feature_idx));
            (0..d).for_each(|i| {
                polys.push(CommittedPolynomial::SoftmaxExponentiationRaD(
                    node.idx,
                    feature_idx,
                    i,
                ))
            });
        }
        polys
    }
}

#[derive(Clone)]
pub struct SoftmaxAxesParams<F: JoltField> {
    r_output: Vec<F::Challenge>,
    computation_node: ComputationNode,
    config: SoftmaxLastAxisConfig,
}

impl<F: JoltField> SoftmaxAxesParams<F> {
    pub fn new(computation_node: ComputationNode, accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let r_output = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(computation_node.idx),
                SumcheckId::Execution,
            )
            .0
            .r;
        Self {
            r_output,
            config: SoftmaxLastAxisConfig::new(&computation_node),
            computation_node,
        }
    }

    pub fn r_features(&self) -> &[F::Challenge] {
        &self.r_output[self.config.num_feature_vectors.log_2()..]
    }

    pub fn r_leading_dims(&self) -> &[F::Challenge] {
        &self.r_output[..self.config.num_feature_vectors.log_2()]
    }

    pub fn num_feature_vectors(&self) -> usize {
        self.config.num_feature_vectors
    }

    pub fn feature_vector_size(&self) -> usize {
        self.config.feature_vector_size
    }
}

pub struct SoftmaxAxesProver<F: JoltField> {
    params: SoftmaxAxesParams<F>,
    output: Tensor<i32>,
    operand: Tensor<i32>,
}

impl<F: JoltField> SoftmaxAxesProver<F> {
    pub fn initialize(trace: &Trace, params: SoftmaxAxesParams<F>) -> Self {
        let LayerData { output, operands } = Trace::layer_data(trace, &params.computation_node);
        Self {
            params,
            output: output.clone(),
            operand: operands[0].clone(),
        }
    }

    #[tracing::instrument(skip_all, name = "SoftmaxAxesProver::prove")]
    pub fn prove<T: Transcript>(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        self.prove_outer(accumulator, transcript);

        let cached_traces = self.generate_trace_cache();

        let div_sum_max_proof = self.prove_div_sum_max(&cached_traces, accumulator, transcript);

        let (exponentiation_read_raf_proof, exponentiation_ra_onehot_proof) =
            self.prove_exponentiation(&cached_traces, accumulator, transcript);

        self.prove_operand_claims(accumulator, transcript);

        vec![
            (
                ProofId(
                    self.params.computation_node.idx,
                    ProofType::SoftmaxDivSumMax,
                ),
                div_sum_max_proof,
            ),
            (
                ProofId(
                    self.params.computation_node.idx,
                    ProofType::SoftmaxExponentiationReadRaf,
                ),
                exponentiation_read_raf_proof,
            ),
            (
                ProofId(
                    self.params.computation_node.idx,
                    ProofType::SoftmaxExponentiationRaOneHot,
                ),
                exponentiation_ra_onehot_proof,
            ),
        ]
    }

    #[tracing::instrument(skip_all)]
    /// Outer proof is implicit in the way we construct the claims for the feature outputs
    fn prove_outer<T: Transcript>(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) {
        // Iterate over each (head, sequence) pair
        for (feature_idx, output_chunk) in self
            .output
            .data()
            .chunks_exact(self.params.feature_vector_size())
            .enumerate()
        {
            let feature_claim = MultilinearPolynomial::from(output_chunk.to_vec())
                .evaluate(self.params.r_features());
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::SoftmaxFeatureOutput(
                    self.params.computation_node.idx,
                    feature_idx,
                ),
                SumcheckId::Execution,
                self.params.r_features().to_vec().into(),
                feature_claim,
            );
        }
    }

    #[tracing::instrument(skip_all)]
    fn generate_trace_cache(
        &self,
    ) -> Vec<atlas_onnx_tracer::tensor::ops::nonlinearities::SoftmaxTrace> {
        let mut cached_traces = Vec::new();
        for operand_chunk in self
            .operand
            .data()
            .chunks_exact(self.params.feature_vector_size())
        {
            let (_, trace) = softmax_fixed_128::<true>(&Tensor::construct(
                operand_chunk.to_vec(),
                vec![self.params.feature_vector_size()],
            ));
            let trace = trace.expect("Softmax trace should be present");
            cached_traces.push(trace);
        }
        cached_traces
    }

    #[tracing::instrument(skip_all)]
    /// Stage 1: Division / Normalization
    ///
    /// Verifies the division step of softmax along with sum and max checks.
    /// Since sum and max are scalar values, they can be batched with division
    /// for efficient verification.
    fn prove_div_sum_max<T: Transcript>(
        &self,
        cached_traces: &[atlas_onnx_tracer::tensor::ops::nonlinearities::SoftmaxTrace],
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> SumcheckInstanceProof<F, T> {
        let mut div_sum_max_instances: Vec<Box<dyn SumcheckInstanceProver<F, _>>> = vec![];

        // Iterate over each (head, sequence) pair (i.e., each feature vector)
        for (feature_idx, trace) in cached_traces.iter().enumerate() {
            let softmax_index = SoftmaxIndex {
                node_idx: self.params.computation_node.idx,
                feature_idx,
            };

            // Send verifier sum claim
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::SoftmaxSumOutput(
                    softmax_index.node_idx,
                    softmax_index.feature_idx,
                ),
                SumcheckId::Execution,
                vec![].into(),
                F::from_i32(trace.exp_sum_q),
            );

            // Send max value and index claims
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::SoftmaxMaxOutput(
                    softmax_index.node_idx,
                    softmax_index.feature_idx,
                ),
                SumcheckId::Execution,
                vec![].into(),
                F::from_i32(trace.max_logit),
            );
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::SoftmaxMaxIndex(
                    softmax_index.node_idx,
                    softmax_index.feature_idx,
                ),
                SumcheckId::Execution,
                vec![].into(),
                F::from_u32(trace.max_index as u32),
            );

            // Construct div/sum/max prover and proof
            let div_params: DivParams<F> = DivParams::new(softmax_index, accumulator);
            let div_prover_sumcheck = DivProver::initialize(trace, div_params);
            let sum_params = SumParams::new(softmax_index, accumulator);
            let sum_prover_sumcheck = SumProver::initialize(trace, sum_params);
            let max_params = max::IndicatorParams::new(softmax_index, accumulator);
            let max_prover_sumcheck = max::IndicatorProver::initialize(trace, max_params);

            div_sum_max_instances.push(Box::new(div_prover_sumcheck));
            div_sum_max_instances.push(Box::new(sum_prover_sumcheck));
            div_sum_max_instances.push(Box::new(max_prover_sumcheck));
        }

        let (div_sum_max_proof, _) = BatchedSumcheck::prove(
            div_sum_max_instances
                .iter_mut()
                .map(|v| &mut **v as _)
                .collect(),
            accumulator,
            transcript,
        );
        div_sum_max_proof
    }

    #[tracing::instrument(skip_all)]
    /// Stage 2: Exponentiation
    ///
    /// Verifies exponentiation using a small LUT approach (similar to zkGPT but using shout instead of lasso).
    fn prove_exponentiation<T: Transcript>(
        &self,
        cached_traces: &[atlas_onnx_tracer::tensor::ops::nonlinearities::SoftmaxTrace],
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> (SumcheckInstanceProof<F, T>, SumcheckInstanceProof<F, T>) {
        let mut read_raf_instances: Vec<Box<dyn SumcheckInstanceProver<F, _>>> = vec![];

        // Iterate over each (head, sequence) pair (i.e., each feature vector)
        for (feature_idx, trace) in cached_traces.iter().enumerate() {
            let softmax_index = SoftmaxIndex {
                node_idx: self.params.computation_node.idx,
                feature_idx,
            };
            let read_raf_params: exponentiation::ReadRafParams<F> =
                exponentiation::ReadRafParams::new(softmax_index, accumulator, transcript);
            let read_raf_prover_sumcheck = exponentiation::ReadRafProver::initialize(
                trace,
                read_raf_params,
                accumulator,
                transcript,
            );

            read_raf_instances.push(Box::new(read_raf_prover_sumcheck));
        }

        let (read_raf_proof, _) = BatchedSumcheck::prove(
            read_raf_instances
                .iter_mut()
                .map(|v| &mut **v as _)
                .collect(),
            accumulator,
            transcript,
        );
        drop_in_background_thread(read_raf_instances);

        let mut ra_onehot_instances: Vec<Box<dyn SumcheckInstanceProver<F, _>>> = vec![];
        for (feature_idx, trace) in cached_traces.iter().enumerate() {
            let softmax_index = SoftmaxIndex {
                node_idx: self.params.computation_node.idx,
                feature_idx,
            };
            let encoding = SoftmaxExpRaEncoding { softmax_index };
            let lookup_indices: Vec<usize> = trace
                .abs_centered_logits
                .data()
                .iter()
                .map(|v| *v as usize)
                .collect();
            let ra_onehot_sumchecks =
                shout::ra_onehot_provers(&encoding, &lookup_indices, accumulator, transcript);

            ra_onehot_instances.extend(ra_onehot_sumchecks);
        }
        let (ra_onehot_proof, _) = BatchedSumcheck::prove(
            ra_onehot_instances
                .iter_mut()
                .map(|v| &mut **v as _)
                .collect(),
            accumulator,
            transcript,
        );

        (read_raf_proof, ra_onehot_proof)
    }

    #[tracing::instrument(skip_all)]
    /// Stage 4: Linking to Main Operand Claim
    ///
    /// Links each per-feature softmax operand claim back to the main operand claim.
    fn prove_operand_claims<T: Transcript>(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) {
        let r_leading_dims_prime =
            transcript.challenge_vector_optimized::<F>(self.params.num_feature_vectors().log_2());
        let (r_features_prime, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::SoftmaxInputLogitsOutput(self.params.computation_node.idx, 0),
            SumcheckId::Execution,
        );
        let r_operand: Vec<F::Challenge> =
            [r_leading_dims_prime.as_slice(), &r_features_prime.r].concat();
        let operand_claim = MultilinearPolynomial::from(self.operand.clone()).evaluate(&r_operand); // TODO: rm clone
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            r_operand.clone().into(),
            operand_claim,
        );
        accumulator.cache_virtual_operand_claims(transcript, &self.params.computation_node);
    }
}

pub struct SoftmaxAxesVerifier<F: JoltField> {
    params: SoftmaxAxesParams<F>,
}

impl<F: JoltField> SoftmaxAxesVerifier<F> {
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = SoftmaxAxesParams::new(computation_node, accumulator);
        Self { params }
    }

    #[tracing::instrument(skip_all, name = "SoftmaxAxesVerifier::verify")]
    pub fn verify<ProofTranscript: Transcript>(
        &self,
        div_sum_max_proof: &SumcheckInstanceProof<F, ProofTranscript>,
        exponentiation_read_raf_proof: &SumcheckInstanceProof<F, ProofTranscript>,
        exponentiation_ra_onehot_proof: &SumcheckInstanceProof<F, ProofTranscript>,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        self.verify_outer(accumulator, transcript)?;

        self.verify_div_sum_max(div_sum_max_proof, accumulator, transcript)?;

        self.verify_exponentiation(
            exponentiation_read_raf_proof,
            exponentiation_ra_onehot_proof,
            accumulator,
            transcript,
        )?;

        self.verify_operand_claims(accumulator, transcript)?;

        Ok(())
    }

    #[tracing::instrument(skip_all)]
    fn verify_outer(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Result<(), ProofVerifyError> {
        let mut softmax_features_claim = Vec::with_capacity(self.params.num_feature_vectors());
        // Iterate over each (head, sequence) pair
        for feature_idx in 0..self.params.num_feature_vectors() {
            // implicitly add feature output claims to transcript
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::SoftmaxFeatureOutput(
                    self.params.computation_node.idx,
                    feature_idx,
                ),
                SumcheckId::Execution,
                self.params.r_features().to_vec().into(),
            );
            softmax_features_claim.push(
                accumulator
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::SoftmaxFeatureOutput(
                            self.params.computation_node.idx,
                            feature_idx,
                        ),
                        SumcheckId::Execution,
                    )
                    .1,
            );
        }

        let expected_softmax_axes_output_claim =
            MultilinearPolynomial::from(softmax_features_claim)
                .evaluate(self.params.r_leading_dims());

        let softmax_axes_output_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;

        if expected_softmax_axes_output_claim != softmax_axes_output_claim {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Softmax axes output claim does not match expected claim".to_string(),
            ));
        }
        Ok(())
    }

    #[tracing::instrument(skip_all)]
    /// Stage 1 (Verifier): Division / Normalization
    ///
    /// Verifies the division step proof along with sum and max checks.
    fn verify_div_sum_max<T: Transcript>(
        &self,
        div_sum_max_proof: &SumcheckInstanceProof<F, T>,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<(), ProofVerifyError> {
        let mut div_sum_max_instances: Vec<Box<dyn SumcheckInstanceVerifier<F, _>>> = vec![];
        for feature_idx in 0..self.params.num_feature_vectors() {
            let softmax_index = SoftmaxIndex {
                node_idx: self.params.computation_node.idx,
                feature_idx,
            };

            // Add verifier opening points (and implicitly append claims to transcript)
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::SoftmaxSumOutput(
                    softmax_index.node_idx,
                    softmax_index.feature_idx,
                ),
                SumcheckId::Execution,
                vec![].into(),
            );
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::SoftmaxMaxOutput(
                    softmax_index.node_idx,
                    softmax_index.feature_idx,
                ),
                SumcheckId::Execution,
                vec![].into(),
            );
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::SoftmaxMaxIndex(
                    softmax_index.node_idx,
                    softmax_index.feature_idx,
                ),
                SumcheckId::Execution,
                vec![].into(),
            );
            let div_verifier_sumcheck = DivVerifier::new(softmax_index, accumulator);
            let sum_verifier_sumcheck = SumVerifier::new(softmax_index, accumulator);
            let max_verifier_sumcheck = max::IndicatorVerifier::new(softmax_index, accumulator);
            div_sum_max_instances.push(Box::new(div_verifier_sumcheck));
            div_sum_max_instances.push(Box::new(sum_verifier_sumcheck));
            div_sum_max_instances.push(Box::new(max_verifier_sumcheck));
        }
        BatchedSumcheck::verify(
            div_sum_max_proof,
            div_sum_max_instances.iter().map(|v| &**v as _).collect(),
            accumulator,
            transcript,
        )?;
        Ok(())
    }

    #[tracing::instrument(skip_all)]
    /// Stage 2 (Verifier): Exponentiation
    ///
    /// Verifies the exponentiation proof using LUT approach.
    fn verify_exponentiation<T: Transcript>(
        &self,
        read_raf_proof: &SumcheckInstanceProof<F, T>,
        ra_one_hot_proof: &SumcheckInstanceProof<F, T>,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<(), ProofVerifyError> {
        let mut read_raf_instances: Vec<Box<dyn SumcheckInstanceVerifier<F, _>>> = vec![];
        for feature_idx in 0..self.params.num_feature_vectors() {
            let softmax_index = SoftmaxIndex {
                node_idx: self.params.computation_node.idx,
                feature_idx,
            };
            let exponentiation_verifier_sumcheck =
                exponentiation::ReadRafVerifier::new(softmax_index, accumulator, transcript);
            read_raf_instances.push(Box::new(exponentiation_verifier_sumcheck));
        }
        BatchedSumcheck::verify(
            read_raf_proof,
            read_raf_instances.iter().map(|v| &**v as _).collect(),
            accumulator,
            transcript,
        )?;

        let mut ra_one_hot_instances: Vec<Box<dyn SumcheckInstanceVerifier<F, _>>> = vec![];
        for feature_idx in 0..self.params.num_feature_vectors() {
            let softmax_index = SoftmaxIndex {
                node_idx: self.params.computation_node.idx,
                feature_idx,
            };
            let encoding = SoftmaxExpRaEncoding { softmax_index };
            let ra_one_hot_sumchecks =
                shout::ra_onehot_verifiers(&encoding, accumulator, transcript);
            ra_one_hot_instances.extend(ra_one_hot_sumchecks);
        }
        BatchedSumcheck::verify(
            ra_one_hot_proof,
            ra_one_hot_instances.iter().map(|v| &**v as _).collect(),
            accumulator,
            transcript,
        )?;
        Ok(())
    }

    #[tracing::instrument(skip_all)]
    /// Stages 3 & 4 (Verifier): Operand Consistency and Linking
    ///
    /// Stage 3: Verifies that softmax_operand_claim = (-raf_claim) + max_logit.
    /// This constraint also implies max value >= other operand values.
    ///
    /// Stage 4: Links each per-feature softmax operand claim back to the main operand claim.
    fn verify_operand_claims<T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<(), ProofVerifyError> {
        let mut softmax_operand_claims = vec![];
        for feature_idx in 0..self.params.num_feature_vectors() {
            let softmax_index = SoftmaxIndex {
                node_idx: self.params.computation_node.idx,
                feature_idx,
            };
            // Stage 3: Check operand consistency - verify softmax_operand_claim = (-raf_claim) + max_logit
            let (_, abs_centered_logits_claim) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxAbsCenteredLogitsOutput(
                    softmax_index.node_idx,
                    softmax_index.feature_idx,
                ),
                SumcheckId::Execution,
            );
            let (_, softmax_operand_claim) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxInputLogitsOutput(
                    softmax_index.node_idx,
                    softmax_index.feature_idx,
                ),
                SumcheckId::Execution,
            );
            let (_, max_logit) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxMaxOutput(
                    softmax_index.node_idx,
                    softmax_index.feature_idx,
                ),
                SumcheckId::Execution,
            );
            if softmax_operand_claim != (-abs_centered_logits_claim) + max_logit {
                return Err(ProofVerifyError::InvalidOpeningProof(
                    "Softmax operand claim does not match expected claim from raf".to_string(),
                ));
            }
            softmax_operand_claims.push(softmax_operand_claim);
        }

        // Stage 4: Link per-feature softmax operand claims to main operand claim
        let r_leading_dims_prime =
            transcript.challenge_vector_optimized::<F>(self.params.r_leading_dims().len());
        let (r_features_prime, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::SoftmaxInputLogitsOutput(self.params.computation_node.idx, 0),
            SumcheckId::Execution,
        );
        let r_operand: Vec<F::Challenge> =
            [r_leading_dims_prime.as_slice(), &r_features_prime.r].concat();

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            r_operand.into(),
        );
        accumulator.append_operand_claims(transcript, self.params.computation_node.idx);
        let [operand_claim] = accumulator.get_operand_claims::<1>(self.params.computation_node.idx);

        let expected_operand_claim =
            MultilinearPolynomial::from(softmax_operand_claims).evaluate(&r_leading_dims_prime);

        if operand_claim != expected_operand_claim {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Softmax operand claim does not match expected claim".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use atlas_onnx_tracer::{
        model::{
            self,
            trace::{LayerData, Trace},
        },
        tensor::Tensor,
    };
    use common::VirtualPolynomial;
    use joltworks::{
        field::JoltField,
        poly::{
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
            opening_proof::{
                OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
                BIG_ENDIAN,
            },
        },
        transcripts::{Blake2bTranscript, Transcript},
    };
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_softmax_axes() {
        let input_shape = vec![4, 64, 64];
        let T: usize = input_shape.iter().product();
        let log_T = T.log_2();
        let mut rng = StdRng::seed_from_u64(0x858);
        let input = Tensor::<i32>::random_small(&mut rng, &input_shape);
        let model = model::test::softmax_axes_model(&input_shape, 2);
        let trace = model.trace(&[input.clone()]);

        let prover_transcript = &mut Blake2bTranscript::new(&[]);
        let mut prover_opening_accumulator: ProverOpeningAccumulator<Fr> =
            ProverOpeningAccumulator::new();

        let r_node_output: Vec<<Fr as JoltField>::Challenge> =
            prover_transcript.challenge_vector_optimized::<Fr>(log_T);

        let output_index = model.outputs()[0];
        let computation_node = &model[output_index];
        let LayerData {
            operands: _,
            output,
        } = Trace::layer_data(&trace, computation_node);

        let softmax_claim = MultilinearPolynomial::from(output.clone()).evaluate(&r_node_output);
        prover_opening_accumulator.append_virtual(
            prover_transcript,
            VirtualPolynomial::NodeOutput(output_index),
            SumcheckId::Execution,
            r_node_output.clone().into(),
            softmax_claim,
        );

        let params: SoftmaxAxesParams<Fr> =
            SoftmaxAxesParams::new(computation_node.clone(), &prover_opening_accumulator);
        let softmax_axes_prover = SoftmaxAxesProver::initialize(&trace, params);

        let proofs = softmax_axes_prover.prove(&mut prover_opening_accumulator, prover_transcript);

        let verifier_transcript = &mut Blake2bTranscript::new(&[]);
        verifier_transcript.compare_to(prover_transcript.clone());
        let mut verifier_opening_accumulator: VerifierOpeningAccumulator<Fr> =
            VerifierOpeningAccumulator::new();
        let _r_node_output: Vec<<Fr as JoltField>::Challenge> =
            verifier_transcript.challenge_vector_optimized::<Fr>(log_T);
        // Take claims
        for (key, (_, value)) in &prover_opening_accumulator.openings {
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
            verifier_opening_accumulator
                .openings
                .insert(*key, (empty_point, *value));
        }
        verifier_opening_accumulator.virtual_operand_claims =
            prover_opening_accumulator.virtual_operand_claims.clone();

        verifier_opening_accumulator.append_virtual(
            verifier_transcript,
            VirtualPolynomial::NodeOutput(output_index),
            SumcheckId::Execution,
            r_node_output.into(),
        );

        let verifier_softmax_axes =
            SoftmaxAxesVerifier::new(computation_node.clone(), &verifier_opening_accumulator);
        let (div_sum_max_proof, exponentiation_read_raf_proof, exponentiation_ra_onehot_proof) =
            match &proofs[..] {
                [(ProofId(_, ProofType::SoftmaxDivSumMax), div_sum_max_proof), (
                    ProofId(_, ProofType::SoftmaxExponentiationReadRaf),
                    exponentiation_read_raf_proof,
                ), (
                    ProofId(_, ProofType::SoftmaxExponentiationRaOneHot),
                    exponentiation_ra_onehot_proof,
                )] => (
                    div_sum_max_proof,
                    exponentiation_read_raf_proof,
                    exponentiation_ra_onehot_proof,
                ),
                _ => panic!("Unexpected proof structure"),
            };
        verifier_softmax_axes
            .verify(
                div_sum_max_proof,
                exponentiation_read_raf_proof,
                exponentiation_ra_onehot_proof,
                &mut verifier_opening_accumulator,
                verifier_transcript,
            )
            .expect("SoftmaxAxes verification failed");

        // check input claim
        let (r_input, input_claim) = verifier_opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NodeOutput(computation_node.inputs[0]),
            SumcheckId::Execution,
        );
        let expected_input_claim = MultilinearPolynomial::from(input.clone()).evaluate(&r_input.r);
        assert_eq!(input_claim, expected_input_claim);
    }
}
