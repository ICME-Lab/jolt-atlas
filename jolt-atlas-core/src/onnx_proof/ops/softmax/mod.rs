use crate::onnx_proof::{
    ops::{OperatorProofTrait, Prover, Verifier},
    ProofId,
};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::Softmax,
    tensor::Tensor,
};
use common::VirtualPolynomial;
use joltworks::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{
            OpeningAccumulator, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
        },
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Softmax {
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let params = SoftmaxParams::<F>::new(node.clone(), &prover.accumulator);
        let softmax_prover = SoftmaxProver::initialize(&prover.trace, params);
        softmax_prover.prove(&mut prover.accumulator, &mut prover.transcript);
        vec![]
    }

    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let softmax_verifier = SoftmaxVerifier::new(node.clone(), &verifier.accumulator);
        softmax_verifier.verify(&mut verifier.accumulator, &mut verifier.transcript)
    }
}

#[derive(Clone)]
pub struct SoftmaxParams<F: JoltField> {
    r_output: Vec<F::Challenge>,
    computation_node: ComputationNode,
}

impl<F: JoltField> SoftmaxParams<F> {
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
            computation_node,
        }
    }
}

pub struct SoftmaxProver<F: JoltField> {
    params: SoftmaxParams<F>,
    claim_A: F,
    output: Tensor<i32>,
}

impl<F: JoltField> SoftmaxProver<F> {
    pub fn initialize(trace: &Trace, params: SoftmaxParams<F>) -> Self {
        let LayerData { output, operands } = Trace::layer_data(trace, &params.computation_node);
        let claim_A = MultilinearPolynomial::from(operands[0].clone()).evaluate(&params.r_output); // TODO: rm clone
        Self {
            params,
            claim_A,
            output: output.clone(),
        }
    }

    pub fn prove(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) {
        let [num_heads, seq_len, features] = self.params.computation_node.output_dims[..] else {
            panic!(
                "Expected output_dims to have exactly three elements: [num_heads, seq_len, features]"
            )
        };
        let num_vars_num_heads = num_heads.log_2();
        let num_vars_seq_len = seq_len.log_2();
        let r_features = &self.params.r_output[num_vars_num_heads + num_vars_seq_len..];

        // Iterate over each (head, sequence) pair
        for (head_seq_idx, output_chunk) in self.output.data().chunks_exact(features).enumerate() {
            let feature_claim =
                MultilinearPolynomial::from(output_chunk.to_vec()).evaluate(r_features);
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::SoftmaxFeatureOutput(
                    self.params.computation_node.idx,
                    head_seq_idx,
                ),
                SumcheckId::Execution,
                r_features.to_vec().into(),
                feature_claim,
            );
        }

        // TODO(Forpee): rm this as this should be added by a cache openings claim method from the sum-check API
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            self.params.r_output.clone().into(),
            self.claim_A,
        );
    }
}

pub struct SoftmaxVerifier<F: JoltField> {
    params: SoftmaxParams<F>,
}

impl<F: JoltField> SoftmaxVerifier<F> {
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = SoftmaxParams::new(computation_node, accumulator);
        Self { params }
    }

    pub fn verify(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Result<(), ProofVerifyError> {
        let [num_heads, seq_len, _features] = self.params.computation_node.output_dims[..] else {
            panic!(
                "Expected output_dims to have exactly three elements: [num_heads, seq_len, features]"
            )
        };
        let num_heads_seq_len = num_heads * seq_len;
        let mut softmax_features_claim = Vec::with_capacity(num_heads_seq_len);
        let num_vars_num_heads = num_heads.log_2();
        let num_vars_seq_len = seq_len.log_2();
        let (r_heads_seq, r_features) = self
            .params
            .r_output
            .split_at(num_vars_num_heads + num_vars_seq_len);

        // Iterate over each (head, sequence) pair
        for head_seq_idx in 0..num_heads_seq_len {
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::SoftmaxFeatureOutput(
                    self.params.computation_node.idx,
                    head_seq_idx,
                ),
                SumcheckId::Execution,
                r_features.to_vec().into(),
            );
            softmax_features_claim.push(
                accumulator
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::SoftmaxFeatureOutput(
                            self.params.computation_node.idx,
                            head_seq_idx,
                        ),
                        SumcheckId::Execution,
                    )
                    .1,
            );
        }
        let softmax_claim =
            MultilinearPolynomial::from(softmax_features_claim).evaluate(r_heads_seq);

        let expected_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;

        if softmax_claim != expected_claim {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Softmax claim does not match expected claim".to_string(),
            ));
        }

        // TODO(Forpee): rm this as this should be added by a cache openings claim method from the sum-check API
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            self.params.r_output.clone().into(),
        );

        Ok(())
    }
}
