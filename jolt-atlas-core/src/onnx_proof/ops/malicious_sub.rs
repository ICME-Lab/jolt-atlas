use crate::{
    impl_standard_params,
    onnx_proof::{ProofId, ProofType, Prover, Verifier, malicious_prover::malicious_sumcheck_prove, ops::OperatorProofTrait},
};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::Sub,
};
use common::VirtualPolynomial;
use joltworks::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        split_eq_poly::GruenSplitEqPolynomial,
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

impl_standard_params!(SubParams, 2);

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Sub {
    #[tracing::instrument(skip(node, prover))]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let params = SubParams::new(node.clone(), &prover.accumulator);
        let mut prover_sumcheck = SubProver::initialize(&prover.trace, params);

        // ここで、malicious_sumchcheck_proveを使う。
        let (proof, r_sumcheck, final_claim) = malicious_sumcheck_prove(

            &mut prover_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        // let (proof, _) = Sumcheck::prove(
        //     &mut prover_sumcheck,
        //     &mut prover.accumulator,
        //     &mut prover.transcript,
        // );
        prover_sumcheck.final_claim = Some(final_claim);
        // Register forged/honest openings after all challenges are known.
        prover_sumcheck.cache_openings(&mut prover.accumulator, &mut prover.transcript, &r_sumcheck);
        vec![(ProofId(node.idx, ProofType::Execution), proof)]
    }

    #[tracing::instrument(skip(node, verifier))]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        use joltworks::subprotocols::sumcheck::Sumcheck;

        let proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let verifier_sumcheck = SubVerifier::new(node.clone(), &verifier.accumulator);
        Sumcheck::verify(
            proof,
            &verifier_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;
        Ok(())
    }
}

/// Prover state for element-wise subtraction sumcheck protocol.
///
/// Maintains the equality polynomial and operand polynomials needed to generate
/// sumcheck messages proving that output[i] = left[i] - right[i] for all i.
pub struct SubProver<F: JoltField> {
    params: SubParams<F>,
    eq_r_node_output: GruenSplitEqPolynomial<F>,
    left_operand: MultilinearPolynomial<F>,
    right_operand: MultilinearPolynomial<F>,
    final_claim: Option<F>,
}

impl<F: JoltField> SubProver<F> {
    /// Initialize the prover with trace data and parameters.
    #[tracing::instrument(skip_all, name = "SubProver::initialize")]
    pub fn initialize(trace: &Trace, params: SubParams<F>) -> Self {
        let eq_r_node_output =
            GruenSplitEqPolynomial::new(&params.r_node_output, BindingOrder::LowToHigh);
        let LayerData {
            operands,
            output: _,
        } = Trace::layer_data(trace, &params.computation_node);
        let [left_operand, right_operand] = operands[..] else {
            panic!("Expected two operands for Sub operation")
        };
        let left_operand = MultilinearPolynomial::from(left_operand.clone());
        let right_operand = MultilinearPolynomial::from(right_operand.clone());
        Self {
            params,
            eq_r_node_output,
            left_operand,
            right_operand,
            final_claim: None
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for SubProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self {
            eq_r_node_output,
            left_operand,
            right_operand,
            ..
        } = self;
        let [q_constant] = eq_r_node_output.par_fold_out_in_unreduced::<9, 1>(&|g| {
            let lo0 = left_operand.get_bound_coeff(2 * g);
            let ro0 = right_operand.get_bound_coeff(2 * g);
            [lo0 - ro0]
        });
        eq_r_node_output.gruen_poly_deg_2(q_constant, previous_claim)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_r_node_output.bind(r_j);
        self.left_operand
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.right_operand
            .bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            opening_point.clone(),
            self.left_operand.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[1]),
            SumcheckId::Execution,
            opening_point,
            self.right_operand.final_sumcheck_claim(),
        );

        // Malicious behavior: forge virtual operand claims while preserving the
        // same subtraction difference, so expected_output_claim remains unchanged.
        let left_claim = accumulator.get_opening(OpeningId::Virtual(
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
        ));
        let right_claim = accumulator.get_opening(OpeningId::Virtual(
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[1]),
            SumcheckId::Execution,
        ));
        let final_claim = self
            .final_claim
            .expect("final_claim must be set before cache_openings");
        let r_node_output_prime = self.params.normalize_opening_point(sumcheck_challenges).r;
        let eq_eval = EqPolynomial::mle(&self.params.r_node_output, &r_node_output_prime);

        // Choose forged claims so that:
        // final_claim == eq_eval * (forged_left - forged_right)
        let forged_left = left_claim + F::one();
        let forged_right = if eq_eval.is_zero() {
            // If eq_eval == 0, the only valid final claim is 0.
            debug_assert!(final_claim.is_zero());
            right_claim
        } else {
            let inv = eq_eval.inverse().expect("non-zero eq_eval must be invertible");
            forged_left - final_claim * inv
        };
        debug_assert_eq!(final_claim, eq_eval * (forged_left - forged_right));

        // Keep transcript in sync with what verifier will append via
        // `append_operand_claims`.
        transcript.append_scalar(&forged_left);
        transcript.append_scalar(&forged_right);
        accumulator.virtual_operand_claims.insert(
            self.params.computation_node.idx,
            vec![forged_left, forged_right],
        );
    }
}

/// Verifier for element-wise subtraction sumcheck protocol.
///
/// Verifies that the prover's sumcheck messages are consistent with the claimed
/// subtraction operation output.
pub struct SubVerifier<F: JoltField> {
    params: SubParams<F>,
}

impl<F: JoltField> SubVerifier<F> {
    /// Create a new verifier for the subtraction operation.
    #[tracing::instrument(skip_all, name = "SubVerifier::new")]
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = SubParams::new(computation_node, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for SubVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r_node_output = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .0
            .r;
        let r_node_output_prime = self.params.normalize_opening_point(sumcheck_challenges).r;
        let eq_eval = EqPolynomial::mle(&r_node_output, &r_node_output_prime);
        let [left_operand_claim, right_operand_claim] =
            accumulator.get_operand_claims::<2>(self.params.computation_node.idx);
        eq_eval * (left_operand_claim - right_operand_claim)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::Execution,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[1]),
            SumcheckId::Execution,
            opening_point,
        );
        accumulator.append_operand_claims(transcript, self.params.computation_node.idx);
    }
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{
        model::{test::ModelBuilder, Model},
        tensor::Tensor,
    };
    use rand::{rngs::StdRng, SeedableRng};

    fn sub_model(rng: &mut StdRng, T: usize) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![T]);
        let c = b.constant(Tensor::random_small(rng, &[T]));
        let res = b.sub(i, c);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_sub() {
        let T = 1 << 16;
        let mut rng = StdRng::seed_from_u64(0x888);
        let input = Tensor::<i32>::random_small(&mut rng, &[T]);
        let model = sub_model(&mut rng, T);
        unit_test_op(model, &[input]);
    }
}
