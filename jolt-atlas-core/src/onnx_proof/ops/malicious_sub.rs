//! Malicious variant of the Sub prover for soundness testing.
//!
//! This module contains a Sub prover that forges virtual operand claims
//! while keeping the underlying sumcheck honest. It is used exclusively
//! by [`MaliciousONNXProof`] to test that the verifier correctly handles
//! (and rejects) such attacks.

use crate::onnx_proof::{malicious_prover::malicious_sumcheck_prove, ProofId, ProofType, Prover};
use common::VirtualPolynomial;
use joltworks::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{ProverOpeningAccumulator, SumcheckId},
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck::SumcheckInstanceProof, sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::SumcheckInstanceParams,
    },
    transcripts::Transcript,
};
use onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
};

use crate::onnx_proof::ops::sub::SubParams;

/// Run the malicious Sub prover for a single node.
///
/// Returns the proof entry suitable for insertion into the proof map.
pub fn malicious_sub_prove<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
    let params = SubParams::new(node.clone(), &prover.accumulator);
    let mut prover_sumcheck = MaliciousSubProver::initialize(&prover.trace, params);

    let (proof, r_sumcheck, final_claim) = malicious_sumcheck_prove(
        &mut prover_sumcheck,
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    prover_sumcheck.final_claim = Some(final_claim);
    prover_sumcheck.cache_openings(&mut prover.accumulator, &mut prover.transcript, &r_sumcheck);
    vec![(ProofId(node.idx, ProofType::Execution), proof)]
}

/// Malicious prover state for element-wise subtraction sumcheck protocol.
///
/// Identical to the honest SubProver in sumcheck computation, but forges
/// operand claims in `cache_openings` to demonstrate the attack vector.
struct MaliciousSubProver<F: JoltField> {
    params: SubParams<F>,
    eq_r_node_output: GruenSplitEqPolynomial<F>,
    left_operand: MultilinearPolynomial<F>,
    right_operand: MultilinearPolynomial<F>,
    final_claim: Option<F>,
}

impl<F: JoltField> MaliciousSubProver<F> {
    /// Initialize the prover with trace data and parameters.
    fn initialize(trace: &Trace, params: SubParams<F>) -> Self {
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
            final_claim: None,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for MaliciousSubProver<F> {
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

        // Malicious behavior: forge virtual operand claims while preserving the
        // same subtraction difference, so expected_output_claim remains unchanged.
        let left_claim = self.left_operand.final_sumcheck_claim();
        let right_claim = self.right_operand.final_sumcheck_claim();
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
            let inv = eq_eval
                .inverse()
                .expect("non-zero eq_eval must be invertible");
            forged_left - final_claim * inv
        };
        debug_assert_eq!(final_claim, eq_eval * (forged_left - forged_right));

        // Use append_virtual with forged claims. This inserts into openings
        // and appends to transcript, keeping prover/verifier in sync.
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
            opening_point.clone(),
            forged_left,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[1]),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
            opening_point,
            forged_right,
        );
    }
}
