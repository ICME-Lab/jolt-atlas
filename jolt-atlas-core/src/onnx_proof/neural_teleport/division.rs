//! Division module for neural teleportation.
//!
//! This module provides division operations to reduce the domain size of lookup tables
//! for activation functions. By dividing inputs by a constant divisor, we can reduce
//! the bit-width needed for lookups.
//!
//! For example, dividing by 2 reduces a 16-bit lookup table to 15 bits.
//!
//! The quotient and remainder satisfy: input = DIVISOR * quotient + remainder
//! The quotient is a virtual polynomial (proven via the lookup operation)
//! The remainder is range-checked to be in [0, DIVISOR)

use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::Tanh,
    tensor::Tensor,
};
use common::VirtualPolynomial;
use joltworks::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::math::Math,
};

const DEGREE_BOUND: usize = 2;

/// Computes quotient and remainder for neural teleportation division.
/// Returns (quotient, remainder) where input = DIVISOR * quotient + remainder
pub fn compute_division(input: &Tensor<i32>, tau: i32) -> (Tensor<i32>, Tensor<i32>) {
    let (remainder_data, quotient_data): (Vec<i32>, Vec<i32>) = input
        .iter()
        .map(|&x| {
            let mut r = x % tau;
            let mut q = x / tau;
            // Ensure remainder has same sign as divisor (Euclidean division)
            if (r < 0 && tau > 0) || (r > 0 && tau < 0) {
                r += tau;
                q -= 1;
            }
            (r, q)
        })
        .unzip();

    let quotient = Tensor::<i32>::construct(quotient_data, input.dims().to_vec());
    let remainder = Tensor::<i32>::construct(remainder_data, input.dims().to_vec());

    (quotient, remainder)
}

/// Parameters for division sumcheck
#[derive(Clone)]
pub struct TeleportDivisionParams<F: JoltField> {
    r_node_output: Vec<F::Challenge>,
    computation_node: ComputationNode,
    tau: i32,
}

impl<F: JoltField> TeleportDivisionParams<F> {
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
        op: &Tanh,
    ) -> Self {
        let r_node_output = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(computation_node.idx),
                SumcheckId::Execution,
            )
            .0
            .r;

        Self {
            r_node_output,
            computation_node,
            tau: op.tau,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for TeleportDivisionParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.computation_node.num_output_elements().log_2()
    }
}

/// Prover for neural teleportation division
pub struct TeleportDivisionProver<F: JoltField> {
    params: TeleportDivisionParams<F>,
    eq_r_node_output: GruenSplitEqPolynomial<F>,
    input: MultilinearPolynomial<F>,
    quotient: MultilinearPolynomial<F>,
    remainder: MultilinearPolynomial<F>,
}

impl<F: JoltField> TeleportDivisionProver<F> {
    pub fn new(trace: &Trace, params: TeleportDivisionParams<F>) -> Self {
        let eq_r_node_output =
            GruenSplitEqPolynomial::new(&params.r_node_output, BindingOrder::LowToHigh);
        let LayerData { operands, .. } = Trace::layer_data(trace, &params.computation_node);
        let input_data = operands[0];

        let (quotient_tensor, remainder_tensor) = compute_division(input_data, params.tau);

        let input = MultilinearPolynomial::from(input_data.clone());
        let quotient = MultilinearPolynomial::from(quotient_tensor);
        let remainder = MultilinearPolynomial::from(remainder_tensor);

        #[cfg(test)]
        {
            let claim: F = (0..input.len())
                .map(|i| {
                    let a = input.get_bound_coeff(i);
                    let q = quotient.get_bound_coeff(i);
                    let r = remainder.get_bound_coeff(i);
                    F::from_i32(params.tau) * q + r - a
                })
                .sum();
            assert_eq!(F::zero(), claim, "Division constraint check failed");
        }

        Self {
            params,
            eq_r_node_output,
            input,
            quotient,
            remainder,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for TeleportDivisionProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self {
            eq_r_node_output,
            input,
            quotient,
            remainder,
            ..
        } = self;

        let divisor = F::from_i32(self.params.tau);
        let [q_constant] = eq_r_node_output.par_fold_out_in_unreduced::<9, 1>(&|g| {
            let inp0 = input.get_bound_coeff(2 * g);
            let q0 = quotient.get_bound_coeff(2 * g);
            let r0 = remainder.get_bound_coeff(2 * g);
            let c0 = (divisor * q0) + r0 - inp0;
            [c0]
        });
        eq_r_node_output.gruen_poly_deg_2(q_constant, previous_claim)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_r_node_output.bind(r_j);
        self.input.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.quotient.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.remainder.bind_parallel(r_j, BindingOrder::LowToHigh);
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
            self.input.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::TeleportQuotient(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point.clone(),
            self.quotient.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::TeleportRemainder(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point.clone(),
            self.remainder.final_sumcheck_claim(),
        );
    }
}

/// Verifier for neural teleportation division
pub struct TeleportDivisionVerifier<F: JoltField> {
    params: TeleportDivisionParams<F>,
}

impl<F: JoltField> TeleportDivisionVerifier<F> {
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
        op: &Tanh,
    ) -> Self {
        let params = TeleportDivisionParams::new(computation_node, accumulator, op);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for TeleportDivisionVerifier<F> {
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

        let input_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
                SumcheckId::Execution,
            )
            .1;
        let quotient_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::TeleportQuotient(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;
        let remainder_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::TeleportRemainder(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;

        let divisor = F::from_i32(self.params.tau);
        eq_eval * ((divisor * quotient_claim) + remainder_claim - input_claim)
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
            VirtualPolynomial::TeleportQuotient(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::TeleportRemainder(self.params.computation_node.idx),
            SumcheckId::Execution,
            opening_point.clone(),
        );
    }
}
