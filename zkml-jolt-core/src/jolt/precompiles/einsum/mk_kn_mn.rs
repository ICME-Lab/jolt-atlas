use crate::jolt::{
    dag::state_manager::StateManager,
    pcs::{SumcheckId, VerifierOpeningAccumulator},
    sumcheck::SumcheckInstance,
    witness::VirtualPolynomial,
};
use jolt_core::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{BIG_ENDIAN, OpeningPoint},
    },
    transcripts::Transcript,
    utils::math::Math,
};
use rayon::prelude::*;

pub struct ExecutionSumcheck<F: JoltField> {
    prover_state: Option<ExecutionProverState<F>>,
    r_x: Vec<F>,
    r_y: Vec<F>,
    rv_claim_c: F,
    index: usize,
    num_rounds: usize,
}

impl<F: JoltField> SumcheckInstance<F> for ExecutionSumcheck<F> {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_claim(&self) -> F {
        self.rv_claim_c
    }

    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        const DEGREE: usize = 2;
        let univariate_poly_evals: [F; DEGREE] = (0..prover_state.a_rx.len() / 2)
            .into_par_iter()
            .map(|i| {
                let a_evals = prover_state
                    .a_rx
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let b_evals = prover_state
                    .b_ry
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                [
                    a_evals[0] * b_evals[0], // eval at 0
                    a_evals[1] * b_evals[1], // eval at 2
                ]
            })
            .reduce(
                || [F::zero(); DEGREE],
                |mut running, new| {
                    for i in 0..DEGREE {
                        running[i] += new[i];
                    }
                    running
                },
            );
        univariate_poly_evals.into()
    }

    fn bind(&mut self, r_j: F, _round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");
        // Bind both polynomials in parallel
        rayon::join(
            || {
                prover_state
                    .a_rx
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            },
            || {
                prover_state
                    .b_ry
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            },
        );
    }

    fn cache_openings_prover(
        &self,
        accumulator: std::rc::Rc<std::cell::RefCell<crate::jolt::pcs::ProverOpeningAccumulator<F>>>,
        opening_point: jolt_core::poly::opening_proof::OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let a_claim = prover_state.a_rx.final_sumcheck_claim();
        let b_claim = prover_state.b_ry.final_sumcheck_claim();
        let r_a = [self.r_x.clone(), opening_point.r.clone()].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileA(self.index),
            SumcheckId::PrecompileExecution,
            r_a.into(),
            a_claim,
        );
        let r_b = [opening_point.r.clone(), self.r_y.clone()].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileB(self.index),
            SumcheckId::PrecompileExecution,
            r_b.into(),
            b_claim,
        );
        let r_c = [self.r_x.clone(), self.r_y.clone()].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileC(self.index),
            SumcheckId::PrecompileExecution,
            r_c.into(),
            self.input_claim(),
        );
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F],
    ) -> jolt_core::poly::opening_proof::OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn expected_output_claim(
        &self,
        opening_accumulator: Option<std::rc::Rc<std::cell::RefCell<VerifierOpeningAccumulator<F>>>>,
        _r: &[F],
    ) -> F {
        let accumulator = opening_accumulator.as_ref().unwrap();
        let (_, a_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::PrecompileA(self.index),
            SumcheckId::PrecompileExecution,
        );
        let (_, b_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::PrecompileB(self.index),
            SumcheckId::PrecompileExecution,
        );
        a_claim * b_claim
    }

    fn cache_openings_verifier(
        &self,
        accumulator: std::rc::Rc<
            std::cell::RefCell<crate::jolt::pcs::VerifierOpeningAccumulator<F>>,
        >,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let r_a = [self.r_x.clone(), opening_point.r.clone()].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileA(self.index),
            SumcheckId::PrecompileExecution,
            r_a.into(),
        );
        let r_b = [opening_point.r.clone(), self.r_y.clone()].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileB(self.index),
            SumcheckId::PrecompileExecution,
            r_b.into(),
        );
    }
}

impl<F: JoltField> ExecutionSumcheck<F> {
    /// Create the prover sum-check instance for the precompile
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        index: usize,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        // Get the final memory state (val_final) from the prover
        let final_memory_state = sm.get_val_final();
        let (pp, _, _) = sm.get_prover_data();
        let pp = &pp.shared.precompiles.instances[index];
        let m = pp.c_dims[0];
        let n = pp.c_dims[1];
        let k = pp.a_dims[1];
        let r_x: Vec<F> = sm.get_transcript().borrow_mut().challenge_vector(m.log_2());
        let r_y: Vec<F> = sm.get_transcript().borrow_mut().challenge_vector(n.log_2());

        // Extract values for operands a and b from memory
        let rv_a = pp.extract_rv(final_memory_state, |m| &m.a_addr);
        let rv_b = pp.extract_rv(final_memory_state, |m| &m.b_addr);
        Self::init_prover(index, r_x, r_y, rv_a, rv_b, (m, n, k))
    }

    fn init_prover(
        index: usize,
        r_x: Vec<F>,
        r_y: Vec<F>,
        rv_a: Vec<i64>,
        rv_b: Vec<i64>,
        (m, n, k): (usize, usize, usize),
    ) -> Self {
        let (a_rx, b_ry) = Self::witness_polys(&r_x, &r_y, &rv_a, &rv_b, m, n, k);
        let rv_claim_c = Self::rv_claim_c(&a_rx, &b_ry);
        Self {
            prover_state: Some(ExecutionProverState { a_rx, b_ry }),
            r_x,
            r_y,
            rv_claim_c,
            index,
            num_rounds: k.log_2(),
        }
    }

    fn witness_polys(
        r_x: &[F],
        r_y: &[F],
        rv_a: &[i64],
        rv_b: &[i64],
        m: usize,
        n: usize,
        k: usize,
    ) -> (MultilinearPolynomial<F>, MultilinearPolynomial<F>) {
        let eq_r_x = EqPolynomial::evals(r_x);
        let eq_r_y = EqPolynomial::evals(r_y);
        let a_rx: MultilinearPolynomial<F> = MultilinearPolynomial::from(
            (0..k)
                .into_par_iter()
                .map(|j| {
                    (0..m)
                        .map(|i| F::from_i64(rv_a[i * k + j]) * eq_r_x[i])
                        .sum()
                })
                .collect::<Vec<F>>(),
        );
        let b_ry: MultilinearPolynomial<F> = MultilinearPolynomial::from(
            (0..k)
                .into_par_iter()
                .map(|i| {
                    (0..n)
                        .map(|j| F::from_i64(rv_b[i * n + j]) * eq_r_y[j])
                        .sum()
                })
                .collect::<Vec<F>>(),
        );
        (a_rx, b_ry)
    }

    fn rv_claim_c(a_rx: &MultilinearPolynomial<F>, b_ry: &MultilinearPolynomial<F>) -> F {
        (0..a_rx.len())
            .map(|i| a_rx.get_bound_coeff(i) * b_ry.get_bound_coeff(i))
            .sum()
    }

    /// Create the verifier sum-check instance for the precompile
    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        index: usize,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        // Get preprocessing data for this matrix multiplication
        let (pp, _, _) = sm.get_verifier_data();
        let pp = &pp.shared.precompiles.instances[index];
        let m = pp.c_dims[0];
        let n = pp.c_dims[1];
        let k = pp.a_dims[1];
        let r_x: Vec<F> = sm.get_transcript().borrow_mut().challenge_vector(m.log_2());
        let r_y: Vec<F> = sm.get_transcript().borrow_mut().challenge_vector(n.log_2());
        // cache r_c
        let verifier_accumulator = sm.get_verifier_accumulator();
        let r_c = [r_x.clone(), r_y.clone()].concat();
        verifier_accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileC(index),
            SumcheckId::PrecompileExecution,
            r_c.into(),
        );
        let rv_claim_c = sm
            .get_virtual_polynomial_opening(
                VirtualPolynomial::PrecompileC(index),
                SumcheckId::PrecompileExecution,
            )
            .1;
        Self {
            prover_state: None,
            r_x,
            r_y,
            rv_claim_c,
            index,
            num_rounds: k.log_2(),
        }
    }
}

/// Stores the "witness" polynomials for the execution sumcheck.
/// These polynomials are virtual
pub struct ExecutionProverState<F: JoltField> {
    a_rx: MultilinearPolynomial<F>,
    b_ry: MultilinearPolynomial<F>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jolt::precompiles::einsum::test::{TestInstances, test_einsum_instances};
    use ark_bn254::Fr;
    use itertools::Itertools;
    use jolt_core::{
        poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        utils::math::Math,
    };
    use onnx_tracer::tensor::Tensor;
    use rand::{Rng, rngs::StdRng};

    /// Generate test instances for mk,kn->mn einsum
    pub fn random_instances(
        mut rng: StdRng,
        num_instances: usize,
        max_dims: (usize, usize, usize, usize), // (m, k, n, _unused)
    ) -> TestInstances {
        let mut prover_instances = Vec::new();
        let mut verifier_instances = Vec::new();
        let mut a_instances = Vec::new();
        let mut b_instances = Vec::new();

        for index in 0..num_instances {
            let m = rng.gen_range(4..=max_dims.0);
            let k = rng.gen_range(4..=max_dims.1);
            let n = rng.gen_range(4..=max_dims.2);
            let m_padded = m.next_power_of_two();
            let k_padded = k.next_power_of_two();
            let n_padded = n.next_power_of_two();

            // Create tensor operands
            let rv_a: Vec<i64> = (0..(m * k)).map(|_| rng.gen_range(0..10)).collect();
            let rv_b: Vec<i64> = (0..(k * n)).map(|_| rng.gen_range(0..10)).collect();

            let mut a_tensor = Tensor::new(
                Some(&rv_a.iter().map(|v| *v as i32).collect::<Vec<i32>>()),
                &[1, m, k],
            )
            .unwrap();
            let mut b_tensor = Tensor::new(
                Some(&rv_b.iter().map(|v| *v as i32).collect::<Vec<i32>>()),
                &[k, n],
            )
            .unwrap();

            // Compute expected result
            let mut c = onnx_tracer::tensor::ops::einsum(
                "amk,kn->mn",
                &[a_tensor.clone(), b_tensor.clone()],
            )
            .unwrap();

            // Pad tensors
            a_tensor.pad_to_dims(&[1, m_padded, k_padded]).unwrap();
            b_tensor.pad_to_dims(&[k_padded, n_padded]).unwrap();

            // Create random evaluation points
            let r_x: Vec<Fr> = (0..m_padded.log_2())
                .map(|_| Fr::random(&mut rng))
                .collect();
            let r_y: Vec<Fr> = (0..n_padded.log_2())
                .map(|_| Fr::random(&mut rng))
                .collect();

            let p_instance = ExecutionSumcheck::<Fr>::init_prover(
                index,
                r_x.clone(),
                r_y.clone(),
                a_tensor.inner.iter().map(|v| *v as i64).collect_vec(),
                b_tensor.inner.iter().map(|v| *v as i64).collect_vec(),
                (m_padded, n_padded, k_padded),
            );

            // Verify claim correctness
            let rv_claim_c = p_instance.rv_claim_c;
            c.pad_to_dims(&[m_padded, n_padded]).unwrap();
            let c_poly = MultilinearPolynomial::from(
                c.inner
                    .iter()
                    .map(|v| Fr::from_i64(*v as i64))
                    .collect_vec(),
            );
            let expected_claim_c = c_poly.evaluate(&[r_x.clone(), r_y.clone()].concat());
            assert_eq!(rv_claim_c, expected_claim_c);

            prover_instances
                .push(Box::new(p_instance) as Box<dyn crate::jolt::sumcheck::SumcheckInstance<Fr>>);

            let v_instance = ExecutionSumcheck::<Fr> {
                prover_state: None,
                r_x: r_x.clone(),
                r_y: r_y.clone(),
                rv_claim_c,
                index,
                num_rounds: k_padded.log_2(),
            };
            verifier_instances
                .push(Box::new(v_instance) as Box<dyn crate::jolt::sumcheck::SumcheckInstance<Fr>>);

            a_instances.push(a_tensor.inner.iter().map(|v| *v as i64).collect_vec());
            b_instances.push(b_tensor.inner.iter().map(|v| *v as i64).collect_vec());
        }

        (
            (prover_instances, a_instances, b_instances),
            verifier_instances,
        )
    }

    #[test]
    fn test_random_matmult_instances() {
        test_einsum_instances(
            random_instances,
            (32, 32, 32, 0), // (m, k, n, unused)
            0xDEAD,
            10,
        );
    }
}
