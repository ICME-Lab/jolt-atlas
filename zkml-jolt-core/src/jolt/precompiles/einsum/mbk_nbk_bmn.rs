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
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
};
use rayon::prelude::*;

pub struct ExecutionSumcheck<F: JoltField> {
    prover_state: Option<ExecutionProverState<F>>,
    r_m: Vec<F>,
    r_b: Vec<F>,
    r_n: Vec<F>,
    rv_claim_c: F,
    index: usize,
    num_rounds: usize,
    log_b: usize,
    log_k: usize,
}

impl<F: JoltField> SumcheckInstance<F> for ExecutionSumcheck<F> {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree(&self) -> usize {
        3
    }

    fn input_claim(&self) -> F {
        self.rv_claim_c
    }

    fn compute_prover_message(&mut self, round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        const DEGREE: usize = 3;
        let univariate_poly_evals: [F; DEGREE] = (0..prover_state.b_rn.len() / 2)
            .into_par_iter()
            .map(|hj| {
                let a_evals = prover_state
                    .a_rm
                    .sumcheck_evals_array::<DEGREE>(hj, BindingOrder::HighToLow);
                let b_evals = prover_state
                    .b_rn
                    .sumcheck_evals_array::<DEGREE>(hj, BindingOrder::HighToLow);
                let eq_evals = if round < self.log_b {
                    let h = hj >> self.log_k;
                    prover_state
                        .eq_rb
                        .sumcheck_evals_array::<DEGREE>(h, BindingOrder::HighToLow)
                } else {
                    let eq_rb_rh = prover_state.eq_rb_rh.expect("eq_rb_rh should be set");
                    [eq_rb_rh; 3]
                };
                [
                    a_evals[0] * b_evals[0] * eq_evals[0], // eval at 0
                    a_evals[1] * b_evals[1] * eq_evals[1], // eval at 2
                    a_evals[2] * b_evals[2] * eq_evals[2], // eval at 3
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

    fn bind(&mut self, r_j: F, round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");
        rayon::join(
            || {
                prover_state
                    .b_rn
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            },
            || {
                prover_state
                    .a_rm
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            },
        );
        if round < self.log_b {
            prover_state
                .eq_rb
                .bind_parallel(r_j, BindingOrder::HighToLow);
        };
        // cache eq eval
        if round == self.log_b - 1 {
            prover_state.eq_rb_rh = Some(prover_state.eq_rb.final_sumcheck_claim());
        }
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
        let r_a = [self.r_m.as_slice(), &opening_point.r].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileA(self.index),
            SumcheckId::PrecompileExecution,
            r_a.into(),
            prover_state.a_rm.final_sumcheck_claim(),
        );
        let r_b = [self.r_n.as_slice(), &opening_point.r].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileB(self.index),
            SumcheckId::PrecompileExecution,
            r_b.into(),
            prover_state.b_rn.final_sumcheck_claim(),
        );
        let r_c = [&self.r_b, self.r_m.as_slice(), &self.r_n].concat();
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
        r: &[F],
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
        let (r_h, _r_j) = r.split_at(self.log_b);
        a_claim * b_claim * EqPolynomial::mle(&self.r_b, r_h)
    }

    fn cache_openings_verifier(
        &self,
        accumulator: std::rc::Rc<
            std::cell::RefCell<crate::jolt::pcs::VerifierOpeningAccumulator<F>>,
        >,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let r_a = [self.r_m.as_slice(), &opening_point.r].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::PrecompileA(self.index),
            SumcheckId::PrecompileExecution,
            r_a.into(),
        );
        let r_b = [self.r_n.as_slice(), &opening_point.r].concat();
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
        let m = pp.a_dims[0];
        let b = pp.a_dims[1];
        let n = pp.b_dims[0];
        let k = pp.a_dims[2];
        let r_m: Vec<F> = sm.get_transcript().borrow_mut().challenge_vector(m.log_2());
        let r_b: Vec<F> = sm.get_transcript().borrow_mut().challenge_vector(b.log_2());
        let r_n: Vec<F> = sm.get_transcript().borrow_mut().challenge_vector(n.log_2());

        // Extract values for operands a and b from memory
        let rv_a = pp.extract_rv(final_memory_state, |m| &m.a_addr);
        let rv_b = pp.extract_rv(final_memory_state, |m| &m.b_addr);
        Self::init_prover(rv_a, rv_b, r_m, r_b, r_n, k, index)
    }

    pub fn init_prover(
        rv_a: Vec<i64>,
        rv_b: Vec<i64>,
        r_m: Vec<F>,
        r_b: Vec<F>,
        r_n: Vec<F>,
        k: usize,
        index: usize,
    ) -> Self {
        let (m, b, n) = (r_m.len().pow2(), r_b.len().pow2(), r_n.len().pow2());
        let eq_r_m = EqPolynomial::evals(&r_m);
        let eq_r_n = EqPolynomial::evals(&r_n);
        let mut a_rm: Vec<F> = unsafe_allocate_zero_vec(k * b);
        let mut b_rn: Vec<F> = unsafe_allocate_zero_vec(k * b);
        rayon::join(
            || {
                a_rm.par_chunks_mut(k).enumerate().for_each(|(h, row)| {
                    for j in 0..k {
                        row[j] = (0..m)
                            .map(|i| F::from_i64(rv_a[i * (k * b) + h * (k) + j]) * eq_r_m[i])
                            .sum();
                    }
                });
            },
            || {
                b_rn.par_chunks_mut(k).enumerate().for_each(|(h, row)| {
                    for j in 0..k {
                        row[j] = (0..n)
                            .map(|l| F::from_i64(rv_b[l * (k * b) + h * (k) + j]) * eq_r_n[l])
                            .sum();
                    }
                });
            },
        );
        let eq_r_b = EqPolynomial::evals(&r_b);
        let rv_claim_c = (0..(k * b))
            .into_par_iter()
            .map(|hj| {
                let h = hj >> k.log_2();
                a_rm[hj] * b_rn[hj] * eq_r_b[h]
            })
            .sum();
        Self {
            prover_state: Some(ExecutionProverState {
                a_rm: MultilinearPolynomial::from(a_rm),
                b_rn: MultilinearPolynomial::from(b_rn),
                eq_rb: MultilinearPolynomial::from(eq_r_b),
                eq_rb_rh: None,
            }),
            r_m,
            r_b,
            r_n,
            rv_claim_c,
            index,
            num_rounds: b.log_2() + k.log_2(),
            log_b: b.log_2(),
            log_k: k.log_2(),
        }
    }

    /// Create the verifier sum-check instance for the precompile
    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        index: usize,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        // Get preprocessing data for this precompile
        let (pp, _, _) = sm.get_verifier_data();
        let pp = &pp.shared.precompiles.instances[index];
        let m = pp.a_dims[0];
        let b = pp.a_dims[1];
        let n = pp.b_dims[0];
        let k = pp.a_dims[2];
        let r_m: Vec<F> = sm.get_transcript().borrow_mut().challenge_vector(m.log_2());
        let r_b: Vec<F> = sm.get_transcript().borrow_mut().challenge_vector(b.log_2());
        let r_n: Vec<F> = sm.get_transcript().borrow_mut().challenge_vector(n.log_2());

        // cache r_c
        let verifier_accumulator = sm.get_verifier_accumulator();
        let r_c = [r_b.clone(), r_m.clone(), r_n.clone()].concat();
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
            r_m,
            r_b,
            r_n,
            rv_claim_c,
            index,
            num_rounds: b.log_2() + k.log_2(),
            log_b: b.log_2(),
            log_k: k.log_2(),
        }
    }
}

/// Stores the "witness" polynomials for the execution sumcheck.
/// These polynomials are virtual
pub struct ExecutionProverState<F: JoltField> {
    a_rm: MultilinearPolynomial<F>,
    b_rn: MultilinearPolynomial<F>,
    eq_rb: MultilinearPolynomial<F>,
    eq_rb_rh: Option<F>,
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

    /// Generate test instances for mbk,nbk->bmn einsum
    pub fn random_instances(
        mut rng: StdRng,
        num_instances: usize,
        max_dims: (usize, usize, usize, usize), // (m, k, n, b)
    ) -> TestInstances {
        let mut prover_instances = Vec::new();
        let mut verifier_instances = Vec::new();
        let mut a_instances = Vec::new();
        let mut b_instances = Vec::new();

        for index in 0..num_instances {
            let m = rng.gen_range(4..=max_dims.0);
            let k = rng.gen_range(4..=max_dims.1);
            let n = rng.gen_range(4..=max_dims.2);
            let b = rng.gen_range(2..=max_dims.3);
            let m_padded = m.next_power_of_two();
            let k_padded = k.next_power_of_two();
            let n_padded = n.next_power_of_two();
            let b_padded = b.next_power_of_two();

            // Create rank-3 tensor operands
            let rv_a: Vec<i64> = (0..(m * b * k)).map(|_| rng.gen_range(0..10)).collect();
            let rv_b: Vec<i64> = (0..(n * b * k)).map(|_| rng.gen_range(0..10)).collect();

            let mut a_tensor = Tensor::new(
                Some(&rv_a.iter().map(|v| *v as i32).collect::<Vec<i32>>()),
                &[m, b, k],
            )
            .unwrap();
            let mut b_tensor = Tensor::new(
                Some(&rv_b.iter().map(|v| *v as i32).collect::<Vec<i32>>()),
                &[n, b, k],
            )
            .unwrap();

            // Compute expected result
            let mut c = onnx_tracer::tensor::ops::einsum(
                "mbk,nbk->abmn",
                &[a_tensor.clone(), b_tensor.clone()],
            )
            .unwrap();

            // Pad tensors
            a_tensor
                .pad_to_dims(&[m_padded, b_padded, k_padded])
                .unwrap();
            b_tensor
                .pad_to_dims(&[n_padded, b_padded, k_padded])
                .unwrap();

            // Create random evaluation points
            let r_b: Vec<Fr> = (0..b_padded.log_2())
                .map(|_| Fr::random(&mut rng))
                .collect();
            let r_m: Vec<Fr> = (0..m_padded.log_2())
                .map(|_| Fr::random(&mut rng))
                .collect();
            let r_n: Vec<Fr> = (0..n_padded.log_2())
                .map(|_| Fr::random(&mut rng))
                .collect();

            let p_instance = ExecutionSumcheck::init_prover(
                a_tensor.inner.iter().map(|v| *v as i64).collect_vec(),
                b_tensor.inner.iter().map(|v| *v as i64).collect_vec(),
                r_m.clone(),
                r_b.clone(),
                r_n.clone(),
                k_padded,
                index,
            );

            // Verify claim correctness
            let rv_claim_c = p_instance.rv_claim_c;
            c.pad_to_dims(&[1, b_padded, m_padded, n_padded]).unwrap();
            let c_poly = MultilinearPolynomial::from(
                c.inner
                    .iter()
                    .map(|v| Fr::from_i64(*v as i64))
                    .collect_vec(),
            );
            let expected_claim_c =
                c_poly.evaluate(&[r_b.clone(), r_m.clone(), r_n.clone()].concat());
            assert_eq!(rv_claim_c, expected_claim_c);

            prover_instances
                .push(Box::new(p_instance) as Box<dyn crate::jolt::sumcheck::SumcheckInstance<Fr>>);

            let v_instance = ExecutionSumcheck {
                prover_state: None,
                r_m,
                r_b,
                r_n,
                rv_claim_c,
                index,
                num_rounds: b_padded.log_2() + k_padded.log_2(),
                log_b: b_padded.log_2(),
                log_k: k_padded.log_2(),
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
            (32, 32, 32, 4), // (m, k, n, b)
            0xDEAD,
            10,
        );
    }
}
