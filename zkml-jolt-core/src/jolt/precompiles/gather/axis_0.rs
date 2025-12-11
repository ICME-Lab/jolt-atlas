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

/// Computes gather(x, W)
/// where:  
///
///         x is a m-dim vector
///         W is a k x n matrix
///         gather(x, W) is a m x n matrix
///
/// Each value of x corresponds to an index in [0; k) and gather outputs the n-dim vector stored in W at this index
///
/// Example:
///
///             x =  [2, 0],        2-dim
///
///             W = [[1, 1, 1, 1],  3 x 4 dim
///                  [2, 2, 2, 2],
///                  [3, 3, 3, 3]]
///
///  gather(x, W) = [[3, 3, 3, 3],  2 x 4 dim
///                  [1, 1, 1, 1]]
///
///
/// To compute this, we first construct X, a m x k dim matrix holding the one-hot encoding of value stored in x
///
///             X = [[0, 0, 1],     2 x 3 dim
///                  [1, 0, 0]]
///
/// We then compute the matrix multiplication of X by W, with the desired output
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
        let k = pp.b_dims[0];
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
        _m: usize,
        n: usize,
        k: usize,
    ) -> (MultilinearPolynomial<F>, MultilinearPolynomial<F>) {
        let eq_r_x = EqPolynomial::evals(r_x);
        let eq_r_y = EqPolynomial::evals(r_y);
        println!("rv_a: {rv_a:?}");
        // Represeents one-hot encoding of input a
        let mut test: Vec<_> = unsafe_allocate_zero_vec(_m * k);
        for (i, &x) in rv_a.iter().enumerate() {
            test[i * k + x as usize] = F::one();
        }
        println!("test: {test:?}, dims: {}", test.len());
        let test = (0..k)
            .into_par_iter()
            .map(|j| (0.._m).map(|i| test[i * k + j] * eq_r_x[i]).sum())
            .collect::<Vec<F>>();
        println!("test a_rx: {test:?}");
        let mut a_rx: Vec<F> = unsafe_allocate_zero_vec(k);
        for (i, gather_index) in rv_a.iter().enumerate() {
            a_rx[*gather_index as usize] += eq_r_x[i];
        }
        println!("a_rx: {a_rx:?}");
        let a_rx: MultilinearPolynomial<F> = MultilinearPolynomial::from(a_rx);
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
        let k = pp.b_dims[0];
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
    use crate::jolt::precompiles::{
        gather::test::TestInstances, gather::test::test_gather_instances,
    };
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
        let mut b_dims = Vec::new();

        for index in 0..num_instances {
            let m: usize = rng.gen_range(4..=max_dims.0);
            let k: usize = rng.gen_range(4..=max_dims.1);
            let n: usize = rng.gen_range(4..=max_dims.2);
            let m_padded = m.next_power_of_two();
            let k_padded = k.next_power_of_two();
            let n_padded = n.next_power_of_two();

            // Create tensor operands
            let rv_a: Vec<i64> = (0..(m)).map(|_| rng.gen_range(0..k) as i64).collect();
            let rv_b: Vec<i64> = (0..(k * n)).map(|_| rng.gen_range(0..10)).collect();

            let mut a_tensor = Tensor::new(
                Some(&rv_a.iter().map(|v| *v as usize).collect::<Vec<usize>>()),
                &[1, m],
            )
            .unwrap();
            let mut b_tensor = Tensor::new(
                Some(&rv_b.iter().map(|v| *v as i32).collect::<Vec<i32>>()),
                &[k, n],
            )
            .unwrap();

            // Compute expected result
            let c = onnx_tracer::tensor::ops::gather(&b_tensor, &a_tensor, 0).unwrap();

            // Pad tensors
            a_tensor.pad_to_dims(&[1, m_padded]).unwrap();
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
            let c_padded = onnx_tracer::tensor::ops::gather(&b_tensor, &a_tensor, 0).unwrap();

            let c_poly = MultilinearPolynomial::from(
                c_padded
                    .inner
                    .iter()
                    .map(|v| Fr::from_i64(*v as i64))
                    .collect_vec(),
            );
            let expected_claim_c = c_poly.evaluate(&[r_x.clone(), r_y.clone()].concat());
            assert_eq!(rv_claim_c, expected_claim_c);

            // For non-power of 2 dimensions, input tensors a and B are padded, and output C as well.
            // This may cause a mismatch between the expected result padded with 0, and actual output.
            // This is due to one-hot encoding the a input:
            // Example: m = 3, k = 8
            // a = [2, 5, 1]
            // expected one-hot:
            // A = [[0, 0, 1, 0, 0, 0, 0, 0],
            //      [0, 0, 0, 0, 0, 1, 0, 0],
            //      [0, 1, 0, 0, 0, 0, 0, 0]]
            //
            // With padding:
            // a = [2, 5, 1, 0]
            // actual one-hot:
            // A = [[0, 0, 1, 0, 0, 0, 0, 0],
            //      [0, 0, 0, 0, 0, 1, 0, 0],
            //      [0, 1, 0, 0, 0, 0, 0, 0],
            //      [1, 0, 0, 0, 0, 0, 0, 0]] Notice the "1" in first column, since 4th input of a is filled with a 0 whereas it should be padded with 0
            let c_final = c_padded.crop_to_dims(&[m, n]).unwrap();
            assert_eq!(c, c_final);

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
            b_dims.push((m_padded, k_padded, n_padded));
        }

        (
            (prover_instances, a_instances, b_instances, b_dims),
            verifier_instances,
        )
    }

    #[test]
    fn test_random_gather_instances() {
        test_gather_instances(
            random_instances,
            (32, 32, 32, 0), // (m, k, n, unused)
            0xDEAD,
            10,
        );
    }
}
