//! A sum-check precompile for matrix multiplication (where the rhs matrix is implicitly transposed).
//! Used for proving correctness of ONNX operators that do a matrix multiplication.
//! You can see it in action in [`crate::jolt_onnx::vm::precompiles`].
//!
//! # Overview:
//!   - [`MatMultPrecompile`] - We specify the precompile for matrix multiplication, by defining the lhs and rhs matrices as [`QuantizedTensor`]s as fields.
//!   - [`MatMultSumcheck`] - Defines the prover and verifier states that will be used to instantiate a [`super::sumcheck_engine::BatchedSumcheck`] instance.
//!     These sum-check instances are then fed into [`super::sumcheck_engine::BatchedSumcheck::prove`] and [`super::sumcheck_engine::BatchedSumcheck::verify`].
//!   - [`MatMultProverState`] - Handles/Defines the prover state for the matrix multiplication sum-check precompile (handles witness polynomials for sum-check prover).
//!   - [`MatMultVerifierState`] - Handles/Defines the verifier state for the matrix multiplication sum-check precompile.
//!
//! # Note:
//!   -  The MatMult protcol deviates slightly from the standard MatMult protocol as we implicitly transpose the rhs matrix B.
//!   -  See https://people.cs.georgetown.edu/jthaler/blogpost.pdf for the original MatMult protocol.

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::Itertools;
use jolt_core::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
    },
    subprotocols::sumcheck::BatchableSumcheckInstance,
    utils::{math::Math, transcript::Transcript},
};
use onnx_tracer::constants::MAX_TENSOR_SIZE;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::jolt::execution_trace::JoltONNXCycle;

/// The dimensions used in the matrix multiplication precompile protocol.
///
/// mat_mult_precompile_dims = (m, n, k) where
/// - `m` is the number of rows in the resulting matrix,
/// - `n` is the number of columns in the resulting matrix,
/// - `k` is the number of columns in the lhs matrix
///
/// We use m, and n to get the required length of the challenge vectors in the sum-check matrix-multiplication precompile.
/// And k is used to determine the number of sum-check rounds
///
/// # Note: We pad the dimensions to the next power of two.
pub type MatMultPrecompileDims = (usize, usize, usize);

/// This struct represents a precompile for matrix multiplication (where we implicitly transpose B).
/// Used to generate the witness for matrix multiplication in Jolt's ONNX execution.
///
/// # Note: We assume tensors are appropriately padded here.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct MatMultPrecompile {
    a: Vec<u64>,
    b: Vec<u64>,
}

impl MatMultPrecompile {
    /// Create a new instance of [`MatMultPrecompile`]
    pub fn new(a: Vec<u64>, b: Vec<u64>) -> Self {
        Self { a, b }
    }

    /// Return the lhs matrix of the multiplication
    pub fn a(&self) -> &Vec<u64> {
        &self.a
    }

    /// Return the rhs matrix of the multiplication
    pub fn b(&self) -> &Vec<u64> {
        &self.b
    }

    /// Matrix multiplication of two quantized tensors.
    ///
    /// # Note:
    ///
    /// - Implicitly transposes the rhs matrix before multiplication.
    /// - Intermediate results are stored as i32 to prevent overflow.
    ///
    /// # Panics:
    ///
    /// - Panics if the inner dimensions of the matrices do not match.
    pub fn matmult_rhs_transposed(&self, dims: MatMultPrecompileDims) -> (Vec<i32>, Vec<usize>) {
        let (m, n, k) = dims;

        // Output shape is [M, N]
        let mut result = vec![0i32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i32;
                for t in 0..k {
                    let a_val = self.a[i * k + t] as u32 as i32;
                    let b_val = self.b[j * k + t] as u32 as i32;
                    acc += a_val * b_val;
                }
                result[i * n + j] = acc;
            }
        }
        (result, vec![m, n])
    }

    /// # Note: Resizes the output to the maximum tensor size
    pub fn output(&self, dims: MatMultPrecompileDims) -> Vec<u64> {
        let (c, _c_shape) = self.matmult_rhs_transposed(dims);
        let mut c = c.iter().map(|&x| x as u32 as u64).collect_vec();
        c.resize(MAX_TENSOR_SIZE, 0);
        c
    }

    /// Return the left operand of the precompile
    /// # Note: Resizes the output to the maximum tensor size
    pub fn left_operand(&self) -> Vec<u64> {
        let mut a = self.a.clone();
        a.resize(MAX_TENSOR_SIZE, 0);
        a
    }

    /// Return the right operand of the precompile
    /// # Note: Resizes the output to the maximum tensor size
    pub fn right_operand(&self) -> Vec<u64> {
        let mut b = self.b.clone();
        b.resize(MAX_TENSOR_SIZE, 0);
        b
    }

    #[cfg(test)]
    /// Return an randomly initialized quantized tensor with the given shape.
    pub fn random(mut rng: impl rand_core::RngCore, dims: MatMultPrecompileDims) -> Self {
        let (m, n, k) = dims;
        let a: Vec<u64> = (0..m * k)
            .map(|_| rng.next_u32() as u8 as u32 as u64)
            .collect();
        let b: Vec<u64> = (0..n * k)
            .map(|_| rng.next_u32() as u8 as u32 as u64)
            .collect();
        Self { a, b }
    }
}

/// # Panics
/// Panics if the JoltONNXCycle is not determined by a matmult operation.
impl From<&JoltONNXCycle> for MatMultPrecompile {
    fn from(cycle: &JoltONNXCycle) -> Self {
        let a = cycle.ts1_read().1;
        let b = cycle.ts2_read().1;
        Self::new(a, b)
    }
}

/// Handles the prover state for the matrix multiplication sum-check precompile.
/// Used to create the sum-check prover instance to input into [`BatchedSumcheck::prove`].
#[derive(Clone, Debug)]
pub struct MatMultProverState<F>
where
    F: JoltField,
{
    /// A(rx, k) evaluations over the boolean hypercube
    pub a: MultilinearPolynomial<F>,
    /// B(ry, k) evaluations over the boolean hypercube
    pub b: MultilinearPolynomial<F>,
    /// C(rx, ry)
    pub input_claim: F,
    /// Number of rounds in the sum-check precompile
    pub num_rounds: usize,
}

impl<F> MatMultProverState<F>
where
    F: JoltField,
{
    #[tracing::instrument(skip_all)]
    /// Create a new instance of [`MatMultProverState`].
    /// We compute the evaluations of the polynomials A(rx, k) and B(ry, k) over the boolean hypercube,
    /// and also compute the input claim C(rx, ry) = Σₖ A(rx, k) * B(ry, k).
    ///
    /// These A(rx, k) and B(ry, k) evaluations serve as the witness for the matrix multiplication precompile.
    ///
    ///
    /// # Note: we implicitly transpose the rhs matrix B in the multiplication.
    ///  This is because the ONNX genneral matmul operator (GEMM) transposes the second matrix.
    pub fn initialize<ProofTranscript>(
        dims: MatMultPrecompileDims,
        input: &MatMultPrecompile,
        transcript: &mut ProofTranscript,
    ) -> Self
    where
        ProofTranscript: Transcript,
    {
        let (m, n, k) = dims;
        let a = input.a();
        let b = input.b();
        let log_m = m.log_2();
        let log_n = n.log_2();
        let rx: Vec<F> = transcript.challenge_scalar_powers(log_m);
        let ry: Vec<F> = transcript.challenge_scalar_powers(log_n);
        let eq_rx = EqPolynomial::evals(&rx);
        let eq_ry = EqPolynomial::evals(&ry);
        let mut A_rx = vec![F::zero(); k];
        for i in 0..m {
            for j in 0..k {
                A_rx[j] += F::from_i64(a[i * k + j] as u32 as i32 as i64) * eq_rx[i];
            }
        }
        // We index B as if it were transposed, i.e., B[i][j] = b.data[j * k + i]
        let mut B_ry = vec![F::zero(); k];
        for i in 0..n {
            for j in 0..k {
                B_ry[j] += F::from_i64(b[i * k + j] as u32 as i32 as i64) * eq_ry[i]
            }
        }
        let (c, _c_shape) = input.matmult_rhs_transposed(dims);
        let c_poly =
            MultilinearPolynomial::from(c.iter().map(|&x| F::from_i64(x as i64)).collect_vec());
        let input_claim = c_poly.evaluate(&[rx.clone(), ry.clone()].concat());
        transcript.append_scalar(&input_claim);
        #[cfg(test)]
        {
            let sum: F = A_rx.iter().zip_eq(B_ry.iter()).map(|(a, b)| *a * b).sum();
            assert_eq!(sum, input_claim)
        }
        Self {
            a: MultilinearPolynomial::from(A_rx),
            b: MultilinearPolynomial::from(B_ry),
            input_claim,
            num_rounds: k.log_2(),
        }
    }
}

/// Handles the verifier state for the matrix multiplication sum-check precompile.
/// Used to create the sum-check verifier instance to input into [`BatchedSumcheck::verify`].
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct MatMultVerifierState<F>
where
    F: JoltField,
{
    num_rounds: usize,
    input_claim: F,
}

impl<F> MatMultVerifierState<F>
where
    F: JoltField,
{
    #[tracing::instrument(skip_all)]
    /// Create a new instance of [`MatMultVerifierState`].
    /// # Note: we mainly update the state by computing the necessary challenges used in the sum-check matmult protocol.
    ///  We also append the input claim to the transcript.
    pub fn initialize<ProofTranscript>(
        m: usize,
        n: usize,
        k: usize,
        input_claim: F,
        transcript: &mut ProofTranscript,
    ) -> Self
    where
        ProofTranscript: Transcript,
    {
        let num_rounds = k.log_2();
        let log_m = m.log_2();
        let log_n = n.log_2();
        let _rx: Vec<F> = transcript.challenge_scalar_powers(log_m);
        let _ry: Vec<F> = transcript.challenge_scalar_powers(log_n);
        transcript.append_scalar(&input_claim);
        Self {
            num_rounds,
            input_claim,
        }
    }
}

/// The final claims for the matrix multiplication sum-check precompile.
///
/// a = A(rx, r_sc)
/// b = B(ry, r_sc)
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct MatMultClaims<F>
where
    F: JoltField,
{
    a: F,
    b: F,
}

/// Batchable sum-check instance for matrix multiplication precompile.
/// Used to construct the [`PrecompileProof`] by passing in these instances into [`BatchedSumcheck`].
#[derive(Clone, Debug)]
pub struct MatMultSumcheck<F>
where
    F: JoltField,
{
    /// Handles state for prover portion of the sum-check protocol.
    pub prover_state: Option<MatMultProverState<F>>,
    /// Handles state for verifier portion of the sum-check protocol.
    pub verifier_state: Option<MatMultVerifierState<F>>,
    /// Holds the final claims for the matrix multiplication sum-check precompile.
    pub claims: Option<MatMultClaims<F>>,
}

impl<F> MatMultSumcheck<F>
where
    F: JoltField,
{
    /// Create a new instance of [`MatMultSumcheck`]
    pub fn new(
        prover_state: Option<MatMultProverState<F>>,
        verifier_state: Option<MatMultVerifierState<F>>,
        claims: Option<MatMultClaims<F>>,
    ) -> Self {
        Self {
            prover_state,
            verifier_state,
            claims,
        }
    }
}

impl<F, ProofTranscript> BatchableSumcheckInstance<F, ProofTranscript> for MatMultSumcheck<F>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    #[inline(always)]
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        if self.prover_state.is_some() {
            self.prover_state.as_ref().unwrap().num_rounds
        } else if self.verifier_state.is_some() {
            self.verifier_state.as_ref().unwrap().num_rounds
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    fn input_claim(&self) -> F {
        if self.prover_state.is_some() {
            self.prover_state.as_ref().unwrap().input_claim
        } else if self.verifier_state.is_some() {
            self.verifier_state.as_ref().unwrap().input_claim
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    #[tracing::instrument(skip_all)]
    fn compute_prover_message(&mut self, _: usize) -> Vec<F> {
        let MatMultProverState { a, b, .. } = self.prover_state.as_ref().unwrap();
        let len = a.len() / 2;
        let univariate_poly_evals: [F; 2] = (0..len)
            .into_par_iter()
            .map(|i| {
                let a_evals = a.sumcheck_evals(i, 2, BindingOrder::HighToLow);
                let b_evals = b.sumcheck_evals(i, 2, BindingOrder::HighToLow);
                [a_evals[0] * b_evals[0], a_evals[1] * b_evals[1]]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );
        univariate_poly_evals.to_vec()
    }

    #[tracing::instrument(skip_all)]
    fn bind(&mut self, r_j: F, _: usize) {
        let MatMultProverState { a, b, .. } = self.prover_state.as_mut().unwrap();
        rayon::join(
            || a.bind_parallel(r_j, BindingOrder::HighToLow),
            || b.bind_parallel(r_j, BindingOrder::HighToLow),
        );
    }

    fn cache_openings(&mut self) {
        debug_assert!(self.claims.is_none());
        let MatMultProverState { a, b, .. } = self.prover_state.as_ref().unwrap();
        self.claims = Some(MatMultClaims {
            a: a.final_sumcheck_claim(),
            b: b.final_sumcheck_claim(),
        });
    }

    fn expected_output_claim(&self, _: &[F]) -> F {
        let MatMultClaims { a, b } = self.claims.as_ref().unwrap();
        *a * b
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use itertools::Itertools;
    use jolt_core::{
        subprotocols::sumcheck::{BatchableSumcheckInstance, BatchedSumcheck},
        utils::transcript::{KeccakTranscript, Transcript},
    };
    use rand::RngCore;

    use crate::jolt::precompiles::matmult::{
        MatMultPrecompile, MatMultPrecompileDims, MatMultProverState, MatMultSumcheck,
        MatMultVerifierState,
    };

    #[test]
    fn test_random_execution_trace() {
        let mut rng = test_rng();
        let trace_length = 10;
        let mut pp: Vec<MatMultPrecompileDims> = Vec::with_capacity(trace_length);
        let mut prover_transcript = KeccakTranscript::new(b"test");
        let mut sumcheck_instances = Vec::with_capacity(trace_length);
        for _ in 0..trace_length {
            let m = (rng.next_u32() as usize % 10 + 1).next_power_of_two();
            let n = (rng.next_u32() as usize % 10 + 1).next_power_of_two();
            let k = (rng.next_u32() as usize % 10 + 1).next_power_of_two();
            let dims = (m, n, k);
            pp.push(dims);
            let precompile = MatMultPrecompile::random(&mut rng, dims);
            let prover_state =
                MatMultProverState::<Fr>::initialize(dims, &precompile, &mut prover_transcript);
            let sumcheck_instance = MatMultSumcheck::new(Some(prover_state), None, None);
            sumcheck_instances.push(sumcheck_instance);
        }
        let init_claims = sumcheck_instances
            .iter()
            .map(|p| p.prover_state.as_ref().unwrap().input_claim)
            .collect_vec();
        let trait_objects: Vec<&mut dyn BatchableSumcheckInstance<Fr, KeccakTranscript>> =
            sumcheck_instances
                .iter_mut()
                .map(|p| p as &mut dyn BatchableSumcheckInstance<Fr, KeccakTranscript>)
                .collect();
        let (sumcheck_proof, _rsc) = BatchedSumcheck::prove(trait_objects, &mut prover_transcript);
        let final_claims = sumcheck_instances
            .iter()
            .map(|p| p.claims.as_ref().unwrap().clone())
            .collect_vec();

        let mut verifier_transcript = KeccakTranscript::new(b"test");
        let mut vsumcheck_instances = Vec::with_capacity(trace_length);
        for (((m, n, k), init_claim), final_claim) in pp
            .iter()
            .zip_eq(init_claims.iter())
            .zip_eq(final_claims.iter())
        {
            let verifier_state = MatMultVerifierState::<Fr>::initialize(
                *m,
                *n,
                *k,
                *init_claim,
                &mut verifier_transcript,
            );
            vsumcheck_instances.push(MatMultSumcheck::new(
                None,
                Some(verifier_state),
                Some(final_claim.clone()),
            ))
        }
        let trait_objects: Vec<&dyn BatchableSumcheckInstance<Fr, KeccakTranscript>> =
            vsumcheck_instances
                .iter()
                .map(|p| p as &dyn BatchableSumcheckInstance<Fr, KeccakTranscript>)
                .collect();
        let _r = BatchedSumcheck::verify(&sumcheck_proof, trait_objects, &mut verifier_transcript)
            .unwrap();
    }
}
