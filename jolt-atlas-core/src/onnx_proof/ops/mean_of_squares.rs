//! Monolithic fused mean-of-squares (`MeanOfSquares`, issue #194).
//!
//! Computes, per retained output element `h` (the reduced axis has size `N`):
//!
//! ```text
//! acc[h]      = Σ_j x[h,j]²                       (i64 squared sum — no i32 overflow)
//! rescaled[h] = acc[h] / D,        D = N·2^S      (mean rebase, D a per-node constant)
//! out[h]      = SatClamp(rescaled[h])             (i32)
//! ```
//!
//! replacing the old `Square → Sum → Div(N)` decomposition (which squared in i32
//! and could saturate, the reason `pre_rebase_nonlinear` existed). The proof chain
//! (mirrors `Sum`, plus the rescale of [`fused_rebase`]):
//!
//! 1. **Reduction sumcheck** ([`MeanOfSquaresReductionProver`], `RescaleArith`):
//!    `acc(r) = Σ_{h,j} eq(r,h)·x[h,j]²`, reducing to one input opening. Its
//!    initial claim is `acc(r) = rescaled·D + R` (the rescale seam). This is the
//!    einsum **High** eq-schedule (`eq·op·op`, eq over the retained/high bits,
//!    free sum over the reduced/low bits) specialized to one operand.
//! 2. **Clamp** (`Execution` + `RaOneHotChecks`, via [`fused_rebase::prove_pre`]):
//!    `out = SatClamp(rescaled)`, appending `rescaled` (`ClampAcc`) and `R`
//!    (`RescaleRemainder`).
//! 3. **Remainder range check** `R < D` (`RangeCheck` + `RescaleRemainderRaChecks`,
//!    via the [`range_checking`](crate::onnx_proof::range_checking) `UnsignedLT`
//!    lookup; `D` is a constant, so the Teleport-style
//!    [`MeanOfSquaresRangeCheckOperands`] is used).
//!
//! The reduction currently requires the reduced count `N` to be a power of two
//! (so the flat-padded operand splits cleanly into `[retained, reduced]`), the
//! same effective constraint as the `Sum` axis reduction; RMSNorm hidden sizes
//! that satisfy this are covered.

use std::array;

use crate::{
    onnx_proof::{
        clamp_lookups::{clamp_committed_polys, is_scalar, recover_small_int, verify_scalar_clamp},
        fused_rebase,
        ops::{OperatorProofTrait, Prover, Verifier},
        range_checking::{
            range_check_operands::MeanOfSquaresRangeCheckOperands, RangeCheckEncoding,
            RangeCheckProvider,
        },
        ProofId, ProofType,
    },
    utils::opening_access::{AccOpeningAccessor, Target},
};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::{mean_of_squares::mos_divisor, MeanOfSquares, Operator},
};
use common::{consts::XLEN, parallel::par_enabled, CommittedPoly, VirtualPoly};
use joltworks::{
    field::{IntoOpening, JoltField},
    lookup_tables::unsigned_less_than::UnsignedLessThanTable,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        shout::{self, RaOneHotEncoding},
        sumcheck::{BatchedSumcheck, Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};
use rayon::prelude::*;

const DEGREE: usize = 3;

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for MeanOfSquares {
    #[tracing::instrument(skip_all, name = "MeanOfSquares::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let scalar = is_scalar(node);
        let mut proofs = Vec::new();

        // (1+2) Append `R` (RescaleRemainder) + `rescaled` (ClampAcc) and prove
        // the clamp `out = SatClamp(rescaled)`.
        proofs.extend(fused_rebase::prove_pre(node, prover));

        // (3) Reduction sumcheck `acc(r) = Σ eq·x²` (claim `rescaled·D + R`).
        let params = MeanOfSquaresReductionParams::new(node.clone(), &prover.accumulator);
        let mut reduction = MeanOfSquaresReductionProver::initialize(&prover.trace, params);
        let (proof, _) = Sumcheck::prove(
            &mut reduction,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        proofs.push((ProofId(node.idx, ProofType::RescaleArith), proof));

        // (4) Remainder range check `R < D`. Scalar outputs prove it in the clear
        // (the one-hot reduction degenerates), checked by the verifier.
        if !scalar {
            let rc_provider = RangeCheckProvider::<MeanOfSquaresRangeCheckOperands>::new(node);
            let (mut rc_sumcheck, lookup_indices) = rc_provider
                .read_raf_prove::<F, T, UnsignedLessThanTable<XLEN>>(
                    &prover.trace,
                    &mut prover.accumulator,
                    &mut prover.transcript,
                );
            let (rc_proof, _) = Sumcheck::prove(
                &mut rc_sumcheck,
                &mut prover.accumulator,
                &mut prover.transcript,
            );
            proofs.push((ProofId(node.idx, ProofType::RangeCheck), rc_proof));

            let encoding = RangeCheckEncoding::<MeanOfSquaresRangeCheckOperands>::new(node);
            let [ra, hw, boolean] = shout::ra_onehot_provers(
                &encoding,
                &lookup_indices,
                &prover.accumulator,
                &mut prover.transcript,
            );
            let mut instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> = vec![ra, hw, boolean];
            let (ra_proof, _) = BatchedSumcheck::prove(
                instances.iter_mut().map(|v| &mut **v as _).collect(),
                &mut prover.accumulator,
                &mut prover.transcript,
            );
            proofs.push((
                ProofId(node.idx, ProofType::RescaleRemainderRaChecks),
                ra_proof,
            ));
        }

        proofs
    }

    #[tracing::instrument(skip_all, name = "MeanOfSquares::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let scalar = is_scalar(node);

        // (1+2) Clamp + remainder advice.
        fused_rebase::verify_pre(node, verifier)?;

        // (3) Reduction sumcheck.
        let proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::RescaleArith))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let reduction = MeanOfSquaresReductionVerifier::new(node.clone(), &verifier.accumulator);
        Sumcheck::verify(
            proof,
            &reduction,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        // (4) Remainder range check `R < D`.
        if scalar {
            // Scalar fallback: `rescaled`, `R`, and `out` open in the clear.
            let accessor = AccOpeningAccessor::new(&verifier.accumulator, node);
            let rescaled_claim = accessor.get_advice(VirtualPoly::ClampAcc).1;
            let output_claim = accessor.get_reduced_opening().1;
            verify_scalar_clamp(rescaled_claim, output_claim, "MeanOfSquares")?;

            let r_claim = accessor.get_advice(VirtualPoly::RescaleRemainder).1;
            verify_scalar_remainder(r_claim, mos_divisor(self))?;
        } else {
            let rc_proof = verifier
                .proofs
                .get(&ProofId(node.idx, ProofType::RangeCheck))
                .ok_or(ProofVerifyError::MissingProof(node.idx))?;
            let rc_provider = RangeCheckProvider::<MeanOfSquaresRangeCheckOperands>::new(node);
            let rc_verifier = rc_provider.read_raf_verify::<F, T, UnsignedLessThanTable<XLEN>>(
                &mut verifier.accumulator,
                &mut verifier.transcript,
            );
            Sumcheck::verify(
                rc_proof,
                &rc_verifier,
                &mut verifier.accumulator,
                &mut verifier.transcript,
            )?;

            let ra_proof = verifier
                .proofs
                .get(&ProofId(node.idx, ProofType::RescaleRemainderRaChecks))
                .ok_or(ProofVerifyError::MissingProof(node.idx))?;
            let encoding = RangeCheckEncoding::<MeanOfSquaresRangeCheckOperands>::new(node);
            let [ra, hw, boolean] = shout::ra_onehot_verifiers(
                &encoding,
                &verifier.accumulator,
                &mut verifier.transcript,
            );
            BatchedSumcheck::verify(
                ra_proof,
                vec![&*ra, &*hw, &*boolean],
                &mut verifier.accumulator,
                &mut verifier.transcript,
            )?;
        }

        Ok(())
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        // Saturating clamp one-hot (64-bit) ...
        let mut polys = clamp_committed_polys(node);
        // ... plus the `R < D` range-check one-hot (skipped for scalar outputs).
        if !is_scalar(node) {
            let d = RangeCheckEncoding::<MeanOfSquaresRangeCheckOperands>::new(node)
                .one_hot_params()
                .instruction_d;
            polys.extend((0..d).map(|i| CommittedPoly::MeanOfSquaresRangeCheckRaD(node.idx, i)));
        }
        polys
    }
}

/// Verify a scalar mean-of-squares remainder directly: `0 ≤ R < D`.
fn verify_scalar_remainder<F: JoltField>(r_claim: F, divisor: i64) -> Result<(), ProofVerifyError> {
    let value = recover_small_int(r_claim).ok_or_else(|| {
        ProofVerifyError::InvalidOpeningProof(
            "mean-of-squares (scalar): remainder claim is not a small integer".to_string(),
        )
    })?;
    if value < 0 || value >= divisor {
        return Err(ProofVerifyError::InvalidOpeningProof(
            "mean-of-squares (scalar): remainder must lie in [0, D)".to_string(),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Reduction sumcheck: acc(r) = Σ_{h,j} eq(r,h)·x[h,j]²
// ---------------------------------------------------------------------------

/// Parameters for the mean-of-squares reduction sumcheck.
#[derive(Clone)]
pub struct MeanOfSquaresReductionParams<F: JoltField> {
    r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    computation_node: ComputationNode,
    /// `D = N·2^S` — the rescale divisor (bound into the input claim).
    divisor: i64,
    /// Number of retained (output) variables — the eq cube, the high rounds.
    log_retained: usize,
    /// Number of reduced (summed) variables — the low rounds.
    log_reduced: usize,
}

impl<F: JoltField> MeanOfSquaresReductionParams<F> {
    /// Build params from the node and accumulator.
    pub fn new(computation_node: ComputationNode, accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let Operator::MeanOfSquares(op) = &computation_node.operator else {
            panic!("MeanOfSquaresReductionParams: expected MeanOfSquares operator");
        };
        assert!(
            op.count.is_power_of_two(),
            "mean-of-squares reduction requires a power-of-two reduced count, got {}",
            op.count
        );
        let divisor = mos_divisor(op);
        let log_reduced = op.count.log_2();
        let accessor = AccOpeningAccessor::new(accumulator, &computation_node);
        let r_node_output = accessor.get_reduced_opening().0;
        let log_retained = r_node_output.r.len();
        Self {
            r_node_output,
            computation_node,
            divisor,
            log_retained,
            log_reduced,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for MeanOfSquaresReductionParams<F> {
    fn degree(&self) -> usize {
        DEGREE
    }

    /// The raw accumulation `acc(r) = rescaled(r)·D + R(r)` (the rescale seam).
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.computation_node);
        let rescaled = accessor.get_advice(VirtualPoly::ClampAcc).1;
        let remainder = accessor.get_advice(VirtualPoly::RescaleRemainder).1;
        rescaled * F::from_u64(self.divisor as u64) + remainder
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(challenges.to_vec())
    }

    fn num_rounds(&self) -> usize {
        self.log_retained + self.log_reduced
    }
}

/// Prover for the mean-of-squares reduction sumcheck.
pub struct MeanOfSquaresReductionProver<F: JoltField> {
    params: MeanOfSquaresReductionParams<F>,
    /// The (flat, padded) input operand, laid out `[retained (high), reduced (low)]`.
    operand: MultilinearPolynomial<F>,
    /// Eq-poly over the retained (output) variables.
    eq: MultilinearPolynomial<F>,
    /// `eq` final claim, cached once its cube is fully bound (after `log_retained`).
    eq_bound_claim: Option<F>,
}

impl<F: JoltField> MeanOfSquaresReductionProver<F> {
    /// Initialize by loading the operand and the retained-dim eq-poly.
    #[tracing::instrument(skip_all, name = "MeanOfSquaresReductionProver::initialize")]
    pub fn initialize(trace: &Trace, params: MeanOfSquaresReductionParams<F>) -> Self {
        let LayerData { operands, .. } = Trace::layer_data(trace, &params.computation_node);
        let [operand] = operands[..] else {
            panic!("Expected one operand for MeanOfSquares operation")
        };
        let operand = MultilinearPolynomial::from(operand.padded_next_power_of_two());
        assert_eq!(
            operand.get_num_vars(),
            params.log_retained + params.log_reduced,
            "mean-of-squares operand size must equal retained·reduced (reduced count must be a power of two)"
        );
        let eq = MultilinearPolynomial::from(EqPolynomial::evals(&params.r_node_output.r));
        // Scalar output (`log_retained == 0`): there are no retained rounds to
        // bind, so the eq cube (over zero variables) is the constant 1 from the
        // start.
        let eq_bound_claim = (params.log_retained == 0).then(F::one);
        Self {
            params,
            operand,
            eq,
            eq_bound_claim,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for MeanOfSquaresReductionProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let half = self.operand.len() / 2;
        let log_retained = self.params.log_retained;
        let log_reduced = self.params.log_reduced;
        let evals: [F; DEGREE] = (0..half)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|i| {
                let l = self
                    .operand
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let e = if round < log_retained {
                    self.eq
                        .sumcheck_evals_array::<DEGREE>(i >> log_reduced, BindingOrder::HighToLow)
                } else {
                    [self.eq_bound_claim.expect("eq claim should be cached"); DEGREE]
                };
                array::from_fn(|k| l[k] * l[k] * e[k])
            })
            .reduce(
                || [F::zero(); DEGREE],
                |a, b| array::from_fn(|k| a[k] + b[k]),
            );
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        self.operand.bind_parallel(r_j, BindingOrder::HighToLow);
        if round < self.params.log_retained {
            self.eq.bind_parallel(r_j, BindingOrder::HighToLow);
            if round == self.params.log_retained - 1 {
                self.eq_bound_claim = Some(self.eq.final_claim());
            }
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let point = OpeningPoint::new(sumcheck_challenges.into_opening());
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .into_provider(transcript, point);
        provider.append_nodeio(Target::Input(0), self.operand.final_claim());
    }
}

/// Verifier for the mean-of-squares reduction sumcheck.
pub struct MeanOfSquaresReductionVerifier<F: JoltField> {
    params: MeanOfSquaresReductionParams<F>,
}

impl<F: JoltField> MeanOfSquaresReductionVerifier<F> {
    /// Create the verifier.
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        Self {
            params: MeanOfSquaresReductionParams::new(computation_node, accumulator),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for MeanOfSquaresReductionVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        // `eq(r, retained challenges) · operand²`, where the retained challenges
        // are the high (first `log_retained`) sumcheck challenges.
        let (r_retained, _r_reduced) = sumcheck_challenges.split_at(self.params.log_retained);
        let eq_eval = EqPolynomial::mle(&self.params.r_node_output.r, r_retained);
        let operand = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .get_nodeio(Target::Input(0))
            .1;
        eq_eval * operand * operand
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let point = OpeningPoint::new(sumcheck_challenges.into_opening());
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .into_provider(transcript, point);
        provider.append_nodeio(Target::Input(0));
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

    /// `[m, n]` input, reduce the last axis (size `n`, a power of two) → `[m, 1]`.
    fn mos_model(m: usize, n: usize) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![m, n]);
        let res = b.mean_of_squares(i, vec![1], vec![m, 1]);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_mean_of_squares() {
        let m = 1 << 4;
        let n = 1 << 7;
        let mut rng = StdRng::seed_from_u64(0x9a2);
        let input = Tensor::<i32>::random_small(&mut rng, &[m, n]);
        unit_test_op(mos_model(m, n), &[input]);
    }

    /// Scalar output (`m = 1`): the range-check / clamp one-hot reductions
    /// degenerate, so `rescaled`/`R` are checked in the clear.
    #[test]
    fn test_mean_of_squares_scalar() {
        let n = 1 << 7;
        let mut rng = StdRng::seed_from_u64(0x9a3);
        let input = Tensor::<i32>::random_small(&mut rng, &[1, n]);
        unit_test_op(mos_model(1, n), &[input]);
    }

    /// Large inputs make `Σx² / (N·2^S)` overflow i32, so the output saturates to
    /// `i32::MAX` (mean-of-squares is ≥ 0) with a non-zero remainder.
    #[test]
    fn test_mean_of_squares_saturating_clamp() {
        let (m, n) = (1 << 1, 1 << 4);
        let big = (1 << 20) + 1;
        let input = Tensor::new(Some(&vec![big; m * n]), &[m, n]).unwrap();
        unit_test_op(mos_model(m, n), &[input]);
    }
}
