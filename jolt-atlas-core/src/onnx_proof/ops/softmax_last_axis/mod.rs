//! Proofs for softmax with the last axis as the reduction dimension (ONNX Softmax operator).
//! For more on the claim flow and design of the protocol, see: https://hackmd.io/@R5TO3fi7TlWQKnY6Ejr2eg/Byjucoqjbx

use crate::onnx_proof::{
    ops::{
        softmax_last_axis::{
            exp_sum::{ExpSumParams, ExpSumProver, ExpSumVerifier},
            exponentiation::{
                mult::{MultParams, MultProver, MultVerifier},
                ExpDigit, ExpReadRafProvider,
            },
            max::{MaxIndicatorParams, MaxIndicatorProver, MaxIndicatorVerifier},
            rc::{SoftmaxRCProvider, SoftmaxRaEncoding, SAT_DIFF_RC_BITS},
            recip_mult::{RecipMultParams, RecipMultProver, RecipMultVerifier},
        },
        OperatorProofTrait,
    },
    ProofId, ProofType, Prover, Verifier,
};
use joltworks::config::{OneHotConfig, OneHotParams};

use atlas_onnx_tracer::{
    node::ComputationNode,
    ops::{
        softmax::{
            generate_exp_lut_decomposed, softmax_last_axis_decomposed, SoftmaxLastAxisTrace,
        },
        SoftmaxLastAxis,
    },
};
use common::{CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{OpeningAccumulator, OpeningPoint, SumcheckId},
    },
    subprotocols::{
        identity_range_check::{identity_rangecheck_prover, identity_rangecheck_verifier},
        shout,
        sumcheck::{BatchedSumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, lookup_bits::LookupBits, math::Math},
};

/// exp_sum(r0_k) = sum_{k,j} eq(r0_k, k) * exp_q[k,j]
pub mod exp_sum;
/// Lookups for the decomposed exponentiation (exp_hi and exp_lo).
pub mod exponentiation;
/// max indicator: max_k(r1_k) = sum_{k,j} eq(r1_k, k) * x[k,j] * e[k,j], where e[k,j] = 1 iff j = argmax_k
pub mod max;
/// Range-check provider
pub mod rc;
/// softmax(r0) * S + R(r0) = sum_{k,j} eq(r0, (k,j)) * exp_q[k,j] * inv_sum[k]
pub mod recip_mult;

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for SoftmaxLastAxis {
    #[tracing::instrument(skip_all, name = "SoftmaxLastAxis::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let softmax_input = prover.trace.operand_tensors(node)[0];
        let (_, trace) = softmax_last_axis_decomposed(softmax_input, self.scale);
        SoftmaxLastAxisProver::new(node, trace, self.scale).prove(prover)
    }

    #[tracing::instrument(skip_all, name = "SoftmaxLastAxis::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        SoftmaxLastAxisVerifier::new(node, self.scale, verifier).verify(node, verifier)
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPolynomial> {
        // self.scale is the actual scale S (e.g. 4096); log_scale is log₂(S) (e.g. 12)
        let log_scale = self.scale.trailing_zeros() as usize;
        let decomp = generate_exp_lut_decomposed(self.scale);
        let log_hi = decomp.lut_hi.len().next_power_of_two().log_2();
        let log_lo = decomp.lut_lo.len().next_power_of_two().log_2();
        let idx = node.idx;

        let mut polys = vec![];
        for (log_k, ctor) in [
            (
                log_scale,
                CommittedPolynomial::SoftmaxRemainderRaD as fn(usize, usize) -> _,
            ),
            (log_scale, CommittedPolynomial::SoftmaxExpRemainderRaD),
            (SAT_DIFF_RC_BITS, CommittedPolynomial::SoftmaxSatDiffRaD),
            (log_hi, CommittedPolynomial::SoftmaxExpZHiRaD),
            (log_lo, CommittedPolynomial::SoftmaxExpZLoRaD),
        ] {
            let d =
                OneHotParams::from_config_and_log_K(&OneHotConfig::default(), log_k).instruction_d;
            polys.extend((0..d).map(|i| ctor(idx, i)));
        }
        polys
    }
}

/// Pre-computed lookup table data shared between stage3 and stage4.
struct LookupTableData {
    table_hi: Vec<i32>,
    table_lo: Vec<i32>,
    z_hi_indices: Vec<usize>,
    z_lo_indices: Vec<usize>,
}

struct SoftmaxLastAxisProver {
    node_idx: usize,
    operand_node_index: usize,
    scale: i32,
    F_N: [usize; 2],
    trace: SoftmaxLastAxisTrace,
}

// ---------------------------------------------------------------------------
// Small helpers used across prover stages
// ---------------------------------------------------------------------------

/// Convert `i32` trace values to usize indices.
fn to_indices(values: &[i32]) -> Vec<usize> {
    values.iter().map(|&v| v as usize).collect()
}

/// Convert `i32` trace values to `LookupBits` with the given bit-width.
fn to_lookup_bits(values: &[i32], bits: usize) -> Vec<LookupBits> {
    values
        .iter()
        .map(|&v| LookupBits::new(v as u64, bits))
        .collect()
}

/// Pad a vector in-place to the next power of two with zeros.
fn pad_to_power_of_two(v: &mut Vec<i32>) {
    v.resize(v.len().next_power_of_two(), 0);
}

/// Run `BatchedSumcheck::prove` with the standard boilerplate.
fn run_batched_prove<F: JoltField, T: Transcript>(
    instances: &mut [Box<dyn SumcheckInstanceProver<F, T>>],
    prover: &mut Prover<F, T>,
) -> SumcheckInstanceProof<F, T> {
    BatchedSumcheck::prove(
        instances.iter_mut().map(|v| &mut **v as _).collect(),
        &mut prover.accumulator,
        &mut prover.transcript,
    )
    .0
}

/// Build an MLE from `data`, evaluate it at `eval_point`, and append the
/// result as a virtual polynomial opening in the prover's accumulator.
fn cache_mle_opening<F: JoltField, T: Transcript>(
    prover: &mut Prover<F, T>,
    data: &[i32],
    eval_point: &[F],
    vp: VirtualPolynomial,
) {
    let poly: MultilinearPolynomial<F> = MultilinearPolynomial::from(data.to_vec());
    let eval = poly.evaluate(eval_point);
    prover.accumulator.append_virtual(
        &mut prover.transcript,
        vp,
        SumcheckId::Execution,
        eval_point.to_vec().into(),
        eval,
    );
}

impl SoftmaxLastAxisProver {
    #[tracing::instrument(name = "SoftmaxLastAxisProver::prove", skip_all)]
    /// Run the full proving pipeline, returning all stage proofs.
    fn prove<F: JoltField, T: Transcript>(
        mut self,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let scale_bits = prover.preprocessing.scale();

        // ── Pre-compute index/lookup data from trace vectors ────────────
        // These borrow the trace fields. After this block, specific trace
        // fields can be consumed (std::mem::take) by cache/stage methods.

        // R-derived data (consumed by stage1 RC + stage2 one-hot)
        let r_lookup_bits = to_lookup_bits(&self.trace.R, scale_bits as usize);
        let r_indices = to_indices(&self.trace.R);

        // r_exp-derived data (consumed by stage2 RC + stage3 one-hot)
        let r_exp_lookup_bits =
            to_lookup_bits(&self.trace.decomposed_exp.r_exp, scale_bits as usize);
        let r_exp_indices = to_indices(&self.trace.decomposed_exp.r_exp);

        // z_hi/z_lo/sat_diff indices (consumed by stage3 + stage4)
        let z_hi_indices = to_indices(&self.trace.decomposed_exp.z_hi);
        let z_lo_indices = to_indices(&self.trace.decomposed_exp.z_lo);
        let sat_diff_lookup_bits =
            to_lookup_bits(&self.trace.decomposed_exp.sat_diff, SAT_DIFF_RC_BITS);
        let sat_diff_indices = to_indices(&self.trace.decomposed_exp.sat_diff);

        // Padded lookup tables (consumed by stage3 Shout + stage4 one-hot)
        let mut table_hi = std::mem::take(&mut self.trace.decomposed_exp.lut.lut_hi);
        pad_to_power_of_two(&mut table_hi);
        let mut table_lo = std::mem::take(&mut self.trace.decomposed_exp.lut.lut_lo);
        pad_to_power_of_two(&mut table_lo);

        let lut_data = LookupTableData {
            table_hi,
            table_lo,
            z_hi_indices,
            z_lo_indices,
        };

        // ── Pipeline ────────────────────────────────────────────────────

        self.send_auxiliary_vectors(prover);
        self.cache_exp_sum(prover);
        self.cache_R(prover);
        let stage_1_proof = self.stage1(prover, r_lookup_bits);

        self.cache_r_exp(prover);
        let stage_2_proof = self.stage2(prover, r_exp_lookup_bits, &r_indices);

        self.cache_z(prover);
        self.cache_sat_diff(prover);
        let stage_3_proof = self.stage3(prover, &lut_data, sat_diff_lookup_bits, &r_exp_indices);

        let stage_4_proof = self.stage4(prover, &lut_data, &sat_diff_indices);

        vec![
            (
                ProofId(self.node_idx, ProofType::SoftmaxStage1),
                stage_1_proof,
            ),
            (
                ProofId(self.node_idx, ProofType::SoftmaxStage2),
                stage_2_proof,
            ),
            (
                ProofId(self.node_idx, ProofType::SoftmaxStage3),
                stage_3_proof,
            ),
            (
                ProofId(self.node_idx, ProofType::SoftmaxStage4),
                stage_4_proof,
            ),
        ]
    }
}

/// Verifier for softmax-last-axis operations.
///
/// Mirror of [`SoftmaxLastAxisProver`] — each stage method corresponds to a
/// prover stage and verifies the associated sumcheck proof.
pub struct SoftmaxLastAxisVerifier {
    node_idx: usize,
    operand_node_index: usize,
    scale: i32,
    F_N: [usize; 2],
    exp_sum: Vec<i32>,
    max_k: Vec<i32>,
    argmax_k: Vec<usize>,
}

/// Pre-computed verifier lookup table data (shared across stage3, operand_link, stage4).
struct VerifierLookupTableData {
    table_hi: Vec<i32>,
    table_lo: Vec<i32>,
    base: u64,
}

impl VerifierLookupTableData {
    fn new(scale: i32) -> Self {
        let decomp = generate_exp_lut_decomposed(scale);
        let mut table_hi = decomp.lut_hi;
        table_hi.resize(table_hi.len().next_power_of_two(), 0);
        let mut table_lo = decomp.lut_lo;
        table_lo.resize(table_lo.len().next_power_of_two(), 0);
        Self {
            table_hi,
            table_lo,
            base: decomp.base as u64,
        }
    }
}

impl SoftmaxLastAxisVerifier {
    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::verify", skip_all)]
    /// Run the full verification pipeline.
    fn verify<F: JoltField, T: Transcript>(
        &mut self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        self.cache_exp_sum(verifier)?;
        self.cache_R(verifier);
        self.stage1(verifier)?;

        self.cache_r_exp(verifier);
        self.stage2(verifier)?;

        let lut = VerifierLookupTableData::new(self.scale);

        self.cache_z(verifier);
        self.cache_sat_diff(verifier);
        self.stage3(verifier, &lut)?;

        self.operand_link(node, verifier, &lut)?;

        self.stage4(verifier, &lut)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SoftmaxLastAxisProver Methods
// ---------------------------------------------------------------------------

impl SoftmaxLastAxisProver {
    fn new(node: &ComputationNode, trace: SoftmaxLastAxisTrace, scale: i32) -> Self {
        let (&n, leading_dims) = node.output_dims.split_last().unwrap();
        let f = leading_dims.iter().product::<usize>();
        Self {
            node_idx: node.idx,
            operand_node_index: node.inputs[0],
            scale,
            F_N: [f, n],
            trace,
        }
    }

    #[tracing::instrument(name = "SoftmaxLastAxisProver::send_auxiliary_vectors", skip_all)]
    /// Send auxiliary vectors (max_k, exp_sum_q, argmax_k) to the transcript.
    ///
    /// These O(F) scalars let the verifier derive `inv_sum[k] = ⌊S²/exp_sum_q[k]⌋`
    /// without relying on a committed polynomial.
    ///
    /// # Why `from_u32(… as u32)` instead of `from_i32`
    ///
    /// `max_k` can be negative (e.g. all-negative logit rows).  We deliberately
    /// use `F::from_u32(val as u32)` so the field element stays in `[0, 2^32)`,
    /// which is recoverable by the verifier via `to_u64() as i32` (two's-complement
    /// wrap).  Using `F::from_i32` on a negative value would produce `p − |val|`,
    /// a huge field element that does not roundtrip through `to_u64()`.
    ///
    /// The transcript only needs prover/verifier consistency (both sides absorb
    /// the same field element).  The recovered `i32` values are then fed into
    /// `MultilinearPolynomial::from(Vec<i32>)`, which correctly uses `from_i32`
    /// for the actual MLE computations.
    fn send_auxiliary_vectors<F: JoltField, T: Transcript>(&self, prover: &mut Prover<F, T>) {
        let [f, _] = self.F_N;
        for k in 0..f {
            prover.accumulator.append_virtual(
                &mut prover.transcript,
                VirtualPolynomial::SoftmaxSumOutput(self.node_idx, k),
                SumcheckId::NodeExecution(self.node_idx),
                OpeningPoint::default(),
                F::from_u32(self.trace.exp_sum_q[k] as u32),
            );
            prover.accumulator.append_virtual(
                &mut prover.transcript,
                VirtualPolynomial::SoftmaxMaxOutput(self.node_idx, k),
                SumcheckId::NodeExecution(self.node_idx),
                OpeningPoint::default(),
                F::from_u32(self.trace.max_k[k] as u32),
            );
            prover.accumulator.append_virtual(
                &mut prover.transcript,
                VirtualPolynomial::SoftmaxMaxIndex(self.node_idx, k),
                SumcheckId::NodeExecution(self.node_idx),
                OpeningPoint::default(),
                F::from_u32(self.trace.argmax_k[k] as u32),
            );
        }
    }

    /// Cache `exp_sum_q(r0_lead)` as a virtual polynomial opening.
    ///
    /// Evaluates the prover-sent `exp_sum_q` vector at the leading part of
    /// `r0` (the initial output opening point).
    #[tracing::instrument(name = "SoftmaxLastAxisProver::cache_exp_sum", skip_all)]
    fn cache_exp_sum<F: JoltField, T: Transcript>(&self, prover: &mut Prover<F, T>) {
        let r0 = prover.accumulator.get_node_output_opening(self.node_idx).0;
        let log_f = self.F_N[0].log_2();
        cache_mle_opening(
            prover,
            &self.trace.exp_sum_q,
            &r0.r[..log_f],
            VirtualPolynomial::SoftmaxExpSum(self.node_idx),
        );
    }

    /// Cache the R (remainder) polynomial to the accumulator.
    fn cache_R<F: JoltField, T: Transcript>(&self, prover: &mut Prover<F, T>) {
        let r0 = prover.accumulator.get_node_output_opening(self.node_idx).0;
        cache_mle_opening(
            prover,
            &self.trace.R,
            &r0.r,
            VirtualPolynomial::SoftmaxRecipMultRemainder(self.node_idx),
        );
    }

    #[tracing::instrument(name = "SoftmaxLastAxisProver::stage1", skip_all)]
    fn stage1<F: JoltField, T: Transcript>(
        &mut self,
        prover: &mut Prover<F, T>,
        r_lookup_bits: Vec<LookupBits>,
    ) -> SumcheckInstanceProof<F, T> {
        // recip mult instance
        let inv_sum_evals: Vec<F> = self.trace.inv_sum.iter().map(|&v| F::from_i32(v)).collect();
        let recip_mult_params = RecipMultParams::new(
            self.node_idx,
            self.scale,
            self.F_N,
            inv_sum_evals,
            &prover.accumulator,
            &mut prover.transcript,
        );
        // Clone exp_q for exp_sum, take for recip_mult (two consumers need owned copies)
        let exp_q_for_sum = self.trace.exp_q.clone();
        let recip_mult_prover = RecipMultProver::initialize(
            std::mem::take(&mut self.trace.exp_q),
            std::mem::take(&mut self.trace.inv_sum),
            recip_mult_params,
        );

        // exp sum instance
        let exp_sum_params = ExpSumParams::new(
            self.node_idx,
            self.F_N,
            &prover.accumulator,
            &mut prover.transcript,
        );
        let exp_sum_prover = ExpSumProver::initialize(exp_q_for_sum, exp_sum_params);

        // range-check R instance (uses pre-computed lookup bits)
        let provider = SoftmaxRCProvider::remainder(self.node_idx, prover.preprocessing.scale());
        let rc_R_prover =
            identity_rangecheck_prover(&provider, r_lookup_bits, &mut prover.accumulator);

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(recip_mult_prover),
            Box::new(exp_sum_prover),
            Box::new(rc_R_prover),
        ];
        run_batched_prove(&mut instances, prover)
    }

    /// Cache the r_exp polynomial to the accumulator.
    #[tracing::instrument(name = "SoftmaxLastAxisProver::cache_r_exp", skip_all)]
    fn cache_r_exp<F: JoltField, T: Transcript>(&self, prover: &mut Prover<F, T>) {
        let r1 = prover
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExpQ(self.node_idx),
                SumcheckId::Execution,
            )
            .0;
        cache_mle_opening(
            prover,
            &self.trace.decomposed_exp.r_exp,
            &r1.r,
            VirtualPolynomial::SoftmaxExpRemainder(self.node_idx),
        );
    }

    #[tracing::instrument(name = "SoftmaxLastAxisProver::stage2", skip_all)]
    fn stage2<F: JoltField, T: Transcript>(
        &mut self,
        prover: &mut Prover<F, T>,
        r_exp_lookup_bits: Vec<LookupBits>,
        r_indices: &[usize],
    ) -> SumcheckInstanceProof<F, T> {
        // max indicator instance
        let [f, _n] = self.F_N;
        let log_f = f.log_2();

        // Get r1 (the exp_q opening point from Stage 1)
        let r1 = prover
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExpQ(self.node_idx),
                SumcheckId::Execution,
            )
            .0;
        let r1_k = &r1.r[..log_f];

        // max_k(r1_k) — evaluate the auxiliary max_k vector at the leading point
        let max_k_poly: MultilinearPolynomial<F> =
            MultilinearPolynomial::from(self.trace.max_k.clone());
        let max_k_eval = max_k_poly.evaluate(r1_k);
        let max_indicator_params = MaxIndicatorParams::new(
            self.node_idx,
            self.operand_node_index,
            self.F_N,
            std::mem::take(&mut self.trace.argmax_k),
            max_k_eval,
            &prover.accumulator,
        );
        let max_indicator_prover =
            MaxIndicatorProver::initialize(std::mem::take(&mut self.trace.x), max_indicator_params);

        // exp mult instance
        let exp_mult_params = MultParams::new(self.node_idx, self.scale, &prover.accumulator);
        let exp_mult_prover = MultProver::initialize(
            std::mem::take(&mut self.trace.decomposed_exp.exp_hi),
            std::mem::take(&mut self.trace.decomposed_exp.exp_lo),
            exp_mult_params,
        );

        // range-check r_exp instance (uses pre-computed lookup bits)
        let provider =
            SoftmaxRCProvider::exp_remainder(self.node_idx, prover.preprocessing.scale());
        let exp_r_rc_prover =
            identity_rangecheck_prover(&provider, r_exp_lookup_bits, &mut prover.accumulator);

        // one-hot check for R ra (uses pre-computed indices)
        let encoding = SoftmaxRaEncoding::remainder(self.node_idx, prover.preprocessing.scale());
        let [R_ra_prover, R_hw_prover, R_bool_prover] = shout::ra_onehot_provers(
            &encoding,
            r_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(exp_mult_prover),
            Box::new(max_indicator_prover),
            Box::new(exp_r_rc_prover),
            R_ra_prover,
            R_hw_prover,
            R_bool_prover,
        ];
        run_batched_prove(&mut instances, prover)
    }

    #[tracing::instrument(name = "SoftmaxLastAxisProver::cache_z", skip_all)]
    /// Cache `z_hi(r2)` and `z_lo(r2)` as virtual polynomial openings.
    ///
    /// These become the RAF claims consumed by Stage 3 Shout.
    fn cache_z<F: JoltField, T: Transcript>(&self, prover: &mut Prover<F, T>) {
        let r2 = prover
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExpHi(self.node_idx),
                SumcheckId::Execution,
            )
            .0;
        cache_mle_opening(
            prover,
            &self.trace.decomposed_exp.z_hi,
            &r2.r,
            VirtualPolynomial::SoftmaxExpZHi(self.node_idx),
        );
        cache_mle_opening(
            prover,
            &self.trace.decomposed_exp.z_lo,
            &r2.r,
            VirtualPolynomial::SoftmaxExpZLo(self.node_idx),
        );
    }

    #[tracing::instrument(name = "SoftmaxLastAxisProver::cache_sat_diff", skip_all)]
    /// Cache `sat_diff(r2)` as a virtual polynomial opening.
    ///
    /// sat_diff is not independently committed — the identity RC commits
    /// to its one-hot encoding.  We only need the evaluation at r2.
    fn cache_sat_diff<F: JoltField, T: Transcript>(&self, prover: &mut Prover<F, T>) {
        let r2 = prover
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExpHi(self.node_idx),
                SumcheckId::Execution,
            )
            .0;
        cache_mle_opening(
            prover,
            &self.trace.decomposed_exp.sat_diff,
            &r2.r,
            VirtualPolynomial::SoftmaxSatDiff(self.node_idx),
        );
    }

    #[tracing::instrument(name = "SoftmaxLastAxisProver::stage3", skip_all)]
    fn stage3<F: JoltField, T: Transcript>(
        &self,
        prover: &mut Prover<F, T>,
        lut_data: &LookupTableData,
        sat_diff_lookup_bits: Vec<LookupBits>,
        r_exp_indices: &[usize],
    ) -> SumcheckInstanceProof<F, T> {
        // read-raf for exponentiation

        // hi
        let hi_provider = ExpReadRafProvider {
            node_idx: self.node_idx,
            table_size: lut_data.table_hi.len(),
            digit: ExpDigit::Hi,
        };
        let hi_prover = shout::read_raf_prover(
            &hi_provider,
            &lut_data.z_hi_indices,
            &lut_data.table_hi,
            &prover.accumulator,
            &mut prover.transcript,
        );

        // lo
        let lo_provider = ExpReadRafProvider {
            node_idx: self.node_idx,
            table_size: lut_data.table_lo.len(),
            digit: ExpDigit::Lo,
        };
        let lo_prover = shout::read_raf_prover(
            &lo_provider,
            &lut_data.z_lo_indices,
            &lut_data.table_lo,
            &prover.accumulator,
            &mut prover.transcript,
        );

        // rc sat diff (uses pre-computed lookup bits)
        let provider = SoftmaxRCProvider::sat_diff(self.node_idx);
        let sat_diff_rc_prover =
            identity_rangecheck_prover(&provider, sat_diff_lookup_bits, &mut prover.accumulator);

        // one-hot checks for r_exp ra (uses pre-computed indices)
        let encoding =
            SoftmaxRaEncoding::exp_remainder(self.node_idx, prover.preprocessing.scale());
        let [exp_r_ra_prover, exp_r_hw_prover, exp_r_bool_prover] = shout::ra_onehot_provers(
            &encoding,
            r_exp_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            hi_prover,
            lo_prover,
            Box::new(sat_diff_rc_prover),
            exp_r_ra_prover,
            exp_r_hw_prover,
            exp_r_bool_prover,
        ];
        run_batched_prove(&mut instances, prover)
    }

    #[tracing::instrument(name = "SoftmaxLastAxisProver::stage4", skip_all)]
    fn stage4<F: JoltField, T: Transcript>(
        &self,
        prover: &mut Prover<F, T>,
        lut_data: &LookupTableData,
        sat_diff_indices: &[usize],
    ) -> SumcheckInstanceProof<F, T> {
        // one-hot checks for z_hi_ra and z_lo_ra

        // hi
        let encoding = SoftmaxRaEncoding::exp_hi(self.node_idx, lut_data.table_hi.len().log_2());
        let [hi_ra_prover, hi_hw_prover, hi_bool_prover] = shout::ra_onehot_provers(
            &encoding,
            &lut_data.z_hi_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );

        // lo
        let encoding = SoftmaxRaEncoding::exp_lo(self.node_idx, lut_data.table_lo.len().log_2());
        let [lo_ra_prover, lo_hw_prover, lo_bool_prover] = shout::ra_onehot_provers(
            &encoding,
            &lut_data.z_lo_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );

        // one-hot checks for sat_diff ra
        let encoding = SoftmaxRaEncoding::sat_diff(self.node_idx);
        let [sat_diff_ra_prover, sat_diff_hw_prover, sat_diff_bool_prover] =
            shout::ra_onehot_provers(
                &encoding,
                sat_diff_indices,
                &prover.accumulator,
                &mut prover.transcript,
            );

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            hi_ra_prover,
            hi_hw_prover,
            hi_bool_prover,
            lo_ra_prover,
            lo_hw_prover,
            lo_bool_prover,
            sat_diff_ra_prover,
            sat_diff_hw_prover,
            sat_diff_bool_prover,
        ];
        run_batched_prove(&mut instances, prover)
    }
}

// ---------------------------------------------------------------------------
// SoftmaxLastAxisVerifier methods
// ---------------------------------------------------------------------------

impl SoftmaxLastAxisVerifier {
    /// Construct the verifier and receive auxiliary vectors from the transcript.
    ///
    /// Mirror of [`SoftmaxLastAxisProver::send_auxiliary_vectors`].
    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::new", skip_all)]
    fn new<F: JoltField, T: Transcript>(
        node: &ComputationNode,
        scale: i32,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Self {
        let (&n, leading_dims) = node.output_dims.split_last().unwrap();
        let f = leading_dims.iter().product::<usize>();
        let node_idx = node.idx;

        for k in 0..f {
            for vp_fn in [
                VirtualPolynomial::SoftmaxSumOutput as fn(usize, usize) -> _,
                VirtualPolynomial::SoftmaxMaxOutput,
                VirtualPolynomial::SoftmaxMaxIndex,
            ] {
                verifier.accumulator.append_virtual(
                    &mut verifier.transcript,
                    vp_fn(node_idx, k),
                    SumcheckId::NodeExecution(node_idx),
                    OpeningPoint::default(),
                );
            }
        }

        /// Read an auxiliary scalar vector from the transcript.
        fn read_aux_scalars<F: JoltField>(
            accumulator: &dyn OpeningAccumulator<F>,
            f: usize,
            node_idx: usize,
            vp_fn: fn(usize, usize) -> VirtualPolynomial,
        ) -> Vec<u64> {
            (0..f)
                .map(|k| {
                    accumulator
                        .get_virtual_polynomial_opening(
                            vp_fn(node_idx, k),
                            SumcheckId::NodeExecution(node_idx),
                        )
                        .1
                        .to_u64()
                        .expect("auxiliary scalar should fit within 64 bits")
                })
                .collect()
        }

        let exp_sum = read_aux_scalars(
            &verifier.accumulator,
            f,
            node_idx,
            VirtualPolynomial::SoftmaxSumOutput,
        )
        .into_iter()
        .map(|v| v as i32)
        .collect();
        let max_k = read_aux_scalars(
            &verifier.accumulator,
            f,
            node_idx,
            VirtualPolynomial::SoftmaxMaxOutput,
        )
        .into_iter()
        .map(|v| v as i32)
        .collect();
        let argmax_k = read_aux_scalars(
            &verifier.accumulator,
            f,
            node_idx,
            VirtualPolynomial::SoftmaxMaxIndex,
        )
        .into_iter()
        .map(|v| v as usize)
        .collect();

        Self {
            node_idx,
            operand_node_index: node.inputs[0],
            scale,
            F_N: [f, n],
            exp_sum,
            max_k,
            argmax_k,
        }
    }

    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::inv_sum_evals", skip_all)]
    fn inv_sum_evals<F: JoltField, T: Transcript>(
        &self,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<Vec<F>, ProofVerifyError> {
        let [f, _] = self.F_N;
        let s_squared = (self.scale as i64) * (self.scale as i64);
        (0..f)
            .map(|k| {
                let (_, exp_sum_q_k) = verifier.accumulator.get_virtual_polynomial_opening(
                    VirtualPolynomial::SoftmaxSumOutput(self.node_idx, k),
                    SumcheckId::NodeExecution(self.node_idx),
                );
                let exp_sum_q_int = exp_sum_q_k
                    .to_u64()
                    .expect("exp_sum_q[k] should fit in u64")
                    as i64;
                if exp_sum_q_int == 0 {
                    return Err(ProofVerifyError::InvalidOpeningProof(format!(
                        "exp_sum_q[{k}] is zero, cannot compute inv_sum"
                    )));
                }
                Ok(F::from_i32((s_squared / exp_sum_q_int) as i32))
            })
            .collect()
    }

    // ── Stage helpers ───────────────────────────────────────────────────────

    /// Cache `exp_sum_q(r0_lead)` — mirror of prover [`cache_exp_sum`].
    /// And verify that the claimed `exp_sum_q` values are consistent with the opening.
    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::cache_exp_sum", skip_all)]
    fn cache_exp_sum<F: JoltField, T: Transcript>(
        &mut self,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let r0 = verifier
            .accumulator
            .get_node_output_opening(self.node_idx)
            .0;
        let log_f = self.F_N[0].log_2();
        let r_lead = &r0.r[..log_f];
        verifier.accumulator.append_virtual(
            &mut verifier.transcript,
            VirtualPolynomial::SoftmaxExpSum(self.node_idx),
            SumcheckId::Execution,
            r_lead.to_vec().into(),
        );
        let exp_sum_poly: MultilinearPolynomial<F> =
            MultilinearPolynomial::from(std::mem::take(&mut self.exp_sum));
        let exp_sum_eval = exp_sum_poly.evaluate(r_lead);
        let claimed_exp_sum_eval = verifier
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExpSum(self.node_idx),
                SumcheckId::Execution,
            )
            .1;
        if exp_sum_eval != claimed_exp_sum_eval {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "exp_sum evaluation mismatch".to_string(),
            ));
        }
        Ok(())
    }

    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::cache_R", skip_all)]
    /// Append the remainder polynomial commitment to the accumulator.
    fn cache_R<F: JoltField, T: Transcript>(&self, verifier: &mut Verifier<'_, F, T>) {
        let r = verifier
            .accumulator
            .get_node_output_opening(self.node_idx)
            .0
            .r;
        verifier.accumulator.append_virtual(
            &mut verifier.transcript,
            VirtualPolynomial::SoftmaxRecipMultRemainder(self.node_idx),
            SumcheckId::Execution,
            r.to_vec().into(),
        );
    }

    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::stage1", skip_all)]
    fn stage1<F: JoltField, T: Transcript>(
        &self,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let recip_mult_verifier = RecipMultVerifier::new(
            self.node_idx,
            self.scale,
            self.F_N,
            self.inv_sum_evals(verifier)?,
            &verifier.accumulator,
            &mut verifier.transcript,
        );
        let exp_sum_verifier = ExpSumVerifier::new(
            self.node_idx,
            self.F_N,
            &verifier.accumulator,
            &mut verifier.transcript,
        );
        let rc_provider =
            SoftmaxRCProvider::remainder(self.node_idx, verifier.preprocessing.scale());
        let rc_R_verifier = identity_rangecheck_verifier(&rc_provider, &mut verifier.accumulator);
        BatchedSumcheck::verify(
            verifier
                .proofs
                .get(&ProofId(self.node_idx, ProofType::SoftmaxStage1))
                .ok_or(ProofVerifyError::MissingProof(self.node_idx))?,
            vec![&recip_mult_verifier, &exp_sum_verifier, &rc_R_verifier],
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;
        Ok(())
    }

    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::cache_r_exp", skip_all)]
    fn cache_r_exp<F: JoltField, T: Transcript>(&self, verifier: &mut Verifier<'_, F, T>) {
        let r = verifier
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExpQ(self.node_idx),
                SumcheckId::Execution,
            )
            .0
            .r;
        verifier.accumulator.append_virtual(
            &mut verifier.transcript,
            VirtualPolynomial::SoftmaxExpRemainder(self.node_idx),
            SumcheckId::Execution,
            r.to_vec().into(),
        );
    }

    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::stage2", skip_all)]
    fn stage2<F: JoltField, T: Transcript>(
        &self,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        // max indicator instance
        let [f, _n] = self.F_N;
        let log_f = f.log_2();

        // Get r1 (the exp_q opening point from Stage 1)
        let r1 = verifier
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExpQ(self.node_idx),
                SumcheckId::Execution,
            )
            .0;
        let r1_k = &r1.r[..log_f];

        // max_k(r1_k) — evaluate the auxiliary max_k vector at the leading point
        let max_k_poly: MultilinearPolynomial<F> = MultilinearPolynomial::from(self.max_k.clone());
        let max_k_eval = max_k_poly.evaluate(r1_k);
        let max_indicator_verifier = MaxIndicatorVerifier::new(
            self.node_idx,
            self.operand_node_index,
            self.F_N,
            self.argmax_k.clone(),
            max_k_eval,
            &verifier.accumulator,
        );

        // exp mult instance
        let exp_mult_verifier = MultVerifier::new(self.node_idx, self.scale, &verifier.accumulator);

        // range-check r_exp instance
        let rc_provider =
            SoftmaxRCProvider::exp_remainder(self.node_idx, verifier.preprocessing.scale());
        let exp_r_rc_verifier =
            identity_rangecheck_verifier(&rc_provider, &mut verifier.accumulator);

        // one-hot check for R
        let encoding = SoftmaxRaEncoding::remainder(self.node_idx, verifier.preprocessing.scale());
        let [R_ra_verifier, R_hw_verifier, R_bool_verifier] =
            shout::ra_onehot_verifiers(&encoding, &verifier.accumulator, &mut verifier.transcript);

        BatchedSumcheck::verify(
            verifier
                .proofs
                .get(&ProofId(self.node_idx, ProofType::SoftmaxStage2))
                .ok_or(ProofVerifyError::MissingProof(self.node_idx))?,
            vec![
                &exp_mult_verifier,
                &max_indicator_verifier,
                &exp_r_rc_verifier,
                &*R_ra_verifier,
                &*R_hw_verifier,
                &*R_bool_verifier,
            ],
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        Ok(())
    }

    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::cache_z", skip_all)]
    /// Cache `z_hi(r2)` and `z_lo(r2)` — mirror of prover [`cache_z`].
    fn cache_z<F: JoltField, T: Transcript>(&self, verifier: &mut Verifier<'_, F, T>) {
        let r2 = verifier
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExpHi(self.node_idx),
                SumcheckId::Execution,
            )
            .0;
        verifier.accumulator.append_virtual(
            &mut verifier.transcript,
            VirtualPolynomial::SoftmaxExpZHi(self.node_idx),
            SumcheckId::Execution,
            r2.clone(),
        );
        verifier.accumulator.append_virtual(
            &mut verifier.transcript,
            VirtualPolynomial::SoftmaxExpZLo(self.node_idx),
            SumcheckId::Execution,
            r2,
        );
    }

    /// Cache `sat_diff(r2)` — mirror of prover [`cache_sat_diff`].
    ///
    /// sat_diff is virtual (identity RC commits to the one-hot encoding).
    fn cache_sat_diff<F: JoltField, T: Transcript>(&self, verifier: &mut Verifier<'_, F, T>) {
        let r2 = verifier
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExpHi(self.node_idx),
                SumcheckId::Execution,
            )
            .0;
        verifier.accumulator.append_virtual(
            &mut verifier.transcript,
            VirtualPolynomial::SoftmaxSatDiff(self.node_idx),
            SumcheckId::Execution,
            r2,
        );
    }

    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::stage3", skip_all)]
    fn stage3<F: JoltField, T: Transcript>(
        &self,
        verifier: &mut Verifier<'_, F, T>,
        lut: &VerifierLookupTableData,
    ) -> Result<(), ProofVerifyError> {
        // read-raf for exponentiation

        // hi
        let provider = ExpReadRafProvider {
            node_idx: self.node_idx,
            table_size: lut.table_hi.len(),
            digit: ExpDigit::Hi,
        };
        let hi_verifier = shout::read_raf_verifier(
            &provider,
            lut.table_hi.clone(),
            &verifier.accumulator,
            &mut verifier.transcript,
        );

        // lo
        let provider = ExpReadRafProvider {
            node_idx: self.node_idx,
            table_size: lut.table_lo.len(),
            digit: ExpDigit::Lo,
        };
        let lo_verifier = shout::read_raf_verifier(
            &provider,
            lut.table_lo.clone(),
            &verifier.accumulator,
            &mut verifier.transcript,
        );

        // rc sat diff
        let provider = SoftmaxRCProvider::sat_diff(self.node_idx);
        let rc_sat_diff_verifier =
            identity_rangecheck_verifier(&provider, &mut verifier.accumulator);

        // one-hot checks for r_exp ra
        let encoding =
            SoftmaxRaEncoding::exp_remainder(self.node_idx, verifier.preprocessing.scale());
        let [r_exp_ra_verifier, r_exp_hw_verifier, r_exp_bool_verifier] =
            shout::ra_onehot_verifiers(&encoding, &verifier.accumulator, &mut verifier.transcript);

        BatchedSumcheck::verify(
            verifier
                .proofs
                .get(&ProofId(self.node_idx, ProofType::SoftmaxStage3))
                .ok_or(ProofVerifyError::MissingProof(self.node_idx))?,
            vec![
                &*hi_verifier,
                &*lo_verifier,
                &rc_sat_diff_verifier,
                &*r_exp_ra_verifier,
                &*r_exp_hw_verifier,
                &*r_exp_bool_verifier,
            ],
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        Ok(())
    }

    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::operand_link", skip_all)]
    /// Stage 3 operand link: derive `X(r2)` algebraically and route upstream.
    ///
    /// `X(r2) = max_k(r2_lead) − z_c(r2) − sat_diff(r2)`
    /// where `z_c(r2) = z_hi(r2)·B + z_lo(r2)`.
    fn operand_link<F: JoltField, T: Transcript>(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
        lut: &VerifierLookupTableData,
    ) -> Result<(), ProofVerifyError> {
        let [f, _n] = self.F_N;
        let log_f = f.log_2();

        // Get r2
        let r2 = verifier
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExpHi(self.node_idx),
                SumcheckId::Execution,
            )
            .0;
        let r2_lead = &r2.r[..log_f];

        // max_k(r2_lead) — verifier evaluates from sent max_k
        let max_k_poly: MultilinearPolynomial<F> = MultilinearPolynomial::from(self.max_k.clone());
        let max_k_eval = max_k_poly.evaluate(r2_lead);

        // z_c(r2) = z_hi(r2)·B + z_lo(r2)
        let (_, z_hi_eval) = verifier.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::SoftmaxExpZHi(self.node_idx),
            SumcheckId::Execution,
        );
        let (_, z_lo_eval) = verifier.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::SoftmaxExpZLo(self.node_idx),
            SumcheckId::Execution,
        );
        let z_c_eval = z_hi_eval * F::from_u64(lut.base) + z_lo_eval;

        // sat_diff(r2)
        let (_, sat_diff_eval) = verifier.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::SoftmaxSatDiff(self.node_idx),
            SumcheckId::Execution,
        );

        // X(r2) = max_k(r2_lead) − z_c(r2) − sat_diff(r2)
        let x_r2 = max_k_eval - z_c_eval - sat_diff_eval;

        // Verify the operand link: prover's claimed X(r2) must match the
        // algebraic derivation from max_k, z_c, and sat_diff.
        let (_, prover_x_r2) = verifier.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::NodeOutput(node.inputs[0]),
            SumcheckId::NodeExecution(self.node_idx),
        );
        if prover_x_r2 != x_r2 {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Operand link failed: prover's X(r2) does not match max_k - z_c - sat_diff"
                    .to_string(),
            ));
        }
        Ok(())
    }

    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::stage4", skip_all)]
    fn stage4<F: JoltField, T: Transcript>(
        &self,
        verifier: &mut Verifier<'_, F, T>,
        lut: &VerifierLookupTableData,
    ) -> Result<(), ProofVerifyError> {
        // one-hot checks for z_hi_ra and z_lo_ra

        // hi
        let encoding = SoftmaxRaEncoding::exp_hi(self.node_idx, lut.table_hi.len().log_2());
        let [hi_ra_verifier, hi_hw_verifier, hi_bool_verifier] =
            shout::ra_onehot_verifiers(&encoding, &verifier.accumulator, &mut verifier.transcript);

        // lo
        let encoding = SoftmaxRaEncoding::exp_lo(self.node_idx, lut.table_lo.len().log_2());
        let [lo_ra_verifier, lo_hw_verifier, lo_bool_verifier] =
            shout::ra_onehot_verifiers(&encoding, &verifier.accumulator, &mut verifier.transcript);

        // one-hot checks for sat_diff ra
        let encoding = SoftmaxRaEncoding::sat_diff(self.node_idx);
        let [sat_diff_ra_verifier, sat_diff_hw_verifier, sat_diff_bool_verifier] =
            shout::ra_onehot_verifiers(&encoding, &verifier.accumulator, &mut verifier.transcript);

        BatchedSumcheck::verify(
            verifier
                .proofs
                .get(&ProofId(self.node_idx, ProofType::SoftmaxStage4))
                .ok_or(ProofVerifyError::MissingProof(self.node_idx))?,
            vec![
                &*hi_ra_verifier,
                &*hi_hw_verifier,
                &*hi_bool_verifier,
                &*lo_ra_verifier,
                &*lo_hw_verifier,
                &*lo_bool_verifier,
                &*sat_diff_ra_verifier,
                &*sat_diff_hw_verifier,
                &*sat_diff_bool_verifier,
            ],
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use atlas_onnx_tracer::{
        model::{test::ModelBuilder, Model},
        tensor::Tensor,
    };

    use crate::onnx_proof::ops::test::unit_test_op;

    fn softmax_last_axis_model(input_shape: &[usize], scale: u32) -> Model {
        let mut b = ModelBuilder::with_scale(scale);
        let i = b.input(input_shape.to_vec());
        let res = b.softmax_last_axis(i);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_softmax_last_axis() {
        // Realistic pre-softmax attention scores shaped [4, 8, 8] = 256 (power of 2).
        // Non-masked scores sourced from GPT-2 layer 0 (scale=12, multiplier=4096).
        //
        // Layout: 4 attention heads, 8×8 causal attention matrix each.
        // Upper-triangular entries (future tokens) use the GPT-2 causal mask ≈ -2^30.
        // Non-masked scores range roughly [-25000, 50000] in fixed-point.
        const M: i32 = -1_073_741_824; // causal attention mask (≈ -2^30)
        #[rustfmt::skip]
        const GPT2_ATTN_SCORES: &[i32] = &[
            // Head 0 — moderate range (GPT-2 heads 0 + 2)
            //   Non-masked range: [-12738, 5440]
              -492,     M,     M,     M,     M,     M,     M,     M,
              5440, -6182,     M,     M,     M,     M,     M,     M,
              4040, -4700, -3995,     M,     M,     M,     M,     M,
              3503, -4425,   753,  -701,     M,     M,     M,     M,
             -2353, -6284, -4118, -4413,-10752,     M,     M,     M,
             -2247, -7704, -5686, -6371,-12738,-12175,     M,     M,
              -652,   749,   826, -1139, -2820, -5039, -2898,     M,
             -2622, -2193, -5682, -6362, -8036, -4648, -9872,-10283,
            // Head 1 — high variance (GPT-2 heads 1 + 10)
            //   Non-masked range: [-7209, 49207]
             13386,     M,     M,     M,     M,     M,     M,     M,
             11265, 36900,     M,     M,     M,     M,     M,     M,
              9668, 19702, 39017,     M,     M,     M,     M,     M,
             10015, 13052, 21247, 49207,     M,     M,     M,     M,
              7392,  6466,  1150, -4119, 30542,     M,     M,     M,
              4939,   493,   848, -7209,  9139, 24853,     M,     M,
             16554, 12057, 15905,  8038, 13262, 12499,  5472,     M,
             11299,  4069,  6294,  3081,  6412,  8400, 10154,  4726,
            // Head 2 — large positive range (GPT-2 heads 5 + 6)
            //   Non-masked range: [-5332, 47013]
             41677,     M,     M,     M,     M,     M,     M,     M,
             30755, 46148,     M,     M,     M,     M,     M,     M,
             31081, 16155, 43315,     M,     M,     M,     M,     M,
             29381, 15600, 12915, 47013,     M,     M,     M,     M,
             19723, 10507,  5700,  5104, 30550,     M,     M,     M,
             16650,  9028, 11201,  6500, 11179, 20697,     M,     M,
             11184,  4791,   371,  1565, -2793, -5332,  2261,     M,
             -2074, -6790, -7238, -7885, -5024, -5917, -2315, -7353,
            // Head 3 — negative bias (GPT-2 heads 7 + 8)
            //   Non-masked range: [-23295, -1045]
             -7215,     M,     M,     M,     M,     M,     M,     M,
             -7363,-18856,     M,     M,     M,     M,     M,     M,
            -12684,-10596,-20225,     M,     M,     M,     M,     M,
            -11926,-11397,-10691,-16112,     M,     M,     M,     M,
            -16957,-19591,-19715,-17550,-16389,     M,     M,     M,
            -20129,-23135,-23295,-21295,-17981,-17724,     M,     M,
             -2898, -9452, -7226, -7040, -6460,-13308, -6362,     M,
             -8241,-10310,-12414, -5334, -1045, -8036,-10493,-12577,
        ];
        let input_shape = vec![4, 8, 8];
        let input = Tensor::new(Some(GPT2_ATTN_SCORES), &input_shape).unwrap();
        let model = softmax_last_axis_model(&input_shape, 12);
        unit_test_op(model, &[input]);
    }

    /// Helper: build a small softmax test at the given scale.
    fn run_softmax_scale_test(scale: u32) {
        let input_shape = vec![2, 8];
        #[rustfmt::skip]
        let data: Vec<i32> = vec![
            10, 20, 30, 40, 50, 60, 5, 15,
            25, 35, 45, 55, 8, 18, 28, 38,
        ];
        let input = Tensor::new(Some(&data), &input_shape).unwrap();
        let model = softmax_last_axis_model(&input_shape, scale);
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_softmax_scale4() {
        run_softmax_scale_test(4);
    }

    #[test]
    fn test_softmax_scale6() {
        run_softmax_scale_test(6);
    }

    #[ignore = "range-check currently does not support odd scales"]
    #[test]
    fn test_softmax_scale7() {
        run_softmax_scale_test(7);
    }

    #[test]
    fn test_softmax_scale8() {
        run_softmax_scale_test(8);
    }

    #[test]
    fn test_softmax_scale10() {
        run_softmax_scale_test(10);
    }

    #[test]
    fn test_softmax_scale12() {
        run_softmax_scale_test(12);
    }

    #[test]
    fn test_softmax_scale14() {
        run_softmax_scale_test(14);
    }

    #[test]
    #[ignore] // scale=16 overflows i32 in tracer LUT generation
    fn test_softmax_scale16() {
        run_softmax_scale_test(16);
    }
}
