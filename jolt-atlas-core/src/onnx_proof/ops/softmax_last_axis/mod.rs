//! Proofs for softmax with the last axis as the reduction dimension (ONNX Softmax operator).
//! For more on the claim flow and design of the protocol, see: https://hackmd.io/@R5TO3fi7TlWQKnY6Ejr2eg/Byjucoqjbx
//!
//! The prover sends three auxiliary vectors (`max_k`, `exp_sum_q`, `argmax_k`) to the
//! verifier via the transcript. These let the verifier derive `inv_sum[k]` without a
//! committed polynomial, but they add O(F) field elements to the proof size.
//!
//! TODO(#218): Remove auxiliary vectors and derive them inside the protocol.

use crate::{
    onnx_proof::{
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
                sat_diff::{
                    SatDiffSlacknessParams, SatDiffSlacknessProver, SatDiffSlacknessVerifier,
                },
            },
            OperatorProofTrait,
        },
        ProofId, ProofType, Prover, Verifier,
    },
    utils::opening_access::{AccOpeningAccessor, Target},
};
use joltworks::{
    config::{OneHotConfig, OneHotParams},
    poly::opening_proof::VerifierOpeningAccumulator,
};
use std::collections::BTreeMap;

use atlas_onnx_tracer::{
    node::ComputationNode,
    ops::{
        softmax::{
            generate_exp_lut_decomposed, softmax_last_axis_decomposed, SoftmaxLastAxisTrace,
        },
        SoftmaxLastAxis,
    },
};
use common::{CommittedPoly, VirtualPoly};
use joltworks::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::OpeningPoint,
    },
    subprotocols::{
        identity_range_check::{identity_rangecheck_prover, identity_rangecheck_verifier},
        shout,
        sumcheck::{BatchedSumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::SumcheckInstanceVerifier,
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
/// Complementary slackness: sat_diff * (z_bound - 1 - z_c) = 0
pub mod sat_diff;

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for SoftmaxLastAxis {
    #[tracing::instrument(skip_all, name = "SoftmaxLastAxis::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let softmax_input = prover.trace.operand_tensors(node)[0];
        let trace = softmax_last_axis_decomposed(softmax_input, self.scale).1;
        SoftmaxLastAxisProver::new(node, trace, self.scale).prove(prover)
    }

    #[tracing::instrument(skip_all, name = "SoftmaxLastAxis::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let scale_bits = verifier.preprocessing.scale();
        let mut sm = SoftmaxLastAxisVerifier::new(
            node,
            self.scale,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );
        sm.verify(
            &mut verifier.accumulator,
            &mut verifier.transcript,
            verifier.proofs,
            scale_bits,
        )
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        // self.scale is the actual scale S (e.g. 4096); log_scale is log₂(S) (e.g. 12)
        let log_scale = self.scale.ilog2() as usize;
        let decomp = generate_exp_lut_decomposed(self.scale);
        let log_hi = decomp.lut_hi.len().next_power_of_two().log_2();
        let log_lo = decomp.lut_lo.len().next_power_of_two().log_2();
        let idx = node.idx;

        let mut polys = vec![];
        for (log_k, ctor) in [
            (
                log_scale,
                CommittedPoly::SoftmaxRemainderRaD as fn(usize, usize) -> _,
            ),
            (log_scale, CommittedPoly::SoftmaxExpRemainderRaD),
            (SAT_DIFF_RC_BITS, CommittedPoly::SoftmaxSatDiffRaD),
            (log_hi, CommittedPoly::SoftmaxZHiRaD),
            (log_lo, CommittedPoly::SoftmaxZLoRaD),
        ] {
            let d =
                OneHotParams::from_config_and_log_K(&OneHotConfig::default(), log_k).instruction_d;
            polys.extend((0..d).map(|i| ctor(idx, i)));
        }
        polys
    }
}

pub(crate) struct SoftmaxLastAxisProver {
    pub(crate) computation_node: ComputationNode,
    pub(crate) scale: i32,
    pub(crate) F_N: [usize; 2],
    pub(crate) trace: SoftmaxLastAxisTrace,
}

impl SoftmaxLastAxisProver {
    #[inline]
    fn idx(&self) -> usize {
        self.computation_node.idx
    }

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
        let base = self.trace.decomposed_exp.lut.base as u64;
        let mut table_hi = std::mem::take(&mut self.trace.decomposed_exp.lut.lut_hi);
        let z_bound_minus_1 = (table_hi.len() as u64) * base - 1;
        pad_to_power_of_two(&mut table_hi);
        let mut table_lo = std::mem::take(&mut self.trace.decomposed_exp.lut.lut_lo);
        pad_to_power_of_two(&mut table_lo);

        let lut_data = LookupTableData {
            table_hi,
            table_lo,
            z_hi_indices,
            z_lo_indices,
            base,
            z_bound_minus_1,
        };

        // ── Pipeline ────────────────────────────────────────────────────

        self.send_auxiliary_vectors(prover);
        self.cache_exp_sum(prover);
        self.cache_R(prover);
        let stage_1_proof = self.stage1(prover, r_lookup_bits);

        self.cache_r_exp(prover);
        let stage_2_proof = self.stage2(prover, r_exp_lookup_bits, &r_indices, &lut_data);

        let stage_3_proof = self.stage3(prover, &lut_data, sat_diff_lookup_bits, &r_exp_indices);

        let stage_4_proof = self.stage4(prover, &lut_data, &sat_diff_indices);

        vec![
            (ProofId(self.idx(), ProofType::SoftmaxStage1), stage_1_proof),
            (ProofId(self.idx(), ProofType::SoftmaxStage2), stage_2_proof),
            (ProofId(self.idx(), ProofType::SoftmaxStage3), stage_3_proof),
            (ProofId(self.idx(), ProofType::SoftmaxStage4), stage_4_proof),
        ]
    }
}

/// Verifier for softmax-last-axis operations.
///
/// Mirror of [`SoftmaxLastAxisProver`] — each stage method corresponds to a
/// prover stage and verifies the associated sumcheck proof.
pub(crate) struct SoftmaxLastAxisVerifier {
    computation_node: ComputationNode,
    scale: i32,
    F_N: [usize; 2],
    exp_sum: Vec<i32>,
    max_k: Vec<i32>,
    argmax_k: Vec<usize>,
}

impl SoftmaxLastAxisVerifier {
    #[inline]
    pub(crate) fn idx(&self) -> usize {
        self.computation_node.idx
    }

    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::verify", skip_all)]
    /// Run the full verification pipeline (non-ZK).
    pub(crate) fn verify<F: JoltField, T: Transcript>(
        &mut self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        proofs: &BTreeMap<ProofId, SumcheckInstanceProof<F, T>>,
        scale_bits: i32,
    ) -> Result<(), ProofVerifyError> {
        self.cache_exp_sum(accumulator, transcript)?;
        self.cache_R(accumulator, transcript);
        self.run_stage(
            ProofType::SoftmaxStage1,
            self.build_stage1_verifiers(accumulator, transcript, scale_bits)?,
            proofs,
            accumulator,
            transcript,
        )?;

        let lut = VerifierLookupTableData::new(self.scale);

        self.cache_r_exp(accumulator, transcript);
        self.run_stage(
            ProofType::SoftmaxStage2,
            self.build_stage2_verifiers(accumulator, transcript, scale_bits, &lut),
            proofs,
            accumulator,
            transcript,
        )?;

        self.run_stage(
            ProofType::SoftmaxStage3,
            self.build_stage3_verifiers(accumulator, transcript, scale_bits, &lut),
            proofs,
            accumulator,
            transcript,
        )?;

        self.operand_link(accumulator, &lut)?;

        self.run_stage(
            ProofType::SoftmaxStage4,
            self.build_stage4_verifiers(accumulator, transcript, &lut),
            proofs,
            accumulator,
            transcript,
        )?;
        Ok(())
    }

    /// Drive a single batched-sumcheck stage given pre-built verifier instances.
    fn run_stage<F: JoltField, T: Transcript>(
        &self,
        proof_type: ProofType,
        instances: Vec<Box<dyn SumcheckInstanceVerifier<F, T>>>,
        proofs: &BTreeMap<ProofId, SumcheckInstanceProof<F, T>>,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<(), ProofVerifyError> {
        let proof = proofs
            .get(&ProofId(self.idx(), proof_type))
            .ok_or(ProofVerifyError::MissingProof(self.idx()))?;
        let refs: Vec<&dyn SumcheckInstanceVerifier<F, T>> =
            instances.iter().map(|b| b.as_ref()).collect();
        BatchedSumcheck::verify(proof, refs, accumulator, transcript)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SoftmaxLastAxisProver Methods
// ---------------------------------------------------------------------------

impl SoftmaxLastAxisProver {
    pub(crate) fn new(node: &ComputationNode, trace: SoftmaxLastAxisTrace, scale: i32) -> Self {
        let (&n, leading_dims) = node
            .output_dims
            .split_last()
            .expect("softmax node must have at least one output dimension");
        let f = leading_dims.iter().product::<usize>();
        Self {
            computation_node: node.clone(),
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
    pub(crate) fn send_auxiliary_vectors<F: JoltField, T: Transcript>(
        &self,
        prover: &mut Prover<F, T>,
    ) {
        let [f, _] = self.F_N;
        let mut provider = AccOpeningAccessor::new(&mut prover.accumulator, &self.computation_node)
            .into_provider(&mut prover.transcript, OpeningPoint::default());
        for k in 0..f {
            provider.append_advice(
                |idx| VirtualPoly::SoftmaxSumOutput(idx, k),
                F::from_u32(self.trace.exp_sum_q[k] as u32),
            );
            provider.append_advice(
                |idx| VirtualPoly::SoftmaxMaxOutput(idx, k),
                F::from_u32(self.trace.max_k[k] as u32),
            );
            provider.append_advice(
                |idx| VirtualPoly::SoftmaxMaxIndex(idx, k),
                F::from_u32(self.trace.argmax_k[k] as u32),
            );
        }
    }

    /// Cache `exp_sum_q(r0_lead)` as a virtual polynomial opening.
    ///
    /// Evaluates the prover-sent `exp_sum_q` vector at the leading part of
    /// `r0` (the initial output opening point).
    #[tracing::instrument(name = "SoftmaxLastAxisProver::cache_exp_sum", skip_all)]
    pub(crate) fn cache_exp_sum<F: JoltField, T: Transcript>(&self, prover: &mut Prover<F, T>) {
        let accessor = AccOpeningAccessor::new(&mut prover.accumulator, &self.computation_node);
        let r0 = accessor.get_reduced_opening().0;
        let log_f = self.F_N[0].log_2();
        let r_lead = r0.split_at(log_f).0;
        let eval = MultilinearPolynomial::from(self.trace.exp_sum_q.clone()).evaluate(&r_lead.r);
        let mut provider = accessor.into_provider(&mut prover.transcript, r_lead);
        provider.append_advice(VirtualPoly::SoftmaxExpSum, eval);
    }

    /// Cache the R (remainder) polynomial to the accumulator.
    pub(crate) fn cache_R<F: JoltField, T: Transcript>(&self, prover: &mut Prover<F, T>) {
        let accessor = AccOpeningAccessor::new(&mut prover.accumulator, &self.computation_node);
        let r0 = accessor.get_reduced_opening().0;
        let eval = MultilinearPolynomial::from(self.trace.R.clone()).evaluate(&r0.r);
        let mut provider = accessor.into_provider(&mut prover.transcript, r0);
        provider.append_advice(VirtualPoly::SoftmaxRecipMultRemainder, eval);
    }

    #[tracing::instrument(name = "SoftmaxLastAxisProver::stage1", skip_all)]
    fn stage1<F: JoltField, T: Transcript>(
        &mut self,
        prover: &mut Prover<F, T>,
        r_lookup_bits: Vec<LookupBits>,
    ) -> SumcheckInstanceProof<F, T> {
        let mut instances = self.build_stage1_instances(prover, r_lookup_bits);
        run_batched_prove(&mut instances, prover)
    }

    /// Build stage 1 sumcheck instances without running the sumcheck.
    pub(crate) fn build_stage1_instances<F: JoltField, T: Transcript>(
        &mut self,
        prover: &mut Prover<F, T>,
        r_lookup_bits: Vec<LookupBits>,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, T>>> {
        #[cfg_attr(not(feature = "zk"), allow(unused_mut))]
        let mut recip_mult_params = RecipMultParams::new(
            self.computation_node.clone(),
            self.scale,
            self.F_N,
            &prover.accumulator,
            &mut prover.transcript,
        );
        #[cfg(feature = "zk")]
        {
            recip_mult_params.inv_sum_evals =
                self.trace.inv_sum.iter().map(|&x| F::from_i32(x)).collect();
        }
        let exp_q_for_sum = self.trace.exp_q.clone();
        let recip_mult_prover = RecipMultProver::initialize(
            std::mem::take(&mut self.trace.exp_q),
            std::mem::take(&mut self.trace.inv_sum),
            recip_mult_params,
        );

        let exp_sum_params = ExpSumParams::new(
            self.computation_node.clone(),
            self.F_N,
            &prover.accumulator,
            &mut prover.transcript,
        );
        let exp_sum_prover = ExpSumProver::initialize(exp_q_for_sum, exp_sum_params);

        let provider = SoftmaxRCProvider::remainder(
            self.computation_node.clone(),
            prover.preprocessing.scale(),
        );
        let rc_R_prover =
            identity_rangecheck_prover(&provider, r_lookup_bits, &mut prover.accumulator);

        vec![
            Box::new(recip_mult_prover),
            Box::new(exp_sum_prover),
            Box::new(rc_R_prover),
        ]
    }

    /// Cache the r_exp polynomial to the accumulator.
    #[tracing::instrument(name = "SoftmaxLastAxisProver::cache_r_exp", skip_all)]
    pub(crate) fn cache_r_exp<F: JoltField, T: Transcript>(&self, prover: &mut Prover<F, T>) {
        let accessor = AccOpeningAccessor::new(&mut prover.accumulator, &self.computation_node);
        let r1 = accessor.get_advice(VirtualPoly::SoftmaxExpQ).0;
        let eval =
            MultilinearPolynomial::from(self.trace.decomposed_exp.r_exp.clone()).evaluate(&r1.r);
        let mut provider = accessor.into_provider(&mut prover.transcript, r1);
        provider.append_advice(VirtualPoly::SoftmaxExpRemainder, eval);
    }

    #[tracing::instrument(name = "SoftmaxLastAxisProver::stage2", skip_all)]
    fn stage2<F: JoltField, T: Transcript>(
        &mut self,
        prover: &mut Prover<F, T>,
        r_exp_lookup_bits: Vec<LookupBits>,
        r_indices: &[usize],
        lut_data: &LookupTableData,
    ) -> SumcheckInstanceProof<F, T> {
        let mut instances =
            self.build_stage2_instances(prover, r_exp_lookup_bits, r_indices, lut_data);
        run_batched_prove(&mut instances, prover)
    }

    /// Build stage 2 sumcheck instances without running the sumcheck.
    pub(crate) fn build_stage2_instances<F: JoltField, T: Transcript>(
        &mut self,
        prover: &mut Prover<F, T>,
        r_exp_lookup_bits: Vec<LookupBits>,
        r_indices: &[usize],
        lut_data: &LookupTableData,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, T>>> {
        let [f, _n] = self.F_N;
        let log_f = f.log_2();

        let accessor = AccOpeningAccessor::new(&prover.accumulator, &self.computation_node);
        let r1 = accessor.get_advice(VirtualPoly::SoftmaxExpQ).0;
        let r1_k = r1.split_at(log_f).0;

        let max_k_eval = MultilinearPolynomial::from(self.trace.max_k.clone()).evaluate(&r1_k.r);
        let max_indicator_params = MaxIndicatorParams::new(
            self.computation_node.clone(),
            self.F_N,
            std::mem::take(&mut self.trace.argmax_k),
            max_k_eval,
            &prover.accumulator,
        );
        let max_indicator_prover =
            MaxIndicatorProver::initialize(std::mem::take(&mut self.trace.x), max_indicator_params);

        let exp_mult_params = MultParams::new(
            self.computation_node.clone(),
            self.scale,
            &prover.accumulator,
        );
        let exp_mult_prover = MultProver::initialize(
            std::mem::take(&mut self.trace.decomposed_exp.exp_hi),
            std::mem::take(&mut self.trace.decomposed_exp.exp_lo),
            exp_mult_params,
        );

        let provider = SoftmaxRCProvider::exp_remainder(
            self.computation_node.clone(),
            prover.preprocessing.scale(),
        );
        let exp_r_rc_prover =
            identity_rangecheck_prover(&provider, r_exp_lookup_bits, &mut prover.accumulator);

        let encoding = SoftmaxRaEncoding::remainder(self.idx(), prover.preprocessing.scale());
        let [R_ra_prover, R_hw_prover, R_bool_prover] = shout::ra_onehot_provers(
            &encoding,
            r_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );

        let sd_params = SatDiffSlacknessParams::new(
            self.computation_node.clone(),
            lut_data.z_bound_minus_1,
            lut_data.base,
            &prover.accumulator,
        );
        let sd_prover = SatDiffSlacknessProver::initialize(
            &self.trace.decomposed_exp.sat_diff,
            &self.trace.decomposed_exp.z_hi,
            &self.trace.decomposed_exp.z_lo,
            sd_params,
        );

        vec![
            Box::new(exp_mult_prover),
            Box::new(max_indicator_prover),
            Box::new(exp_r_rc_prover),
            R_ra_prover,
            R_hw_prover,
            R_bool_prover,
            Box::new(sd_prover),
        ]
    }

    #[tracing::instrument(name = "SoftmaxLastAxisProver::stage3", skip_all)]
    fn stage3<F: JoltField, T: Transcript>(
        &self,
        prover: &mut Prover<F, T>,
        lut_data: &LookupTableData,
        sat_diff_lookup_bits: Vec<LookupBits>,
        r_exp_indices: &[usize],
    ) -> SumcheckInstanceProof<F, T> {
        let mut instances =
            self.build_stage3_instances(prover, lut_data, sat_diff_lookup_bits, r_exp_indices);
        run_batched_prove(&mut instances, prover)
    }

    /// Build stage 3 sumcheck instances without running the sumcheck.
    pub(crate) fn build_stage3_instances<F: JoltField, T: Transcript>(
        &self,
        prover: &mut Prover<F, T>,
        lut_data: &LookupTableData,
        sat_diff_lookup_bits: Vec<LookupBits>,
        r_exp_indices: &[usize],
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, T>>> {
        let hi_provider = ExpReadRafProvider {
            node: self.computation_node.clone(),
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

        let lo_provider = ExpReadRafProvider {
            node: self.computation_node.clone(),
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

        let provider = SoftmaxRCProvider::sat_diff(self.computation_node.clone());
        let sat_diff_rc_prover =
            identity_rangecheck_prover(&provider, sat_diff_lookup_bits, &mut prover.accumulator);

        let encoding = SoftmaxRaEncoding::exp_remainder(self.idx(), prover.preprocessing.scale());
        let [exp_r_ra_prover, exp_r_hw_prover, exp_r_bool_prover] = shout::ra_onehot_provers(
            &encoding,
            r_exp_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );

        vec![
            hi_prover,
            lo_prover,
            Box::new(sat_diff_rc_prover),
            exp_r_ra_prover,
            exp_r_hw_prover,
            exp_r_bool_prover,
        ]
    }

    fn stage4<F: JoltField, T: Transcript>(
        &self,
        prover: &mut Prover<F, T>,
        lut_data: &LookupTableData,
        sat_diff_indices: &[usize],
    ) -> SumcheckInstanceProof<F, T> {
        let mut instances = self.build_stage4_instances(prover, lut_data, sat_diff_indices);
        run_batched_prove(&mut instances, prover)
    }

    /// Build stage 4 sumcheck instances without running the sumcheck.
    pub(crate) fn build_stage4_instances<F: JoltField, T: Transcript>(
        &self,
        prover: &mut Prover<F, T>,
        lut_data: &LookupTableData,
        sat_diff_indices: &[usize],
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, T>>> {
        let encoding = SoftmaxRaEncoding::exp_hi(self.idx(), lut_data.table_hi.len().log_2());
        let [hi_ra_prover, hi_hw_prover, hi_bool_prover] = shout::ra_onehot_provers(
            &encoding,
            &lut_data.z_hi_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );

        let encoding = SoftmaxRaEncoding::exp_lo(self.idx(), lut_data.table_lo.len().log_2());
        let [lo_ra_prover, lo_hw_prover, lo_bool_prover] = shout::ra_onehot_provers(
            &encoding,
            &lut_data.z_lo_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );

        let encoding = SoftmaxRaEncoding::sat_diff(self.idx());
        let [sat_diff_ra_prover, sat_diff_hw_prover, sat_diff_bool_prover] =
            shout::ra_onehot_provers(
                &encoding,
                sat_diff_indices,
                &prover.accumulator,
                &mut prover.transcript,
            );

        vec![
            hi_ra_prover,
            hi_hw_prover,
            hi_bool_prover,
            lo_ra_prover,
            lo_hw_prover,
            lo_bool_prover,
            sat_diff_ra_prover,
            sat_diff_hw_prover,
            sat_diff_bool_prover,
        ]
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
    pub(crate) fn new<F: JoltField, T: Transcript>(
        node: &ComputationNode,
        scale: i32,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Self {
        let (&n, leading_dims) = node
            .output_dims
            .split_last()
            .expect("softmax node must have at least one output dimension");
        let f = leading_dims.iter().product::<usize>();

        let accessor = AccOpeningAccessor::new(&mut *accumulator, node);
        let mut provider = accessor.into_provider(&mut *transcript, OpeningPoint::default());
        for k in 0..f {
            provider.append_advice(|idx| VirtualPoly::SoftmaxSumOutput(idx, k));
            provider.append_advice(|idx| VirtualPoly::SoftmaxMaxOutput(idx, k));
            provider.append_advice(|idx| VirtualPoly::SoftmaxMaxIndex(idx, k));
        }

        /// Read an auxiliary scalar vector from the transcript.
        fn read_aux_scalars<F: JoltField>(
            accessor: &AccOpeningAccessor<'_, F, VerifierOpeningAccumulator<F>>,
            f: usize,
            vp_fn: fn(usize, usize) -> VirtualPoly,
        ) -> Vec<u64> {
            (0..f)
                .map(|k| {
                    accessor
                        .get_advice(|idx| vp_fn(idx, k))
                        .1
                        .to_u64()
                        .expect("auxiliary scalar should fit within 64 bits")
                })
                .collect()
        }

        let accessor = AccOpeningAccessor::new(&*accumulator, node);

        let exp_sum = read_aux_scalars(&accessor, f, VirtualPoly::SoftmaxSumOutput)
            .into_iter()
            .map(|v| v as i32)
            .collect();
        let max_k = read_aux_scalars(&accessor, f, VirtualPoly::SoftmaxMaxOutput)
            .into_iter()
            .map(|v| v as i32)
            .collect();
        let argmax_k = read_aux_scalars(&accessor, f, VirtualPoly::SoftmaxMaxIndex)
            .into_iter()
            .map(|v| v as usize)
            .collect();

        Self {
            computation_node: node.clone(),
            scale,
            F_N: [f, n],
            exp_sum,
            max_k,
            argmax_k,
        }
    }

    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::inv_sum_evals", skip_all)]
    pub(crate) fn inv_sum_evals<F: JoltField>(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Result<Vec<F>, ProofVerifyError> {
        let [f, _] = self.F_N;
        let s_squared = (self.scale as i64) * (self.scale as i64);
        let accessor = AccOpeningAccessor::new(accumulator, &self.computation_node);
        (0..f)
            .map(|k| {
                let exp_sum_q_k = accessor
                    .get_advice(|idx| VirtualPoly::SoftmaxSumOutput(idx, k))
                    .1;
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
    pub(crate) fn cache_exp_sum<F: JoltField, T: Transcript>(
        &mut self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<(), ProofVerifyError> {
        let accessor = AccOpeningAccessor::new(accumulator, &self.computation_node);
        let r0 = accessor.get_reduced_opening().0;
        let log_f = self.F_N[0].log_2();
        let r_lead = r0.split_at(log_f).0;
        let mut provider = accessor.into_provider(transcript, r_lead.clone());
        provider.append_advice(VirtualPoly::SoftmaxExpSum);
        let exp_sum_eval =
            MultilinearPolynomial::from(std::mem::take(&mut self.exp_sum)).evaluate(&r_lead.r);
        let claimed_exp_sum_eval = provider.get_advice(VirtualPoly::SoftmaxExpSum).1;
        if exp_sum_eval != claimed_exp_sum_eval {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "exp_sum evaluation mismatch".to_string(),
            ));
        }
        Ok(())
    }

    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::cache_R", skip_all)]
    /// Append the remainder polynomial commitment to the accumulator (non-ZK).
    pub(crate) fn cache_R<F: JoltField, T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) {
        let accessor = AccOpeningAccessor::new(accumulator, &self.computation_node);
        let r = accessor.get_reduced_opening().0;
        let mut provider = accessor.into_provider(transcript, r);
        provider.append_advice(VirtualPoly::SoftmaxRecipMultRemainder);
    }

    /// Cache the remainder polynomial in the ZK pipeline.
    ///
    /// In ZK mode the verifier does not see the claim value (it is private and
    /// Pedersen-committed by the prover). This method mirrors [`cache_R`] but:
    /// - inserts an explicit `(r0, F::zero())` placeholder at the same `OpeningId`
    ///   so subsequent stage-1 verifiers that read this opening get an explicit
    ///   placeholder (rather than relying on the accumulator's permissive
    ///   `zk_mode` fallback), and
    /// - absorbs the Pedersen commitment into the transcript (the prover side
    ///   appends the same commitment in `prove_softmax_zk`).
    #[cfg(feature = "zk")]
    pub(crate) fn cache_R_zk<F: JoltField, T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        commitment: &impl ark_serialize::CanonicalSerialize,
    ) {
        use joltworks::poly::opening_proof::{OpeningId, SumcheckId};
        let accessor = AccOpeningAccessor::new(&*accumulator, &self.computation_node);
        let r0 = accessor.get_reduced_opening().0;
        let opening_id = OpeningId::new(
            VirtualPoly::SoftmaxRecipMultRemainder(self.idx()),
            SumcheckId::NodeExecution(self.idx()),
        );
        accumulator.openings.insert(opening_id, (r0, F::zero()));
        transcript.append_serializable(commitment);
    }

    /// Build stage 1 verifier instances. The caller drives the actual sumcheck.
    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::build_stage1_verifiers", skip_all)]
    pub(crate) fn build_stage1_verifiers<F: JoltField, T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        scale_bits: i32,
    ) -> Result<Vec<Box<dyn SumcheckInstanceVerifier<F, T>>>, ProofVerifyError> {
        let inv_sum = self.inv_sum_evals(&*accumulator)?;
        let recip_mult_verifier = RecipMultVerifier::new(
            self.computation_node.clone(),
            self.scale,
            self.F_N,
            inv_sum,
            &*accumulator,
            transcript,
        );
        let exp_sum_verifier = ExpSumVerifier::new(
            self.computation_node.clone(),
            self.F_N,
            &*accumulator,
            transcript,
        );
        let rc_provider =
            SoftmaxRCProvider::remainder(self.computation_node.clone(), scale_bits);
        let rc_R_verifier = identity_rangecheck_verifier(&rc_provider, accumulator);
        Ok(vec![
            Box::new(recip_mult_verifier),
            Box::new(exp_sum_verifier),
            Box::new(rc_R_verifier),
        ])
    }

    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::cache_r_exp", skip_all)]
    pub(crate) fn cache_r_exp<F: JoltField, T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) {
        let accessor = AccOpeningAccessor::new(accumulator, &self.computation_node);
        let r = accessor.get_advice(VirtualPoly::SoftmaxExpQ).0;
        let mut provider = accessor.into_provider(transcript, r);
        provider.append_advice(VirtualPoly::SoftmaxExpRemainder);
    }

    /// Cache the exp-remainder polynomial in the ZK pipeline.
    ///
    /// Same shape as [`cache_R_zk`]: insert an explicit `(r1, F::zero())`
    /// placeholder for the private `SoftmaxExpRemainder` opening and absorb
    /// the Pedersen commitment from the bundle into the transcript.
    #[cfg(feature = "zk")]
    pub(crate) fn cache_r_exp_zk<F: JoltField, T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        commitment: &impl ark_serialize::CanonicalSerialize,
    ) {
        use joltworks::poly::opening_proof::{OpeningId, SumcheckId};
        let accessor = AccOpeningAccessor::new(&*accumulator, &self.computation_node);
        let r1 = accessor.get_advice(VirtualPoly::SoftmaxExpQ).0;
        let opening_id = OpeningId::new(
            VirtualPoly::SoftmaxExpRemainder(self.idx()),
            SumcheckId::NodeExecution(self.idx()),
        );
        accumulator.openings.insert(opening_id, (r1, F::zero()));
        transcript.append_serializable(commitment);
    }

    /// Build stage 2 verifier instances.
    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::build_stage2_verifiers", skip_all)]
    pub(crate) fn build_stage2_verifiers<F: JoltField, T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        scale_bits: i32,
        lut: &VerifierLookupTableData,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> {
        let [f, _n] = self.F_N;
        let log_f = f.log_2();

        let accessor = AccOpeningAccessor::new(&*accumulator, &self.computation_node);
        let r1 = accessor.get_advice(VirtualPoly::SoftmaxExpQ).0;
        let r1_k = r1.split_at(log_f).0;

        let max_k_eval = MultilinearPolynomial::from(self.max_k.clone()).evaluate(&r1_k.r);
        let max_indicator_verifier = MaxIndicatorVerifier::new(
            self.computation_node.clone(),
            self.F_N,
            self.argmax_k.clone(),
            max_k_eval,
            &*accumulator,
        );

        let exp_mult_verifier =
            MultVerifier::new(self.computation_node.clone(), self.scale, &*accumulator);

        let rc_provider =
            SoftmaxRCProvider::exp_remainder(self.computation_node.clone(), scale_bits);
        let exp_r_rc_verifier = identity_rangecheck_verifier(&rc_provider, accumulator);

        let encoding = SoftmaxRaEncoding::remainder(self.idx(), scale_bits);
        let [R_ra_verifier, R_hw_verifier, R_bool_verifier] =
            shout::ra_onehot_verifiers(&encoding, &*accumulator, transcript);

        let sd_verifier = SatDiffSlacknessVerifier::new(
            self.computation_node.clone(),
            lut.z_bound_minus_1,
            lut.base,
            &*accumulator,
        );

        vec![
            Box::new(exp_mult_verifier),
            Box::new(max_indicator_verifier),
            Box::new(exp_r_rc_verifier),
            R_ra_verifier,
            R_hw_verifier,
            R_bool_verifier,
            Box::new(sd_verifier),
        ]
    }

    /// Build stage 3 verifier instances.
    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::build_stage3_verifiers", skip_all)]
    pub(crate) fn build_stage3_verifiers<F: JoltField, T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        scale_bits: i32,
        lut: &VerifierLookupTableData,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> {
        let provider = ExpReadRafProvider {
            node: self.computation_node.clone(),
            table_size: lut.table_hi.len(),
            digit: ExpDigit::Hi,
        };
        let hi_verifier =
            shout::read_raf_verifier(&provider, lut.table_hi.clone(), &*accumulator, transcript);

        let provider = ExpReadRafProvider {
            node: self.computation_node.clone(),
            table_size: lut.table_lo.len(),
            digit: ExpDigit::Lo,
        };
        let lo_verifier =
            shout::read_raf_verifier(&provider, lut.table_lo.clone(), &*accumulator, transcript);

        let provider = SoftmaxRCProvider::sat_diff(self.computation_node.clone());
        let rc_sat_diff_verifier = identity_rangecheck_verifier(&provider, accumulator);

        let encoding = SoftmaxRaEncoding::exp_remainder(self.idx(), scale_bits);
        let [r_exp_ra_verifier, r_exp_hw_verifier, r_exp_bool_verifier] =
            shout::ra_onehot_verifiers(&encoding, &*accumulator, transcript);

        vec![
            hi_verifier,
            lo_verifier,
            Box::new(rc_sat_diff_verifier),
            r_exp_ra_verifier,
            r_exp_hw_verifier,
            r_exp_bool_verifier,
        ]
    }

    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::operand_link", skip_all)]
    /// Stage 3 operand link: derive `X(r2)` algebraically and route upstream.
    ///
    /// `X(r2) = max_k(r2_lead) − z_c(r2) − sat_diff(r2)`
    /// where `z_c(r2) = z_hi(r2)·B + z_lo(r2)`.
    pub(crate) fn operand_link<F: JoltField>(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        lut: &VerifierLookupTableData,
    ) -> Result<(), ProofVerifyError> {
        let [f, _n] = self.F_N;
        let log_f = f.log_2();
        let accessor = AccOpeningAccessor::new(accumulator, &self.computation_node);

        let r2 = accessor.get_advice(VirtualPoly::SoftmaxExpHi).0;
        let r2_lead = r2.split_at(log_f).0;

        let max_k_eval = MultilinearPolynomial::from(self.max_k.clone()).evaluate(&r2_lead.r);

        let z_hi_eval = accessor.get_advice(VirtualPoly::SoftmaxZHi).1;
        let z_lo_eval = accessor.get_advice(VirtualPoly::SoftmaxZLo).1;
        let z_c_eval = z_hi_eval * F::from_u64(lut.base) + z_lo_eval;

        let sat_diff_eval = accessor.get_advice(VirtualPoly::SoftmaxSatDiff).1;

        let x_r2 = max_k_eval - z_c_eval - sat_diff_eval;

        let prover_x_r2 = accessor.get_nodeio(Target::Input(0)).1;
        if prover_x_r2 != x_r2 {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Operand link failed: prover's X(r2) does not match max_k - z_c - sat_diff"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Build stage 4 verifier instances.
    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::build_stage4_verifiers", skip_all)]
    pub(crate) fn build_stage4_verifiers<F: JoltField, T: Transcript>(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        lut: &VerifierLookupTableData,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> {
        let encoding = SoftmaxRaEncoding::exp_hi(self.idx(), lut.table_hi.len().log_2());
        let [hi_ra_verifier, hi_hw_verifier, hi_bool_verifier] =
            shout::ra_onehot_verifiers(&encoding, accumulator, transcript);

        let encoding = SoftmaxRaEncoding::exp_lo(self.idx(), lut.table_lo.len().log_2());
        let [lo_ra_verifier, lo_hw_verifier, lo_bool_verifier] =
            shout::ra_onehot_verifiers(&encoding, accumulator, transcript);

        let encoding = SoftmaxRaEncoding::sat_diff(self.idx());
        let [sat_diff_ra_verifier, sat_diff_hw_verifier, sat_diff_bool_verifier] =
            shout::ra_onehot_verifiers(&encoding, accumulator, transcript);

        vec![
            hi_ra_verifier,
            hi_hw_verifier,
            hi_bool_verifier,
            lo_ra_verifier,
            lo_hw_verifier,
            lo_bool_verifier,
            sat_diff_ra_verifier,
            sat_diff_hw_verifier,
            sat_diff_bool_verifier,
        ]
    }
}

// ---------------------------------------------------------------------------
// Small helpers used across prover stages
// ---------------------------------------------------------------------------

/// Convert `i32` trace values to usize indices.
pub(crate) fn to_indices(values: &[i32]) -> Vec<usize> {
    values.iter().map(|&v| v as usize).collect()
}

/// Convert `i32` trace values to `LookupBits` with the given bit-width.
pub(crate) fn to_lookup_bits(values: &[i32], bits: usize) -> Vec<LookupBits> {
    values
        .iter()
        .map(|&v| LookupBits::new(v as u64, bits))
        .collect()
}

/// Pad a vector in-place to the next power of two with zeros.
pub(crate) fn pad_to_power_of_two(v: &mut Vec<i32>) {
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

/// Pre-computed lookup table data shared between stage3 and stage4.
pub(crate) struct LookupTableData {
    pub(crate) table_hi: Vec<i32>,
    pub(crate) table_lo: Vec<i32>,
    pub(crate) z_hi_indices: Vec<usize>,
    pub(crate) z_lo_indices: Vec<usize>,
    /// `z_bound - 1 = K_hi * B - 1` (unpadded).
    pub(crate) z_bound_minus_1: u64,
    /// Digit base B.
    pub(crate) base: u64,
}

/// Pre-computed verifier lookup table data (shared across stage3, operand_link, stage4).
pub(crate) struct VerifierLookupTableData {
    pub(crate) table_hi: Vec<i32>,
    pub(crate) table_lo: Vec<i32>,
    pub(crate) base: u64,
    /// `z_bound − 1 = K_hi * B − 1`, the maximum clamped logit value.
    /// Used in the complementary-slackness check.
    pub(crate) z_bound_minus_1: u64,
}

impl VerifierLookupTableData {
    pub(crate) fn new(scale: i32) -> Self {
        let decomp = generate_exp_lut_decomposed(scale);
        let z_bound_minus_1 = (decomp.lut_hi.len() * decomp.base - 1) as u64;
        let mut table_hi = decomp.lut_hi;
        table_hi.resize(table_hi.len().next_power_of_two(), 0);
        let mut table_lo = decomp.lut_lo;
        table_lo.resize(table_lo.len().next_power_of_two(), 0);
        Self {
            table_hi,
            table_lo,
            base: decomp.base as u64,
            z_bound_minus_1,
        }
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
