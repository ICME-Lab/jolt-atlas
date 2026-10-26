//! Proofs for softmax with the last axis as the reduction dimension (ONNX Softmax operator).
//! For more on the claim flow and design of the protocol, see: https://hackmd.io/@R5TO3fi7TlWQKnY6Ejr2eg/Byjucoqjbx
//!
//! The prover sends three auxiliary vectors (`max_k`, `exp_sum_q`, `argmax_k`) to the
//! verifier via the transcript. These let the verifier derive `inv_sum[k]` without a
//! committed polynomial, but they add O(F) field elements to the proof size.
//!
//! TODO(#218): Remove auxiliary vectors and derive them inside the protocol.
//!
//! The saturating clamp `z_c = min(z, z_bound - 1)` (`z = max_k - x`) is proven via a
//! `ClampBoundedTable` lookup (`joltworks::lookup_tables::clamp::SoftmaxClampTable`, via
//! `significance_clamp::SoftmaxSignificanceClampOperands`) — see `significance_clamp.rs`'s module doc for the relation being
//! proven. This lookup only supports the compile-time `common::consts::MODEL_SCALE` (its bound is
//! a const generic), so `SoftmaxLastAxis` only supports that one scale at runtime.
use crate::onnx_proof::op_lookups::OpLookupProve;
use crate::onnx_proof::ops::ProveOp;
pub use jolt_atlas_verifier::onnx_proof::ops::softmax_last_axis::*;

use crate::{
    onnx_proof::{
        deferred_lookups::{DeferredBatch, DeferredOneHot},
        op_lookups::OpLookupProvider,
        ops::softmax_last_axis::{
            exp_sum::{ExpSumParams, ExpSumProver},
            exponentiation::{
                mult::{MultParams, MultProver},
                ExpDigit, ExpReadRafProvider,
            },
            max::{MaxIndicatorParams, MaxIndicatorProver},
            rc::{SoftmaxRCProvider, SoftmaxRaEncoding},
            recip_mult::{RecipMultParams, RecipMultProver},
            significance_clamp::SoftmaxSignificanceClampOperands,
        },
        trace_view::TraceView,
        ProofId, ProofType, Prover,
    },
    utils::{compute_lookup_indices_from_operands, opening_access::AccOpeningAccessor},
};
use joltworks::lookup_tables::clamp::SoftmaxClampTable;

use atlas_trace::{op::SoftmaxLastAxis, Tensor};

use crate::node_view::ComputationNode;
use common::{
    consts::{MODEL_SCALE, XLEN},
    VirtualPoly,
};
use joltworks::{
    field::JoltField,
    poly::{multilinear_polynomial::EvaluateMle, opening_proof::OpeningPoint},
    subprotocols::{
        identity_range_check::identity_rangecheck_prover,
        shout,
        sumcheck::{BatchedSumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
    },
    transcripts::Transcript,
    utils::{lookup_bits::LookupBits, math::Math},
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
/// The saturating-clamp lookup helper (`SoftmaxSignificanceClampOperands`) and its `ClampSpec` table.
pub mod significance_clamp;

impl<F: JoltField, T: Transcript> ProveOp<F, T> for SoftmaxLastAxis {
    #[tracing::instrument(skip_all, name = "SoftmaxLastAxis::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        debug_assert_eq!(
            node.scale, MODEL_SCALE as u8,
            "SoftmaxLastAxis only supports the currently-compiled MODEL_SCALE={MODEL_SCALE}"
        );
        // The masked (non-full-row) softmax needs a selector polynomial in
        // these sumchecks; until that lands, only full rows are provable —
        // the same capability the old pipeline actually had.
        let last_dim = *node.output_dims.last().expect("softmax rank >= 1");
        assert!(
            self.active_len.is_none_or(|n| n == last_dim),
            "masked softmax (active_len {:?} of {last_dim}) is not yet provable; \
             use a power-of-two softmax axis",
            self.active_len
        );
        let scale = 1i32 << node.scale;
        let witness = prover
            .trace
            .softmax_witness(node.idx)
            .unwrap_or_else(|| panic!("softmax node {} has no trace witness", node.idx));
        SoftmaxLastAxisProver::new(node, witness.clone(), scale).prove(prover)
    }
}

pub(crate) struct SoftmaxLastAxisProver {
    pub(crate) computation_node: ComputationNode,
    pub(crate) scale: i32,
    pub(crate) F_N: [usize; 2],
    pub(crate) witness: atlas_trace::SoftmaxWitness,
    pub(crate) lut: ExpLutDecomposed,
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
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        let scale_bits = prover.preprocessing.scale();

        // Invariant guard: the SatClamp commitment is sized in
        // `get_committed_polynomials` from the per-node operator scale
        // (`self.scale.ilog2()`), while the prover/verifier range checks use the
        // model-wide `preprocessing.scale()`. A single global scale makes these
        // identical; assert it so a future move to per-node scales fails loudly
        // here instead of silently mismatching commitment and range-check widths.
        debug_assert_eq!(
            self.scale.ilog2() as u8,
            scale_bits,
            "softmax operator scale (log2 {}) disagrees with preprocessing scale {scale_bits}",
            self.scale.ilog2(),
        );

        // ── Pre-compute index/lookup data from trace vectors ────────────

        // R-derived data (consumed by stage1 RC + stage2 one-hot)
        let r_lookup_bits = to_lookup_bits(&self.witness.r, scale_bits as usize);
        let r_indices = to_indices(&self.witness.r);

        // r_exp-derived data (consumed by stage2 RC + stage3 one-hot)
        let r_exp_lookup_bits = to_lookup_bits(&self.witness.r_exp, scale_bits as usize);
        let r_exp_indices = to_indices(&self.witness.r_exp);

        // z_hi/z_lo indices (consumed by stage3 + stage4)
        let z_hi_indices = to_indices(&self.witness.z_hi);
        let z_lo_indices = to_indices(&self.witness.z_lo);

        // Padded lookup tables (consumed by stage3 Shout + stage4 one-hot)
        let mut table_hi = std::mem::take(&mut self.lut.lut_hi);
        pad_to_power_of_two(&mut table_hi);
        let mut table_lo = std::mem::take(&mut self.lut.lut_lo);
        pad_to_power_of_two(&mut table_lo);

        let lut_data = LookupTableData {
            table_hi,
            table_lo,
            z_hi_indices,
            z_lo_indices,
        };

        // z = max_k - x, the sat-clamp lookup's witness. Computed here (not inside
        // `build_stage3_instances`) because stage 2 takes ownership of `self.witness.x` (for
        // `max_indicator_prover`) — by the time stage 3 runs, `self.witness.x` is empty.
        let last_dim = self.F_N[1];
        let z = softmax_z(&self.witness.x, &self.witness.max_k, last_dim);
        let z_tensor = Tensor::from_slice(&z, &[z.len()])
            .expect("softmax_z tensor construction")
            .padded_pow2()
            .map(|v| v as i64);
        let significance_clamp_indices: Vec<usize> =
            compute_lookup_indices_from_operands(&[&z_tensor.map(|v| v as i32)], false)
                .iter()
                .map(|&x| x.into())
                .collect();

        // ── Pipeline ────────────────────────────────────────────────────

        self.send_auxiliary_vectors(prover);
        self.cache_exp_sum(prover);
        self.cache_R(prover);
        let stage_1_proof = self.stage1(prover, r_lookup_bits);

        self.cache_r_exp(prover);
        let stage_2_proof = self.stage2(prover, r_exp_lookup_bits, &r_indices);

        // z_hi/z_lo need to be cached fresh at r2 (established by stage 2's exp_mult instance).
        self.cache_z_hi_lo(prover);

        // Stage 3 (exp-digit lookups + significance-clamp read-raf) is built
        // now — its challenges are drawn and its claims appended here — and
        // proven after the node loop in the `SoftmaxStage3` batch; stage 4 (the
        // one-hot checks) follows that batch.
        let log_hi = lut_data.table_hi.len().log_2();
        let log_lo = lut_data.table_lo.len().log_2();
        for instance in self.build_stage3_instances(prover, &lut_data, &r_exp_indices, z_tensor) {
            prover.defer(DeferredBatch::SoftmaxStage3, instance);
        }
        let idx = self.idx();
        prover.deferred_onehots.push((
            DeferredOneHot::SoftmaxExpHi(idx, log_hi),
            lut_data.z_hi_indices,
        ));
        prover.deferred_onehots.push((
            DeferredOneHot::SoftmaxExpLo(idx, log_lo),
            lut_data.z_lo_indices,
        ));
        prover.deferred_onehots.push((
            DeferredOneHot::SoftmaxClamp(self.computation_node.clone()),
            significance_clamp_indices,
        ));

        vec![
            (ProofId(self.idx(), ProofType::SoftmaxStage1), stage_1_proof),
            (ProofId(self.idx(), ProofType::SoftmaxStage2), stage_2_proof),
        ]
    }
}

// ---------------------------------------------------------------------------
// SoftmaxLastAxisProver Methods
// ---------------------------------------------------------------------------

impl SoftmaxLastAxisProver {
    pub(crate) fn new(
        node: &ComputationNode,
        witness: atlas_trace::SoftmaxWitness,
        scale: i32,
    ) -> Self {
        let (&n, leading_dims) = node
            .output_dims
            .split_last()
            .expect("softmax node must have at least one output dimension");
        let f = leading_dims.iter().product::<usize>();
        Self {
            computation_node: node.clone(),
            scale,
            F_N: [f, n],
            lut: generate_exp_lut_decomposed(scale),
            witness,
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
                F::from_u32(self.witness.exp_sum_q[k] as u32),
            );
            provider.append_advice(
                |idx| VirtualPoly::SoftmaxMaxOutput(idx, k),
                F::from_u32(self.witness.max_k[k] as u32),
            );
            provider.append_advice(
                |idx| VirtualPoly::SoftmaxMaxIndex(idx, k),
                F::from_u32(self.witness.argmax_k[k] as u32),
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
        let eval = self.witness.exp_sum_q.evaluate_mle(&r_lead.r);
        let mut provider = accessor.into_provider(&mut prover.transcript, r_lead);
        provider.append_advice(VirtualPoly::SoftmaxExpSum, eval);
    }

    /// Cache the R (remainder) polynomial to the accumulator.
    pub(crate) fn cache_R<F: JoltField, T: Transcript>(&self, prover: &mut Prover<F, T>) {
        let accessor = AccOpeningAccessor::new(&mut prover.accumulator, &self.computation_node);
        let r0 = accessor.get_reduced_opening().0;
        let eval = self.witness.r.evaluate_mle(&r0.r);
        let mut provider = accessor.into_provider(&mut prover.transcript, r0);
        provider.append_advice(VirtualPoly::SoftmaxRecipMultRemainder, eval);
    }

    #[tracing::instrument(name = "SoftmaxLastAxisProver::stage1", skip_all)]
    fn stage1<F: JoltField, T: Transcript>(
        &mut self,
        prover: &mut Prover<F, T>,
        r_lookup_bits: Vec<LookupBits>,
    ) -> SumcheckInstanceProof<F> {
        let mut instances = self.build_stage1_instances(prover, r_lookup_bits);
        run_batched_prove(&mut instances, prover)
    }

    /// Build stage 1 sumcheck instances without running the sumcheck.
    pub(crate) fn build_stage1_instances<F: JoltField, T: Transcript>(
        &mut self,
        prover: &mut Prover<F, T>,
        r_lookup_bits: Vec<LookupBits>,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, T>>> {
        let recip_mult_params = RecipMultParams::new(
            self.computation_node.clone(),
            self.scale,
            self.F_N,
            &prover.accumulator,
            &mut prover.transcript,
        );
        let exp_q_for_sum = self.witness.exp_q.clone();
        let recip_mult_prover = RecipMultProver::initialize(
            std::mem::take(&mut self.witness.exp_q),
            std::mem::take(&mut self.witness.inv_sum),
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
        let eval = self.witness.r_exp.evaluate_mle(&r1.r);
        let mut provider = accessor.into_provider(&mut prover.transcript, r1);
        provider.append_advice(VirtualPoly::SoftmaxExpRemainder, eval);
    }

    #[tracing::instrument(name = "SoftmaxLastAxisProver::stage2", skip_all)]
    fn stage2<F: JoltField, T: Transcript>(
        &mut self,
        prover: &mut Prover<F, T>,
        r_exp_lookup_bits: Vec<LookupBits>,
        r_indices: &[usize],
    ) -> SumcheckInstanceProof<F> {
        let mut instances = self.build_stage2_instances(prover, r_exp_lookup_bits, r_indices);
        run_batched_prove(&mut instances, prover)
    }

    /// Build stage 2 sumcheck instances (the saturating clamp is a lookup, folded into stage
    /// 3/4 instead).
    pub(crate) fn build_stage2_instances<F: JoltField, T: Transcript>(
        &mut self,
        prover: &mut Prover<F, T>,
        r_exp_lookup_bits: Vec<LookupBits>,
        r_indices: &[usize],
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, T>>> {
        let [f, _n] = self.F_N;
        let log_f = f.log_2();

        let accessor = AccOpeningAccessor::new(&prover.accumulator, &self.computation_node);
        let r1 = accessor.get_advice(VirtualPoly::SoftmaxExpQ).0;
        let r1_k = r1.split_at(log_f).0;

        let max_k_eval = self.witness.max_k.evaluate_mle(&r1_k.r);
        let max_indicator_params = MaxIndicatorParams::new(
            self.computation_node.clone(),
            self.F_N,
            std::mem::take(&mut self.witness.argmax_k),
            max_k_eval,
            &prover.accumulator,
        );
        let max_indicator_prover = MaxIndicatorProver::initialize(
            std::mem::take(&mut self.witness.x),
            max_indicator_params,
        );

        let exp_mult_params = MultParams::new(
            self.computation_node.clone(),
            self.scale,
            &prover.accumulator,
        );
        let exp_mult_prover = MultProver::initialize(
            std::mem::take(&mut self.witness.exp_hi),
            std::mem::take(&mut self.witness.exp_lo),
            exp_mult_params,
        );

        let provider = SoftmaxRCProvider::exp_remainder(
            self.computation_node.clone(),
            prover.preprocessing.scale(),
        );
        let exp_r_rc_prover =
            identity_rangecheck_prover(&provider, r_exp_lookup_bits, &mut prover.accumulator);

        let encoding = SoftmaxRaEncoding::remainder(self.idx(), prover.preprocessing.scale());
        let [R_ra_prover, R_bool_prover] = shout::ra_onehot_provers(
            &encoding,
            r_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );

        vec![
            Box::new(exp_mult_prover),
            Box::new(max_indicator_prover),
            Box::new(exp_r_rc_prover),
            R_ra_prover,
            R_bool_prover,
        ]
    }

    /// Cache `z_hi`/`z_lo` fresh at `r2` (mirrors `cache_r_exp`) — `r2` is `SoftmaxExpHi`'s
    /// opening point, established by stage 2's `exp_mult_prover`.
    #[tracing::instrument(name = "SoftmaxLastAxisProver::cache_z_hi_lo", skip_all)]
    pub(crate) fn cache_z_hi_lo<F: JoltField, T: Transcript>(&self, prover: &mut Prover<F, T>) {
        let accessor = AccOpeningAccessor::new(&mut prover.accumulator, &self.computation_node);
        let r2 = accessor.get_advice(VirtualPoly::SoftmaxExpHi).0;
        let z_hi_eval = self.witness.z_hi.evaluate_mle(&r2.r);
        let z_lo_eval = self.witness.z_lo.evaluate_mle(&r2.r);
        let mut provider = accessor.into_provider(&mut prover.transcript, r2);
        provider.append_advice(VirtualPoly::SoftmaxZHi, z_hi_eval);
        provider.append_advice(VirtualPoly::SoftmaxZLo, z_lo_eval);
    }

    #[allow(dead_code)]
    #[tracing::instrument(name = "SoftmaxLastAxisProver::stage3", skip_all)]
    fn stage3<F: JoltField, T: Transcript>(
        &mut self,
        prover: &mut Prover<F, T>,
        lut_data: &LookupTableData,
        r_exp_indices: &[usize],
        z_tensor: Tensor<i64>,
    ) -> SumcheckInstanceProof<F> {
        let mut instances = self.build_stage3_instances(prover, lut_data, r_exp_indices, z_tensor);
        run_batched_prove(&mut instances, prover)
    }

    /// Build stage 3 sumcheck instances: exp-digit lookups plus the saturating-clamp lookup's
    /// execution proof. `z_tensor` is computed by the caller (`prove`), since by this point
    /// `self.witness.x` has already been consumed by stage 2's `max_indicator_prover`.
    pub(crate) fn build_stage3_instances<F: JoltField, T: Transcript>(
        &mut self,
        prover: &mut Prover<F, T>,
        lut_data: &LookupTableData,
        r_exp_indices: &[usize],
        z_tensor: Tensor<i64>,
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

        let significance_clamp_provider = OpLookupProvider::with_helper(
            self.computation_node.clone(),
            SoftmaxSignificanceClampOperands::new(z_tensor),
        );
        let (significance_clamp_prover, _) = significance_clamp_provider
            .read_raf_prove::<F, T, SoftmaxClampTable<XLEN>, XLEN>(
                &prover.trace,
                &mut prover.accumulator,
                &mut prover.transcript,
            );

        let encoding = SoftmaxRaEncoding::exp_remainder(self.idx(), prover.preprocessing.scale());
        let [exp_r_ra_prover, exp_r_bool_prover] = shout::ra_onehot_provers(
            &encoding,
            r_exp_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );

        vec![
            hi_prover,
            lo_prover,
            Box::new(significance_clamp_prover),
            exp_r_ra_prover,
            exp_r_bool_prover,
        ]
    }

    #[allow(dead_code)]
    fn stage4<F: JoltField, T: Transcript>(
        &mut self,
        prover: &mut Prover<F, T>,
        lut_data: &LookupTableData,
        significance_clamp_indices: &[usize],
    ) -> SumcheckInstanceProof<F> {
        let mut instances =
            self.build_stage4_instances(prover, lut_data, significance_clamp_indices);
        run_batched_prove(&mut instances, prover)
    }

    /// Build stage 4 sumcheck instances: exp-digit one-hot checks plus the saturating-clamp
    /// lookup's one-hot checks.
    #[allow(dead_code)]
    pub(crate) fn build_stage4_instances<F: JoltField, T: Transcript>(
        &mut self,
        prover: &mut Prover<F, T>,
        lut_data: &LookupTableData,
        significance_clamp_indices: &[usize],
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, T>>> {
        let encoding = SoftmaxRaEncoding::exp_hi(self.idx(), lut_data.table_hi.len().log_2());
        let [hi_ra_prover, hi_bool_prover] = shout::ra_onehot_provers(
            &encoding,
            &lut_data.z_hi_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );

        let encoding = SoftmaxRaEncoding::exp_lo(self.idx(), lut_data.table_lo.len().log_2());
        let [lo_ra_prover, lo_bool_prover] = shout::ra_onehot_provers(
            &encoding,
            &lut_data.z_lo_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );

        let significance_clamp_provider: OpLookupProvider<SoftmaxSignificanceClampOperands> =
            OpLookupProvider::new(self.computation_node.clone());
        let encoding = significance_clamp_provider.encoding();
        let [significance_clamp_ra_prover, significance_clamp_bool_prover] =
            shout::ra_onehot_provers(
                &encoding,
                significance_clamp_indices,
                &prover.accumulator,
                &mut prover.transcript,
            );

        vec![
            hi_ra_prover,
            hi_bool_prover,
            lo_ra_prover,
            lo_bool_prover,
            significance_clamp_ra_prover,
            significance_clamp_bool_prover,
        ]
    }
}

/// Run `BatchedSumcheck::prove` with the standard boilerplate.
fn run_batched_prove<F: JoltField, T: Transcript>(
    instances: &mut [Box<dyn SumcheckInstanceProver<F, T>>],
    prover: &mut Prover<F, T>,
) -> SumcheckInstanceProof<F> {
    BatchedSumcheck::prove(
        instances.iter_mut().map(|v| &mut **v as _).collect(),
        &mut prover.accumulator,
        &mut prover.transcript,
    )
    .0
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

/// Pre-computed lookup table data shared between stage3 and stage4.
pub(crate) struct LookupTableData {
    pub(crate) table_hi: Vec<i32>,
    pub(crate) table_lo: Vec<i32>,
    pub(crate) z_hi_indices: Vec<usize>,
    pub(crate) z_lo_indices: Vec<usize>,
}
