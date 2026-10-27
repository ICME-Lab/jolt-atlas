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
//! `ClampBoundedTable` lookup (`atlas_lookup_tables::clamp::SoftmaxClampTable`, via
//! `significance_clamp::SoftmaxSignificanceClampOperands`) — see `significance_clamp.rs`'s module doc for the relation being
//! proven. This lookup only supports the compile-time `common::consts::MODEL_SCALE` (its bound is
//! a const generic), so `SoftmaxLastAxis` only supports that one scale at runtime.
use crate::onnx_proof::ops::{OpLayout, VerifyOp};

use crate::{
    onnx_proof::{
        deferred_lookups::{DeferredBatch, DeferredOneHot},
        op_lookups::OpLookupProvider,
        ops::softmax_last_axis::{
            exp_sum::ExpSumParams,
            exponentiation::{mult::MultParams, ExpDigit, ExpReadRafProvider},
            max::MaxIndicatorParams,
            rc::{SoftmaxRCProvider, SoftmaxRaEncoding},
            recip_mult::{RecipMultParams, RecipMultRelation},
            significance_clamp::SoftmaxSignificanceClampOperands,
        },
        ProofId, ProofType, Verifier,
    },
    utils::opening_access::{AccOpeningAccessor, Target},
};
use atlas_lookup_tables::clamp::SoftmaxClampTable;
use atlas_sumcheck::{
    config::{OneHotConfig, OneHotParams},
    opening_proof::VerifierOpeningAccumulator,
};
use std::collections::BTreeMap;

use atlas_trace::op::SoftmaxLastAxis;

use crate::node_view::ComputationNode;
use atlas_algebra::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{EvaluateMle, MultilinearPolynomial, PolynomialEvaluation},
        opening_point::OpeningPoint,
    },
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_sumcheck::{
    identity_range_check::identity_rangecheck_verifier,
    shout,
    sumcheck::{BatchedSumcheck, SumcheckInstanceProof},
    sumcheck_verifier::SumcheckInstanceVerifier,
};
use atlas_transcript::Transcript;
use common::{
    consts::{MODEL_SCALE, XLEN},
    CommittedPoly, VirtualPoly,
};
pub use exp_lut::{generate_exp_lut_decomposed, softmax_z, ExpLutDecomposed};

/// Exp-LUT materialization and small softmax helpers.
pub mod exp_lut;
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

impl OpLayout for SoftmaxLastAxis {
    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        let log_scale = node.scale as usize;
        let decomp = generate_exp_lut_decomposed(1i32 << node.scale);
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
            (XLEN, CommittedPoly::SoftmaxClampRaD),
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

impl<F: JoltField, T: Transcript> VerifyOp<F, T> for SoftmaxLastAxis {
    #[tracing::instrument(skip_all, name = "SoftmaxLastAxis::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        debug_assert_eq!(
            node.scale, MODEL_SCALE as u8,
            "SoftmaxLastAxis only supports the currently-compiled MODEL_SCALE={MODEL_SCALE}"
        );
        // Mirror of the prove-side gate: the sumchecks below check the
        // *unmasked* relations, so a model declaring a masked softmax must
        // be rejected here too (deserialized preprocessing bypasses the
        // preprocess()-time assert).
        let last_dim = *node.output_dims.last().expect("softmax rank >= 1");
        if self.active_len.is_some_and(|n| n != last_dim) {
            return Err(ProofVerifyError::InvalidOpeningProof(format!(
                "masked softmax (active_len {:?} of {last_dim}) is not yet verifiable",
                self.active_len
            )));
        }
        let scale_bits = verifier.preprocessing.scale();
        let scale = 1i32 << node.scale;
        let mut sm = SoftmaxLastAxisVerifier::new(
            node,
            scale,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );
        let (stage3, onehots) = sm.verify(
            &mut verifier.accumulator,
            &mut verifier.transcript,
            verifier.proofs,
            scale_bits as i32,
        )?;
        for instance in stage3 {
            verifier.defer(DeferredBatch::SoftmaxStage3, instance);
        }
        verifier.deferred_onehots.extend(onehots);
        Ok(())
    }
}

/// Verifier for softmax-last-axis operations.
///
/// Mirror of [`SoftmaxLastAxisProver`] — each stage method corresponds to a
/// prover stage and verifies the associated sumcheck proof.
pub struct SoftmaxLastAxisVerifier {
    computation_node: ComputationNode,
    scale: i32,
    F_N: [usize; 2],
    exp_sum: Vec<i32>,
    max_k: Vec<i32>,
    argmax_k: Vec<usize>,
}

impl SoftmaxLastAxisVerifier {
    /// The softmax node's index.
    #[inline]
    pub fn idx(&self) -> usize {
        self.computation_node.idx
    }

    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::verify", skip_all)]
    /// Run the inline part of the verification pipeline: stages 1–2
    /// verified here; the stage-3 verifiers and stage-4 one-hot jobs are
    /// returned for the caller to defer (see `deferred_lookups`).
    #[allow(clippy::type_complexity)]
    pub fn verify<F: JoltField, T: Transcript>(
        &mut self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        proofs: &BTreeMap<ProofId, SumcheckInstanceProof<F>>,
        scale_bits: i32,
    ) -> Result<
        (
            Vec<Box<dyn SumcheckInstanceVerifier<F, T>>>,
            Vec<DeferredOneHot>,
        ),
        ProofVerifyError,
    > {
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
            self.build_stage2_verifiers(accumulator, transcript, scale_bits),
            proofs,
            accumulator,
            transcript,
        )?;

        self.cache_z_hi_lo(accumulator, transcript);

        // Stage 3 / 4 are deferred (see the prover); their verifiers are
        // built here and handed back to the caller via `deferred`.
        let stage3 = self.build_stage3_verifiers(accumulator, transcript, &lut);
        // The operand link only needs claims appended during construction.
        self.operand_link(accumulator)?;
        let onehots = vec![
            DeferredOneHot::SoftmaxExpHi(self.idx(), lut.table_hi.len().log_2()),
            DeferredOneHot::SoftmaxExpLo(self.idx(), lut.table_lo.len().log_2()),
            DeferredOneHot::SoftmaxClamp(self.computation_node.clone()),
        ];
        let _ = proofs;
        Ok((stage3, onehots))
    }

    /// Drive a single batched-sumcheck stage given pre-built verifier instances.
    fn run_stage<F: JoltField, T: Transcript>(
        &self,
        proof_type: ProofType,
        instances: Vec<Box<dyn SumcheckInstanceVerifier<F, T>>>,
        proofs: &BTreeMap<ProofId, SumcheckInstanceProof<F>>,
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
// SoftmaxLastAxisVerifier methods
// ---------------------------------------------------------------------------

impl SoftmaxLastAxisVerifier {
    /// Construct the verifier and receive auxiliary vectors from the transcript.
    ///
    /// Mirror of [`SoftmaxLastAxisProver::send_auxiliary_vectors`].
    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::new", skip_all)]
    pub fn new<F: JoltField, T: Transcript>(
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

    /// Evaluations of the per-row inverse sum at the stage point.
    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::inv_sum_evals", skip_all)]
    pub fn inv_sum_evals<F: JoltField>(
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
    pub fn cache_exp_sum<F: JoltField, T: Transcript>(
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
    /// Append the remainder polynomial commitment to the accumulator.
    pub fn cache_R<F: JoltField, T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) {
        let accessor = AccOpeningAccessor::new(accumulator, &self.computation_node);
        let r = accessor.get_reduced_opening().0;
        let mut provider = accessor.into_provider(transcript, r);
        provider.append_advice(VirtualPoly::SoftmaxRecipMultRemainder);
    }

    /// Build stage 1 verifier instances. The caller drives the actual sumcheck.
    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::build_stage1_verifiers", skip_all)]
    pub fn build_stage1_verifiers<F: JoltField, T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        scale_bits: i32,
    ) -> Result<Vec<Box<dyn SumcheckInstanceVerifier<F, T>>>, ProofVerifyError> {
        let inv_sum = self.inv_sum_evals(&*accumulator)?;
        let recip_mult_verifier = RecipMultRelation::with_inv_sum(
            RecipMultParams::new(
                self.computation_node.clone(),
                self.scale,
                self.F_N,
                &*accumulator,
                transcript,
            ),
            inv_sum,
        )
        .into_verifier();
        let exp_sum_verifier = ExpSumParams::new(
            self.computation_node.clone(),
            self.F_N,
            &*accumulator,
            transcript,
        )
        .into_verifier();
        let rc_provider =
            SoftmaxRCProvider::remainder(self.computation_node.clone(), scale_bits as u8);
        let rc_R_verifier = identity_rangecheck_verifier(&rc_provider, accumulator);
        Ok(vec![
            Box::new(recip_mult_verifier),
            Box::new(exp_sum_verifier),
            Box::new(rc_R_verifier),
        ])
    }

    /// Cache the exponentiation stage's opening point.
    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::cache_r_exp", skip_all)]
    pub fn cache_r_exp<F: JoltField, T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) {
        let accessor = AccOpeningAccessor::new(accumulator, &self.computation_node);
        let r = accessor.get_advice(VirtualPoly::SoftmaxExpQ).0;
        let mut provider = accessor.into_provider(transcript, r);
        provider.append_advice(VirtualPoly::SoftmaxExpRemainder);
    }

    /// Build stage 2 verifier instances.
    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::build_stage2_verifiers", skip_all)]
    pub fn build_stage2_verifiers<F: JoltField, T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        scale_bits: i32,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> {
        let [f, _n] = self.F_N;
        let log_f = f.log_2();

        let accessor = AccOpeningAccessor::new(&*accumulator, &self.computation_node);
        let r1 = accessor.get_advice(VirtualPoly::SoftmaxExpQ).0;
        let r1_k = r1.split_at(log_f).0;

        let max_k_eval = self.max_k.evaluate_mle(&r1_k.r);
        let max_indicator_verifier = MaxIndicatorParams::new(
            self.computation_node.clone(),
            self.F_N,
            self.argmax_k.clone(),
            max_k_eval,
            &*accumulator,
        )
        .into_verifier();

        let exp_mult_verifier =
            MultParams::new(self.computation_node.clone(), self.scale, &*accumulator)
                .into_verifier();

        let rc_provider =
            SoftmaxRCProvider::exp_remainder(self.computation_node.clone(), scale_bits as u8);
        let exp_r_rc_verifier = identity_rangecheck_verifier(&rc_provider, accumulator);

        let encoding = SoftmaxRaEncoding::remainder(self.idx(), scale_bits as u8);
        let [R_ra_verifier, R_bool_verifier] =
            shout::ra_onehot_verifiers(&encoding, &*accumulator, transcript);

        vec![
            Box::new(exp_mult_verifier),
            Box::new(max_indicator_verifier),
            Box::new(exp_r_rc_verifier),
            R_ra_verifier,
            R_bool_verifier,
        ]
    }

    /// Verifier counterpart of [`SoftmaxLastAxisProver::cache_z_hi_lo`]: append the
    /// `SoftmaxZHi`/`SoftmaxZLo` opening requests (claims loaded from the proof).
    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::cache_z_hi_lo", skip_all)]
    pub fn cache_z_hi_lo<F: JoltField, T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) {
        let accessor = AccOpeningAccessor::new(accumulator, &self.computation_node);
        let r2 = accessor.get_advice(VirtualPoly::SoftmaxExpHi).0;
        let mut provider = accessor.into_provider(transcript, r2);
        provider.append_advice(VirtualPoly::SoftmaxZHi);
        provider.append_advice(VirtualPoly::SoftmaxZLo);
    }

    /// Build stage 3 verifier instances.
    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::build_stage3_verifiers", skip_all)]
    pub fn build_stage3_verifiers<F: JoltField, T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
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

        let significance_clamp_provider: OpLookupProvider<SoftmaxSignificanceClampOperands> =
            OpLookupProvider::new(self.computation_node.clone());
        let significance_clamp_verifier = significance_clamp_provider
            .read_raf_verify::<F, T, SoftmaxClampTable<XLEN>, XLEN>(accumulator, transcript);

        let scale_bits = self.scale.ilog2() as i32;
        let encoding = SoftmaxRaEncoding::exp_remainder(self.idx(), scale_bits as u8);
        let [r_exp_ra_verifier, r_exp_bool_verifier] =
            shout::ra_onehot_verifiers(&encoding, &*accumulator, transcript);

        vec![
            hi_verifier,
            lo_verifier,
            Box::new(significance_clamp_verifier),
            r_exp_ra_verifier,
            r_exp_bool_verifier,
        ]
    }

    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::operand_link", skip_all)]
    /// Operand link: derive `X(r2)` algebraically and route upstream.
    ///
    /// `X(r2) = max_k(r2_lead) − z(r2)`, where `z(r2)` is the sat-clamp lookup's witness claim
    /// (`z = max_k - x`, already tied to `z_c = ClampTable[z]` and `z_hi*base + z_lo` by the
    /// stage-3 lookup itself — see `significance_clamp.rs`).
    pub fn operand_link<F: JoltField>(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Result<(), ProofVerifyError> {
        let [f, _n] = self.F_N;
        let log_f = f.log_2();
        let accessor = AccOpeningAccessor::new(accumulator, &self.computation_node);

        let r2 = accessor.get_advice(VirtualPoly::SoftmaxExpHi).0;
        let r2_lead = r2.split_at(log_f).0;

        let max_k_eval = self.max_k.evaluate_mle(&r2_lead.r);

        let z_eval = accessor.get_advice(VirtualPoly::SoftmaxClampWitness).1;

        let x_r2 = max_k_eval - z_eval;

        let prover_x_r2 = accessor.get_nodeio(Target::Input(0)).1;
        if prover_x_r2 != x_r2 {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Operand link failed: prover's X(r2) does not match max_k - z".to_string(),
            ));
        }
        Ok(())
    }

    /// Build stage 4 verifier instances.
    #[allow(dead_code)]
    #[tracing::instrument(name = "SoftmaxLastAxisVerifier::build_stage4_verifiers", skip_all)]
    pub fn build_stage4_verifiers<F: JoltField, T: Transcript>(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        lut: &VerifierLookupTableData,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> {
        let encoding = SoftmaxRaEncoding::exp_hi(self.idx(), lut.table_hi.len().log_2());
        let [hi_ra_verifier, hi_bool_verifier] =
            shout::ra_onehot_verifiers(&encoding, accumulator, transcript);

        let encoding = SoftmaxRaEncoding::exp_lo(self.idx(), lut.table_lo.len().log_2());
        let [lo_ra_verifier, lo_bool_verifier] =
            shout::ra_onehot_verifiers(&encoding, accumulator, transcript);

        let significance_clamp_provider: OpLookupProvider<SoftmaxSignificanceClampOperands> =
            OpLookupProvider::new(self.computation_node.clone());
        let encoding = significance_clamp_provider.encoding();
        let [significance_clamp_ra_verifier, significance_clamp_bool_verifier] =
            shout::ra_onehot_verifiers(&encoding, accumulator, transcript);

        vec![
            hi_ra_verifier,
            hi_bool_verifier,
            lo_ra_verifier,
            lo_bool_verifier,
            significance_clamp_ra_verifier,
            significance_clamp_bool_verifier,
        ]
    }
}

/// Pre-computed verifier lookup table data (shared across stage3, operand_link, stage4).
pub struct VerifierLookupTableData {
    /// High half of the decomposed exp table.
    pub table_hi: Vec<i32>,
    /// Low half of the decomposed exp table.
    pub table_lo: Vec<i32>,
}

impl VerifierLookupTableData {
    /// The decomposed exp tables at `scale`.
    pub fn new(scale: i32) -> Self {
        let decomp = generate_exp_lut_decomposed(scale);
        let mut table_hi = decomp.lut_hi;
        table_hi.resize(table_hi.len().next_power_of_two(), 0);
        let mut table_lo = decomp.lut_lo;
        table_lo.resize(table_lo.len().next_power_of_two(), 0);
        Self { table_hi, table_lo }
    }
}
