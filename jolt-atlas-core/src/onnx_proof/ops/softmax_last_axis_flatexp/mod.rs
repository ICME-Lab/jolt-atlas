//! Benchmark-only sibling of `softmax_last_axis`/`softmax_last_axis_satclamp` using a single
//! flat `exp_z` lookup table (no digit split, no `exp_hi*exp_lo` multiplication relation, no
//! `r_exp` remainder) instead of the two-sub-table decomposition. Compares proving cost of "one
//! big table" against "two small tables + combine".
//!
//! Reuses `exp_sum`/`max`/`recip_mult` directly from the original `softmax_last_axis` module
//! (unchanged — softmax normalization is independent of how `exp` is computed). Drops
//! `exponentiation`/`sat_diff`/`rc`'s `exp_remainder`/`sat_diff` range checks entirely: with no
//! digit split there's no `r_exp` remainder, and the saturating clamp is proven via the same
//! `ClampBoundedTable` lookup approach as `softmax_last_axis_satclamp`.
//!
//! Anchoring: this variant has no digit-split stage, so there's no need for the `r1 -> r2`
//! two-step anchoring `softmax_last_axis_satclamp` needs. Everything (`z`, `z_c`, `exp_q`)
//! anchors directly at `r1` (`SoftmaxExpQ`'s opening point, established by stage 1's
//! `RecipMultProver`) — one fewer intermediate point.
//!
//! Pipeline: stage 1 (recip_mult/exp_sum/R — unchanged) establishes `r1`; `cache_z_c` caches
//! `z_c = min(z, z_bound-1)` fresh at `r1`; stage 2 batches `max_indicator` + the two read-raf
//! lookups (`z -> z_c` via `ClampBoundedTable`, `z_c -> exp_q` via a flat dense Shout lookup);
//! `operand_link` ties `z` back to `max_k - x`; stage 3 batches both lookups' one-hot checks.

use crate::{
    onnx_proof::{
        op_lookups::OpLookupProvider,
        ops::{
            softmax_last_axis::{
                exp_sum::{ExpSumParams, ExpSumProver, ExpSumVerifier},
                max::{MaxIndicatorParams, MaxIndicatorProver, MaxIndicatorVerifier},
                rc::SoftmaxRCProvider,
                recip_mult::{RecipMultParams, RecipMultProver, RecipMultVerifier},
            },
            OperatorProofTrait,
        },
        ProofId, ProofType, Prover, Verifier,
    },
    utils::opening_access::{AccOpeningAccessor, Target},
};
use std::collections::BTreeMap;

use atlas_onnx_tracer::{
    node::ComputationNode,
    ops::{
        softmax::{generate_exp_lut_flat, softmax_last_axis_flat, SoftmaxLastAxisFlatTrace},
        SoftmaxLastAxisFlatExp,
    },
    tensor::Tensor,
    utils::quantize::scale_to_multiplier,
};
use common::{consts::XLEN, CommittedPoly, VirtualPoly};
use joltworks::{
    config::{OneHotConfig, OneHotParams},
    field::JoltField,
    lookup_tables::clamp::SoftmaxSatClampTable,
    poly::{
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{
            OpeningAccumulator, OpeningId, OpeningPoint, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
    },
    subprotocols::{
        shout::{self, RaOneHotEncoding, ReadRafProvider},
        sumcheck::{BatchedSumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};

mod flat_clamp;
use flat_clamp::FlatZClampOperands;

// ---------------------------------------------------------------------------
// Flat exp table read-raf provider + one-hot encoding
// ---------------------------------------------------------------------------

/// Dense Shout read-raf provider for the single flat exp table: `exp_q = table[z_c]`. Both
/// `z_c` (address/`raf`) and `exp_q` (value/`rv`) are already cached at `r1` — `z_c` by
/// `cache_z_c`, `exp_q` by stage 1's `RecipMultProver`.
#[derive(Clone)]
struct FlatExpReadRafProvider {
    node: ComputationNode,
    table_size: usize,
}

impl<F: JoltField> ReadRafProvider<F> for FlatExpReadRafProvider {
    fn log_K(&self) -> usize {
        self.table_size.log_2()
    }

    fn r(&self, accumulator: &dyn OpeningAccumulator<F>) -> OpeningPoint<BIG_ENDIAN, F> {
        AccOpeningAccessor::new(accumulator, &self.node)
            .get_advice(VirtualPoly::SoftmaxExpQ)
            .0
    }

    fn ra_poly(&self) -> (VirtualPoly, SumcheckId) {
        (
            VirtualPoly::SoftmaxFlatExpRa(self.node.idx),
            SumcheckId::NodeExecution(self.node.idx),
        )
    }

    fn raf_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        AccOpeningAccessor::new(accumulator, &self.node)
            .get_advice(VirtualPoly::SoftmaxFlatZC)
            .1
    }

    fn rv_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        AccOpeningAccessor::new(accumulator, &self.node)
            .get_advice(VirtualPoly::SoftmaxExpQ)
            .1
    }
}

/// One-hot encoding for the flat exp table's read-address polynomial.
struct FlatExpRaEncoding {
    node_idx: usize,
    log_k: usize,
}

impl RaOneHotEncoding for FlatExpRaEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPoly {
        CommittedPoly::SoftmaxFlatExpRaD(self.node_idx, d)
    }

    fn r_cycle_source(&self) -> OpeningId {
        OpeningId::new(
            VirtualPoly::SoftmaxExpQ(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn ra_source(&self) -> OpeningId {
        OpeningId::new(
            VirtualPoly::SoftmaxFlatExpRa(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn log_k(&self) -> usize {
        self.log_k
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), self.log_k)
    }
}

/// Build the (padded) flat exp table for a given operator scale.
fn flat_exp_table(scale: i32) -> Vec<i32> {
    let mut table = generate_exp_lut_flat(scale);
    table.resize(table.len().next_power_of_two(), 0);
    table
}

// ---------------------------------------------------------------------------
// OperatorProofTrait
// ---------------------------------------------------------------------------

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for SoftmaxLastAxisFlatExp {
    #[tracing::instrument(skip_all, name = "SoftmaxLastAxisFlatExp::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let scale = scale_to_multiplier(self.scale) as i32;
        let softmax_input = prover.trace.operand_tensors(node)[0];
        let trace = softmax_last_axis_flat(softmax_input, scale).1;
        SoftmaxLastAxisFlatExpProver::new(node, trace, scale).prove(prover)
    }

    #[tracing::instrument(skip_all, name = "SoftmaxLastAxisFlatExp::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let scale_bits = verifier.preprocessing.scale();
        let scale = scale_to_multiplier(self.scale) as i32;
        let mut sm = SoftmaxLastAxisFlatExpVerifier::new(
            node,
            scale,
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
        let log_scale = self.scale as usize;
        let table = flat_exp_table(scale_to_multiplier(self.scale) as i32);
        let log_table = table.len().log_2();
        let idx = node.idx;

        let mut polys = vec![];
        for (log_k, ctor) in [
            (
                log_scale,
                CommittedPoly::SoftmaxRemainderRaD as fn(usize, usize) -> _,
            ),
            (XLEN, CommittedPoly::SoftmaxFlatClampRaD),
            (log_table, CommittedPoly::SoftmaxFlatExpRaD),
        ] {
            let d =
                OneHotParams::from_config_and_log_K(&OneHotConfig::default(), log_k).instruction_d;
            polys.extend((0..d).map(|i| ctor(idx, i)));
        }
        polys
    }
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

pub(crate) struct SoftmaxLastAxisFlatExpProver {
    computation_node: ComputationNode,
    scale: i32,
    F_N: [usize; 2],
    trace: SoftmaxLastAxisFlatTrace,
}

impl SoftmaxLastAxisFlatExpProver {
    #[inline]
    fn idx(&self) -> usize {
        self.computation_node.idx
    }

    pub(crate) fn new(node: &ComputationNode, trace: SoftmaxLastAxisFlatTrace, scale: i32) -> Self {
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

    #[tracing::instrument(name = "SoftmaxLastAxisFlatExpProver::prove", skip_all)]
    fn prove<F: JoltField, T: Transcript>(
        self,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let scale_bits = prover.preprocessing.scale();
        debug_assert_eq!(
            self.scale.ilog2() as i32,
            scale_bits,
            "softmax operator scale (log2 {}) disagrees with preprocessing scale {scale_bits}",
            self.scale.ilog2(),
        );

        let r_lookup_bits = crate::onnx_proof::ops::softmax_last_axis::to_lookup_bits(
            &self.trace.R,
            scale_bits as usize,
        );
        let r_indices = crate::onnx_proof::ops::softmax_last_axis::to_indices(&self.trace.R);

        let table = self.trace.flat_exp.table.clone();
        let z_bound_minus_1 = self.trace.flat_exp.z_bound_minus_1;
        let z_c_indices: Vec<usize> = self
            .trace
            .z
            .iter()
            .map(|&zi| zi.min(z_bound_minus_1) as usize)
            .collect();

        self.send_auxiliary_vectors(prover);
        self.cache_exp_sum(prover);
        self.cache_R(prover);
        let stage_1_proof = self.stage1(prover, r_lookup_bits);

        self.cache_z_c(prover, z_bound_minus_1);

        let (stage_2_proof, z_clamp_indices) =
            self.stage2(prover, &table, &z_c_indices, &r_indices);

        let stage_3_proof = self.stage3(prover, table.len(), &z_clamp_indices, &z_c_indices);

        vec![
            (ProofId(self.idx(), ProofType::SoftmaxStage1), stage_1_proof),
            (ProofId(self.idx(), ProofType::SoftmaxStage2), stage_2_proof),
            (ProofId(self.idx(), ProofType::SoftmaxStage3), stage_3_proof),
        ]
    }

    /// Send auxiliary vectors (max_k, exp_sum_q, argmax_k) to the transcript. Identical to
    /// `softmax_last_axis::SoftmaxLastAxisProver::send_auxiliary_vectors`.
    fn send_auxiliary_vectors<F: JoltField, T: Transcript>(&self, prover: &mut Prover<F, T>) {
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

    fn cache_exp_sum<F: JoltField, T: Transcript>(&self, prover: &mut Prover<F, T>) {
        let accessor = AccOpeningAccessor::new(&mut prover.accumulator, &self.computation_node);
        let r0 = accessor.get_reduced_opening().0;
        let log_f = self.F_N[0].log_2();
        let r_lead = r0.split_at(log_f).0;
        let eval = MultilinearPolynomial::from(self.trace.exp_sum_q.clone()).evaluate(&r_lead.r);
        let mut provider = accessor.into_provider(&mut prover.transcript, r_lead);
        provider.append_advice(VirtualPoly::SoftmaxExpSum, eval);
    }

    fn cache_R<F: JoltField, T: Transcript>(&self, prover: &mut Prover<F, T>) {
        let accessor = AccOpeningAccessor::new(&mut prover.accumulator, &self.computation_node);
        let r0 = accessor.get_reduced_opening().0;
        let eval = MultilinearPolynomial::from(self.trace.R.clone()).evaluate(&r0.r);
        let mut provider = accessor.into_provider(&mut prover.transcript, r0);
        provider.append_advice(VirtualPoly::SoftmaxRecipMultRemainder, eval);
    }

    /// Stage 1 batches `recip_mult`/`exp_sum`/`rc_R` (unchanged from the original) **plus**
    /// `max_indicator` — moved here (vs. the original's stage 2) so that `x`'s claim and
    /// `exp_q`'s claim land at the *same* shared point (both stage 1 batch members). Unlike the
    /// original (which reads `r1` — `SoftmaxExpQ`'s point, not yet established — for
    /// `max_indicator`'s eq-binding), this variant's `max_indicator` starts from `r0`'s leading
    /// part instead (available immediately, exactly like `exp_sum` already does) — there is no
    /// analogous `exp_mult` here to "transport" a claim from one point to another, so `x` and
    /// `exp_q` must be established by the *same* batch from the start.
    #[tracing::instrument(name = "SoftmaxLastAxisFlatExpProver::stage1", skip_all)]
    fn stage1<F: JoltField, T: Transcript>(
        &self,
        prover: &mut Prover<F, T>,
        r_lookup_bits: Vec<joltworks::utils::lookup_bits::LookupBits>,
    ) -> SumcheckInstanceProof<F, T> {
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
            self.trace.exp_q.clone(),
            self.trace.inv_sum.clone(),
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
        let rc_R_prover = joltworks::subprotocols::identity_range_check::identity_rangecheck_prover(
            &provider,
            r_lookup_bits,
            &mut prover.accumulator,
        );

        let [f, _n] = self.F_N;
        let log_f = f.log_2();
        let accessor = AccOpeningAccessor::new(&prover.accumulator, &self.computation_node);
        let r0 = accessor.get_reduced_opening().0;
        let r0_k = r0.split_at(log_f).0.r;
        let max_k_eval = MultilinearPolynomial::from(self.trace.max_k.clone()).evaluate(&r0_k);
        let mut e: Vec<u32> = vec![0; self.F_N.iter().product()];
        for (k, &am) in self.trace.argmax_k.iter().enumerate() {
            e[k * self.F_N[1] + am] = 1;
        }
        let max_indicator_params = MaxIndicatorParams {
            node: self.computation_node.clone(),
            r1_k: r0_k,
            input_claim: max_k_eval,
            F_N: self.F_N,
            e,
            argmax_k: self.trace.argmax_k.clone(),
        };
        let max_indicator_prover =
            MaxIndicatorProver::initialize(self.trace.x.clone(), max_indicator_params);

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> = vec![
            Box::new(recip_mult_prover),
            Box::new(exp_sum_prover),
            Box::new(rc_R_prover),
            Box::new(max_indicator_prover),
        ];
        run_batched_prove(&mut instances, prover)
    }

    /// Cache `z_c = min(z, z_bound-1)` fresh at `r1` (`SoftmaxExpQ`'s opening point).
    #[tracing::instrument(name = "SoftmaxLastAxisFlatExpProver::cache_z_c", skip_all)]
    fn cache_z_c<F: JoltField, T: Transcript>(
        &self,
        prover: &mut Prover<F, T>,
        z_bound_minus_1: i32,
    ) {
        let accessor = AccOpeningAccessor::new(&mut prover.accumulator, &self.computation_node);
        let r1 = accessor.get_advice(VirtualPoly::SoftmaxExpQ).0;
        let z_c: Vec<i32> = self
            .trace
            .z
            .iter()
            .map(|&zi| zi.min(z_bound_minus_1))
            .collect();
        let eval = MultilinearPolynomial::from(z_c).evaluate(&r1.r);
        let mut provider = accessor.into_provider(&mut prover.transcript, r1);
        provider.append_advice(VirtualPoly::SoftmaxFlatZC, eval);
    }

    /// Stage 2: the two read-raf lookups (`z -> z_c` via `ClampBoundedTable`, `z_c -> exp_q` via
    /// the flat dense Shout lookup) plus `R`'s one-hot triple. `max_indicator` is *not* here —
    /// see `stage1`'s doc comment for why.
    #[tracing::instrument(name = "SoftmaxLastAxisFlatExpProver::stage2", skip_all)]
    fn stage2<F: JoltField, T: Transcript>(
        &self,
        prover: &mut Prover<F, T>,
        table: &[i32],
        z_c_indices: &[usize],
        r_indices: &[usize],
    ) -> (SumcheckInstanceProof<F, T>, Vec<usize>) {
        let z_tensor = Tensor::new(Some(&self.trace.z), &[self.trace.z.len()])
            .expect("z tensor construction")
            .padded_next_power_of_two()
            .map(|v| v as i64);
        let z_clamp_provider = OpLookupProvider::with_helper(
            self.computation_node.clone(),
            FlatZClampOperands::new(z_tensor),
        );
        let (z_clamp_prover, z_clamp_indices) = z_clamp_provider
            .read_raf_prove::<F, T, SoftmaxSatClampTable<XLEN>, XLEN>(
                &prover.trace,
                &mut prover.accumulator,
                &mut prover.transcript,
            );

        let exp_provider = FlatExpReadRafProvider {
            node: self.computation_node.clone(),
            table_size: table.len(),
        };
        let exp_prover = shout::read_raf_prover(
            &exp_provider,
            z_c_indices,
            table,
            &prover.accumulator,
            &mut prover.transcript,
        );

        // R's one-hot triple stays batched here too (unchanged position relative to the
        // original, which batched it alongside the exp-related checks).
        let encoding = crate::onnx_proof::ops::softmax_last_axis::rc::SoftmaxRaEncoding::remainder(
            self.idx(),
            prover.preprocessing.scale(),
        );
        let [r_ra_prover, r_hw_prover, r_bool_prover] = shout::ra_onehot_provers(
            &encoding,
            r_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> = vec![
            Box::new(z_clamp_prover),
            exp_prover,
            r_ra_prover,
            r_hw_prover,
            r_bool_prover,
        ];
        (run_batched_prove(&mut instances, prover), z_clamp_indices)
    }

    #[tracing::instrument(name = "SoftmaxLastAxisFlatExpProver::stage3", skip_all)]
    fn stage3<F: JoltField, T: Transcript>(
        &self,
        prover: &mut Prover<F, T>,
        table_size: usize,
        z_clamp_indices: &[usize],
        z_c_indices: &[usize],
    ) -> SumcheckInstanceProof<F, T> {
        let z_clamp_provider: OpLookupProvider<FlatZClampOperands> =
            OpLookupProvider::new(self.computation_node.clone());
        let encoding = z_clamp_provider.encoding();
        let [zc_ra, zc_hw, zc_bool] = shout::ra_onehot_provers(
            &encoding,
            z_clamp_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );

        let encoding = FlatExpRaEncoding {
            node_idx: self.idx(),
            log_k: table_size.log_2(),
        };
        let [exp_ra, exp_hw, exp_bool] = shout::ra_onehot_provers(
            &encoding,
            z_c_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> =
            vec![zc_ra, zc_hw, zc_bool, exp_ra, exp_hw, exp_bool];
        run_batched_prove(&mut instances, prover)
    }
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

pub(crate) struct SoftmaxLastAxisFlatExpVerifier {
    computation_node: ComputationNode,
    scale: i32,
    F_N: [usize; 2],
    exp_sum: Vec<i32>,
    max_k: Vec<i32>,
    argmax_k: Vec<usize>,
}

impl SoftmaxLastAxisFlatExpVerifier {
    #[inline]
    fn idx(&self) -> usize {
        self.computation_node.idx
    }

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

    fn inv_sum_evals<F: JoltField>(
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

    #[tracing::instrument(name = "SoftmaxLastAxisFlatExpVerifier::verify", skip_all)]
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

        let table = flat_exp_table(self.scale);
        self.cache_z_c(accumulator, transcript);

        self.run_stage(
            ProofType::SoftmaxStage2,
            self.build_stage2_verifiers(accumulator, transcript, scale_bits, table.clone()),
            proofs,
            accumulator,
            transcript,
        )?;

        self.operand_link(accumulator)?;

        self.run_stage(
            ProofType::SoftmaxStage3,
            self.build_stage3_verifiers(accumulator, transcript, table.len()),
            proofs,
            accumulator,
            transcript,
        )?;
        Ok(())
    }

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

    fn cache_exp_sum<F: JoltField, T: Transcript>(
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

    fn cache_R<F: JoltField, T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) {
        let accessor = AccOpeningAccessor::new(accumulator, &self.computation_node);
        let r = accessor.get_reduced_opening().0;
        let mut provider = accessor.into_provider(transcript, r);
        provider.append_advice(VirtualPoly::SoftmaxRecipMultRemainder);
    }

    fn build_stage1_verifiers<F: JoltField, T: Transcript>(
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
        let rc_provider = SoftmaxRCProvider::remainder(self.computation_node.clone(), scale_bits);
        let rc_R_verifier =
            joltworks::subprotocols::identity_range_check::identity_rangecheck_verifier(
                &rc_provider,
                accumulator,
            );

        let [f, _n] = self.F_N;
        let log_f = f.log_2();
        let accessor = AccOpeningAccessor::new(&*accumulator, &self.computation_node);
        let r0 = accessor.get_reduced_opening().0;
        let r0_k = r0.split_at(log_f).0.r;
        let max_k_eval = MultilinearPolynomial::from(self.max_k.clone()).evaluate(&r0_k);
        let mut e: Vec<u32> = vec![0; self.F_N.iter().product()];
        for (k, &am) in self.argmax_k.iter().enumerate() {
            e[k * self.F_N[1] + am] = 1;
        }
        let max_indicator_params = MaxIndicatorParams {
            node: self.computation_node.clone(),
            r1_k: r0_k,
            input_claim: max_k_eval,
            F_N: self.F_N,
            e,
            argmax_k: self.argmax_k.clone(),
        };
        let max_indicator_verifier = MaxIndicatorVerifier::with_params(max_indicator_params);

        Ok(vec![
            Box::new(recip_mult_verifier),
            Box::new(exp_sum_verifier),
            Box::new(rc_R_verifier),
            Box::new(max_indicator_verifier),
        ])
    }

    fn cache_z_c<F: JoltField, T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) {
        let accessor = AccOpeningAccessor::new(accumulator, &self.computation_node);
        let r1 = accessor.get_advice(VirtualPoly::SoftmaxExpQ).0;
        let mut provider = accessor.into_provider(transcript, r1);
        provider.append_advice(VirtualPoly::SoftmaxFlatZC);
    }

    fn build_stage2_verifiers<F: JoltField, T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        scale_bits: i32,
        table: Vec<i32>,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> {
        let z_clamp_provider: OpLookupProvider<FlatZClampOperands> =
            OpLookupProvider::new(self.computation_node.clone());
        let z_clamp_verifier = z_clamp_provider
            .read_raf_verify::<F, T, SoftmaxSatClampTable<XLEN>, XLEN>(accumulator, transcript);

        let exp_provider = FlatExpReadRafProvider {
            node: self.computation_node.clone(),
            table_size: table.len(),
        };
        let exp_verifier =
            shout::read_raf_verifier(&exp_provider, table, &*accumulator, transcript);

        let encoding = crate::onnx_proof::ops::softmax_last_axis::rc::SoftmaxRaEncoding::remainder(
            self.idx(),
            scale_bits,
        );
        let [r_ra_verifier, r_hw_verifier, r_bool_verifier] =
            shout::ra_onehot_verifiers(&encoding, &*accumulator, transcript);

        vec![
            Box::new(z_clamp_verifier),
            exp_verifier,
            r_ra_verifier,
            r_hw_verifier,
            r_bool_verifier,
        ]
    }

    /// `X(r1) = max_k(r1_lead) - z(r1)`.
    fn operand_link<F: JoltField>(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Result<(), ProofVerifyError> {
        let [f, _n] = self.F_N;
        let log_f = f.log_2();
        let accessor = AccOpeningAccessor::new(accumulator, &self.computation_node);

        let r1 = accessor.get_advice(VirtualPoly::SoftmaxExpQ).0;
        let r1_lead = r1.split_at(log_f).0;

        let max_k_eval = MultilinearPolynomial::from(self.max_k.clone()).evaluate(&r1_lead.r);
        let z_eval = accessor.get_advice(VirtualPoly::SoftmaxFlatZWitness).1;

        let x_r1 = max_k_eval - z_eval;
        let prover_x_r1 = accessor.get_nodeio(Target::Input(0)).1;
        if prover_x_r1 != x_r1 {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Operand link failed: prover's X(r1) does not match max_k - z".to_string(),
            ));
        }
        Ok(())
    }

    fn build_stage3_verifiers<F: JoltField, T: Transcript>(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        table_size: usize,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> {
        let z_clamp_provider: OpLookupProvider<FlatZClampOperands> =
            OpLookupProvider::new(self.computation_node.clone());
        let encoding = z_clamp_provider.encoding();
        let [zc_ra, zc_hw, zc_bool] =
            shout::ra_onehot_verifiers(&encoding, accumulator, transcript);

        let encoding = FlatExpRaEncoding {
            node_idx: self.idx(),
            log_k: table_size.log_2(),
        };
        let [exp_ra, exp_hw, exp_bool] =
            shout::ra_onehot_verifiers(&encoding, accumulator, transcript);

        vec![zc_ra, zc_hw, zc_bool, exp_ra, exp_hw, exp_bool]
    }
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

#[cfg(test)]
mod tests {
    use atlas_onnx_tracer::{
        model::{test::ModelBuilder, Model},
        tensor::Tensor,
    };

    use crate::onnx_proof::ops::test::unit_test_op;

    fn softmax_last_axis_flatexp_model(input_shape: &[usize], scale: u32) -> Model {
        let mut b = ModelBuilder::with_scale(scale);
        let i = b.input(input_shape.to_vec());
        let res = b.softmax_last_axis_flatexp(i);
        b.mark_output(res);
        b.build()
    }

    fn run_softmax_scale_test(scale: u32) {
        let input_shape = vec![2, 8];
        #[rustfmt::skip]
        let data: Vec<i32> = vec![
            10, 20, 30, 40, 50, 60, 5, 15,
            25, 35, 45, 55, 8, 18, 28, 38,
        ];
        let input = Tensor::new(Some(&data), &input_shape).unwrap();
        let model = softmax_last_axis_flatexp_model(&input_shape, scale);
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_softmax_flatexp_model_scale() {
        run_softmax_scale_test(common::consts::MODEL_SCALE as u32);
    }

    /// Small random-data regression test: catches divergence between how a variant computes
    /// `exp_q`/`R` and any *other* code path that (re-)derives it differently (e.g. `witness.rs`
    /// silently falling back to the digit-decomposed trace for a new operator variant). Repeated
    /// or degenerate-per-row data can mask this kind of bug, so this uses genuinely random,
    /// per-row-varying values even at a small shape.
    #[test]
    fn test_softmax_flatexp_random_small() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let scale = common::consts::MODEL_SCALE as i32;
        let input_shape = vec![2, 8];
        let mut rng = StdRng::seed_from_u64(0x5a7c);
        let data: Vec<i32> = (0..16)
            .map(|_| rng.gen_range(-(1 << (scale + 2))..(1 << (scale + 2))))
            .collect();
        let input = Tensor::new(Some(&data), &input_shape).unwrap();
        let model = softmax_last_axis_flatexp_model(&input_shape, scale as u32);
        unit_test_op(model, &[input]);
    }

    /// GPT-2-sized causal-attention fixture (4 heads, 8x8 each), at whatever `MODEL_SCALE` is
    /// currently compiled to — exercises a realistic shape/magnitude, not just the tiny fixture.
    #[test]
    fn test_softmax_flatexp_gpt2_sized() {
        use atlas_onnx_tracer::utils::quantize::mask_sentinel_magnitude;
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let scale = common::consts::MODEL_SCALE as i32;
        let mask_c = mask_sentinel_magnitude(scale) as i32;
        let mask_value = -(mask_c << scale);

        let input_shape = vec![4, 8, 8];
        let mut rng = StdRng::seed_from_u64(0x5a7c);
        let mut data = vec![0i32; 4 * 8 * 8];
        for head in 0..4 {
            for row in 0..8 {
                for col in 0..8 {
                    let idx = head * 64 + row * 8 + col;
                    data[idx] = if col > row {
                        mask_value
                    } else {
                        rng.gen_range(-(1 << (scale + 2))..(1 << (scale + 2)))
                    };
                }
            }
        }
        let input = Tensor::new(Some(&data), &input_shape).unwrap();
        let model = softmax_last_axis_flatexp_model(&input_shape, scale as u32);
        unit_test_op(model, &[input]);
    }
}
