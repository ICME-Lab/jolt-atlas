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
use crate::onnx_proof::ops::OpLayout;
use atlas_sumcheck::relation::DynRelationVerifier;

use crate::{
    onnx_proof::{
        deferred_lookups::DeferredBatch,
        op_lookups::OpLookupProvider,
        ops::softmax_last_axis::{
            exp_sum::ExpSumParams,
            exponentiation::{mult::MultParams, ExpDigit, ExpReadRafProvider},
            max::MaxIndicatorParams,
            rc::{SoftmaxRCProvider, SoftmaxRaEncoding},
            recip_mult::{RecipMultParams, RecipMultRelation},
            significance_clamp::SoftmaxSignificanceClampOperands,
        },
        plan::{OpOpen, OpPlan, OpRelation, OpTie, Plan, Step},
        ProofType,
    },
    utils::opening_access::{AccOpeningAccessor, Target},
};
use atlas_lookup_tables::clamp::SoftmaxClampTable;
use atlas_sumcheck::{
    config::{OneHotConfig, OneHotParams},
    opening_proof::{OpeningAccumulator, VerifierOpeningAccumulator},
};

use atlas_trace::{op::SoftmaxLastAxis, trace::ModelExecutionIO, Model};

use crate::node_view::ComputationNode;
use atlas_algebra::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{EvaluateMle, MultilinearPolynomial, PolynomialEvaluation},
        opening_point::OpeningPoint,
    },
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_sumcheck::{identity_range_check::identity_rangecheck_verifier, shout};
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

/// The leading `log_f` (row) coordinates of a bound point; `Err` if the point is shorter.
fn row_coordinates<F>(point: &[F], log_f: usize) -> Result<&[F], ProofVerifyError> {
    point.get(..log_f).ok_or_else(|| {
        ProofVerifyError::Malformed(format!(
            "softmax: a point of {} coordinates has no {log_f} row coordinates",
            point.len()
        ))
    })
}

impl OpLayout for SoftmaxLastAxis {
    fn get_committed_polynomials(
        &self,
        node: &ComputationNode,
    ) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
        let log_scale = node.scale as usize;
        let decomp = generate_exp_lut_decomposed(1i32 << node.scale)?;
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
        Ok(polys)
    }
}

/// The plan (the stage driver as data): the evaluation reduction; the auxiliary vectors; the
/// `exp_sum` opening and its check against them; the recip-mult remainder; stage 1 as one
/// batch; the exp remainder; stage 2 as one batch; `z_hi`/`z_lo`; stage 3 queued for its
/// post-loop batch; the operand link in the clear; stage 4 (the one-hot checks of stage 3's
/// read addresses) queued after stage 3's batch.
impl OpPlan for SoftmaxLastAxis {
    fn plan(
        &self,
        node: &ComputationNode,
        _model: &Model,
        _io: &ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        if node.scale != MODEL_SCALE as u8 {
            return Err(ProofVerifyError::Malformed(format!(
                "model: SoftmaxLastAxis only supports MODEL_SCALE={MODEL_SCALE}, node {} has \
                 scale {}",
                node.idx, node.scale
            )));
        }
        // The sumchecks check the *unmasked* relations, so a model declaring a masked softmax
        // is rejected.
        let last_dim = *node.output_dims.last().ok_or_else(|| {
            ProofVerifyError::Malformed(format!("model: softmax node {} has rank 0", node.idx))
        })?;
        if self.active_len.is_some_and(|n| n != last_dim) {
            return Err(ProofVerifyError::InvalidOpeningProof(format!(
                "masked softmax (active_len {:?} of {last_dim}) is not yet verifiable",
                self.active_len
            )));
        }
        let shape = SoftmaxShape::for_node(node)?;
        let open = |open: fn(SoftmaxShape) -> SoftmaxOpen| {
            Step::Open(OpOpen::Softmax(open(shape.clone())))
        };
        let tie = |tie: fn(SoftmaxShape) -> SoftmaxTie| Step::Tie {
            relation: OpTie::Softmax(tie(shape.clone())),
        };
        let relation =
            |stage: fn(SoftmaxShape) -> SoftmaxRelation| OpRelation::Softmax(stage(shape.clone()));
        Ok(Plan(vec![
            Step::Reduce,
            open(SoftmaxOpen::Aux),
            open(SoftmaxOpen::ExpSum),
            tie(SoftmaxTie::ExpSum),
            open(SoftmaxOpen::RecipMultRemainder),
            Step::Batch {
                proof: ProofType::SoftmaxStage1,
                relations: vec![relation(SoftmaxRelation::Stage1)],
            },
            open(SoftmaxOpen::ExpRemainder),
            Step::Batch {
                proof: ProofType::SoftmaxStage2,
                relations: vec![relation(SoftmaxRelation::Stage2)],
            },
            open(SoftmaxOpen::ZHiLo),
            Step::Defer {
                batch: DeferredBatch::SoftmaxStage3,
                relation: relation(SoftmaxRelation::Stage3),
            },
            tie(SoftmaxTie::OperandLink),
            Step::Defer {
                batch: DeferredBatch::SoftmaxRaChecks,
                relation: relation(SoftmaxRelation::ExpHiOneHot),
            },
            Step::Defer {
                batch: DeferredBatch::SoftmaxRaChecks,
                relation: relation(SoftmaxRelation::ExpLoOneHot),
            },
            Step::Defer {
                batch: DeferredBatch::SoftmaxRaChecks,
                relation: relation(SoftmaxRelation::ClampOneHot),
            },
        ]))
    }
}

/// A softmax node's shape: its rows `F` and row length `N`, and the model scale.
#[derive(Clone)]
pub struct SoftmaxShape {
    /// The softmax node.
    pub node: ComputationNode,
    /// The fixed-point scale `S = 2^scale_bits`.
    pub scale: i32,
    /// `log₂ S`.
    pub scale_bits: u8,
    /// `[F, N]`: the leading dims' product and the softmax axis.
    pub F_N: [usize; 2],
}

/// The auxiliary vectors the prover sends in the clear, decoded from their openings.
pub struct SoftmaxAux {
    /// `exp_sum_q[k]`, the row sums of the quantized exponentials.
    pub exp_sum: Vec<i32>,
    /// `max_k[k]`, the row maxima.
    pub max_k: Vec<i32>,
    /// `argmax_k[k]`, the row argmaxes.
    pub argmax_k: Vec<usize>,
}

impl SoftmaxShape {
    /// The shape of a softmax `node`. `Err` if the node has rank 0.
    pub fn for_node(node: &ComputationNode) -> Result<Self, ProofVerifyError> {
        let (&n, leading_dims) = node.output_dims.split_last().ok_or_else(|| {
            ProofVerifyError::Malformed(format!("model: softmax node {} has rank 0", node.idx))
        })?;
        Ok(Self {
            node: node.clone(),
            scale: 1i32 << node.scale,
            scale_bits: node.scale,
            F_N: [leading_dims.iter().product::<usize>(), n],
        })
    }

    /// The softmax node's index.
    pub fn idx(&self) -> usize {
        self.node.idx
    }

    fn log_f(&self) -> usize {
        self.F_N[0].log_2()
    }

    /// Decode the auxiliary vectors from their openings (appended by [`SoftmaxOpen::Aux`]).
    /// `Err` if one is missing or does not fit in 64 bits.
    pub fn aux<F: JoltField>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Result<SoftmaxAux, ProofVerifyError> {
        let [f, _] = self.F_N;
        let accessor = AccOpeningAccessor::new(accumulator, &self.node);
        let read = |vp_fn: fn(usize, usize) -> VirtualPoly| -> Result<Vec<u64>, ProofVerifyError> {
            (0..f)
                .map(|k| {
                    accessor
                        .get_advice(|idx| vp_fn(idx, k))?
                        .1
                        .to_u64()
                        .ok_or_else(|| {
                            ProofVerifyError::Malformed(format!(
                                "softmax auxiliary scalar {k} does not fit within 64 bits"
                            ))
                        })
                })
                .collect()
        };
        Ok(SoftmaxAux {
            exp_sum: read(VirtualPoly::SoftmaxSumOutput)?
                .into_iter()
                .map(|v| v as i32)
                .collect(),
            max_k: read(VirtualPoly::SoftmaxMaxOutput)?
                .into_iter()
                .map(|v| v as i32)
                .collect(),
            argmax_k: read(VirtualPoly::SoftmaxMaxIndex)?
                .into_iter()
                .map(|v| v as usize)
                .collect(),
        })
    }

    /// Evaluations of the per-row inverse sum `⌊S²/exp_sum_q[k]⌋`.
    /// `Err` if a row sum is missing, oversized or zero.
    pub fn inv_sum_evals<F: JoltField>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Result<Vec<F>, ProofVerifyError> {
        let [f, _] = self.F_N;
        let s_squared = (self.scale as i64) * (self.scale as i64);
        let accessor = AccOpeningAccessor::new(accumulator, &self.node);
        (0..f)
            .map(|k| {
                let exp_sum_q_k = accessor
                    .get_advice(|idx| VirtualPoly::SoftmaxSumOutput(idx, k))?
                    .1;
                let exp_sum_q_int = exp_sum_q_k.to_u64().ok_or_else(|| {
                    ProofVerifyError::Malformed(format!("exp_sum_q[{k}] does not fit in u64"))
                })? as i64;
                if exp_sum_q_int == 0 {
                    return Err(ProofVerifyError::InvalidOpeningProof(format!(
                        "exp_sum_q[{k}] is zero, cannot compute inv_sum"
                    )));
                }
                Ok(F::from_i32((s_squared / exp_sum_q_int) as i32))
            })
            .collect()
    }
}

/// A softmax advice opening.
pub enum SoftmaxOpen {
    /// The auxiliary vectors (`exp_sum_q`, `max_k`, `argmax_k`), each scalar at the empty
    /// point.
    Aux(SoftmaxShape),
    /// `exp_sum_q` at the row coordinates of the reduced output point.
    ExpSum(SoftmaxShape),
    /// The recip-mult remainder at the reduced output point.
    RecipMultRemainder(SoftmaxShape),
    /// The exponentiation remainder at stage 1's `exp_q` point.
    ExpRemainder(SoftmaxShape),
    /// `z_hi` and `z_lo` at stage 2's `exp_hi` point.
    ZHiLo(SoftmaxShape),
}

impl SoftmaxOpen {
    /// Append the openings. `Err` if the accumulator lacks a claim or point it reads.
    pub fn verify<F: JoltField, T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<(), ProofVerifyError> {
        match self {
            Self::Aux(shape) => {
                let [f, _] = shape.F_N;
                let mut provider = AccOpeningAccessor::new(accumulator, &shape.node)
                    .into_provider(transcript, OpeningPoint::default());
                for k in 0..f {
                    provider.append_advice(|idx| VirtualPoly::SoftmaxSumOutput(idx, k))?;
                    provider.append_advice(|idx| VirtualPoly::SoftmaxMaxOutput(idx, k))?;
                    provider.append_advice(|idx| VirtualPoly::SoftmaxMaxIndex(idx, k))?;
                }
                Ok(())
            }
            Self::ExpSum(shape) => {
                let accessor = AccOpeningAccessor::new(accumulator, &shape.node);
                let r0 = accessor.get_reduced_opening()?.0;
                let r_lead = r0.split_at(shape.log_f()).0;
                accessor
                    .into_provider(transcript, r_lead)
                    .append_advice(VirtualPoly::SoftmaxExpSum)
            }
            Self::RecipMultRemainder(shape) => {
                let accessor = AccOpeningAccessor::new(accumulator, &shape.node);
                let r0 = accessor.get_reduced_opening()?.0;
                accessor
                    .into_provider(transcript, r0)
                    .append_advice(VirtualPoly::SoftmaxRecipMultRemainder)
            }
            Self::ExpRemainder(shape) => {
                let accessor = AccOpeningAccessor::new(accumulator, &shape.node);
                let r1 = accessor.get_advice(VirtualPoly::SoftmaxExpQ)?.0;
                accessor
                    .into_provider(transcript, r1)
                    .append_advice(VirtualPoly::SoftmaxExpRemainder)
            }
            Self::ZHiLo(shape) => {
                let accessor = AccOpeningAccessor::new(accumulator, &shape.node);
                let r2 = accessor.get_advice(VirtualPoly::SoftmaxExpHi)?.0;
                let mut provider = accessor.into_provider(transcript, r2);
                provider.append_advice(VirtualPoly::SoftmaxZHi)?;
                provider.append_advice(VirtualPoly::SoftmaxZLo)
            }
        }
    }
}

/// A softmax check in the clear.
pub enum SoftmaxTie {
    /// The `exp_sum` opening equals the auxiliary `exp_sum_q` vector's MLE at its point.
    ExpSum(SoftmaxShape),
    /// The operand link: the prover's `X(r2)` is `max_k(r2_lead) − z(r2)`, where `z(r2)` is
    /// the significance clamp's witness claim (`z = max_k − x`, tied to `z_c = ClampTable[z]`
    /// and `z_hi·base + z_lo` by the stage-3 lookup itself; see `significance_clamp.rs`).
    OperandLink(SoftmaxShape),
}

impl SoftmaxTie {
    /// Check the identity. `Err` if a claim is missing or the identity fails.
    pub fn verify<F: JoltField, T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        _transcript: &mut T,
    ) -> Result<(), ProofVerifyError> {
        match self {
            Self::ExpSum(shape) => {
                let exp_sum = shape.aux(accumulator)?.exp_sum;
                let accessor = AccOpeningAccessor::new(&*accumulator, &shape.node);
                let (r_lead, claimed) = accessor.get_advice(VirtualPoly::SoftmaxExpSum)?;
                let expected = MultilinearPolynomial::from(exp_sum).evaluate(&r_lead.r);
                if expected != claimed {
                    return Err(ProofVerifyError::InvalidOpeningProof(
                        "exp_sum evaluation mismatch".to_string(),
                    ));
                }
                Ok(())
            }
            Self::OperandLink(shape) => {
                let max_k = shape.aux(accumulator)?.max_k;
                let accessor = AccOpeningAccessor::new(&*accumulator, &shape.node);
                let r2 = accessor.get_advice(VirtualPoly::SoftmaxExpHi)?.0;
                let r2_lead = row_coordinates(&r2.r, shape.log_f())?;
                let max_k_eval = max_k.evaluate_mle(r2_lead);
                let z_eval = accessor.get_advice(VirtualPoly::SoftmaxClampWitness)?.1;
                let x_r2 = max_k_eval - z_eval;
                let prover_x_r2 = accessor.get_nodeio(Target::Input(0))?.1;
                if prover_x_r2 != x_r2 {
                    return Err(ProofVerifyError::InvalidOpeningProof(
                        "Operand link failed: prover's X(r2) does not match max_k - z".to_string(),
                    ));
                }
                Ok(())
            }
        }
    }
}

/// A softmax stage's relations.
pub enum SoftmaxRelation {
    /// Stage 1: recip-mult, `exp_sum` and the recip-mult remainder's identity range check.
    Stage1(SoftmaxShape),
    /// Stage 2: the exponentiation multiply, the max indicator, the exp remainder's identity
    /// range check, and the recip-mult remainder's one-hot checks.
    Stage2(SoftmaxShape),
    /// Stage 3: the exp-digit lookups, the significance clamp read-raf, and the exp
    /// remainder's one-hot checks.
    Stage3(SoftmaxShape),
    /// Stage 4: the one-hot checks of the high exp-digit lookup.
    ExpHiOneHot(SoftmaxShape),
    /// Stage 4: the one-hot checks of the low exp-digit lookup.
    ExpLoOneHot(SoftmaxShape),
    /// Stage 4: the one-hot checks of the significance clamp.
    ClampOneHot(SoftmaxShape),
}

impl SoftmaxRelation {
    /// The stage's verifiers, set up in the stage's order.
    /// `Err` if the accumulator lacks a claim a setup reads.
    pub fn verifiers<F: JoltField, T: Transcript>(
        self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<Vec<Box<dyn DynRelationVerifier<F, T>>>, ProofVerifyError> {
        match self {
            Self::Stage1(shape) => {
                let inv_sum = shape.inv_sum_evals(&*accumulator)?;
                let recip_mult_verifier = RecipMultRelation::with_inv_sum(
                    RecipMultParams::new(
                        shape.node.clone(),
                        shape.scale,
                        shape.F_N,
                        &*accumulator,
                        transcript,
                    )?,
                    inv_sum,
                )
                .into_verifier();
                let exp_sum_verifier =
                    ExpSumParams::new(shape.node.clone(), shape.F_N, &*accumulator, transcript)?
                        .into_verifier();
                let rc_provider =
                    SoftmaxRCProvider::remainder(shape.node.clone(), shape.scale_bits);
                let rc_R_verifier = identity_rangecheck_verifier(&rc_provider, accumulator)?;
                Ok(vec![
                    Box::new(recip_mult_verifier),
                    Box::new(exp_sum_verifier),
                    Box::new(rc_R_verifier),
                ])
            }
            Self::Stage2(shape) => {
                let SoftmaxAux {
                    max_k, argmax_k, ..
                } = shape.aux(accumulator)?;
                let accessor = AccOpeningAccessor::new(&*accumulator, &shape.node);
                let r1 = accessor.get_advice(VirtualPoly::SoftmaxExpQ)?.0;
                let r1_k = row_coordinates(&r1.r, shape.log_f())?;
                let max_k_eval = max_k.evaluate_mle(r1_k);
                let max_indicator_verifier = MaxIndicatorParams::new(
                    shape.node.clone(),
                    shape.F_N,
                    argmax_k,
                    max_k_eval,
                    &*accumulator,
                )?
                .into_verifier();

                let exp_mult_verifier =
                    MultParams::new(shape.node.clone(), shape.scale, &*accumulator)?
                        .into_verifier();

                let rc_provider =
                    SoftmaxRCProvider::exp_remainder(shape.node.clone(), shape.scale_bits);
                let exp_r_rc_verifier = identity_rangecheck_verifier(&rc_provider, accumulator)?;

                let encoding = SoftmaxRaEncoding::remainder(shape.idx(), shape.scale_bits);
                let [R_ra_verifier, R_bool_verifier] =
                    shout::ra_onehot_verifiers(&encoding, &*accumulator, transcript)?;

                Ok(vec![
                    Box::new(exp_mult_verifier),
                    Box::new(max_indicator_verifier),
                    Box::new(exp_r_rc_verifier),
                    R_ra_verifier,
                    R_bool_verifier,
                ])
            }
            Self::Stage3(shape) => {
                let lut = VerifierLookupTableData::new(shape.scale)?;
                let provider = ExpReadRafProvider {
                    node: shape.node.clone(),
                    table_size: lut.table_hi.len(),
                    digit: ExpDigit::Hi,
                };
                let hi_verifier =
                    shout::read_raf_verifier(&provider, lut.table_hi, &*accumulator, transcript)?;

                let provider = ExpReadRafProvider {
                    node: shape.node.clone(),
                    table_size: lut.table_lo.len(),
                    digit: ExpDigit::Lo,
                };
                let lo_verifier =
                    shout::read_raf_verifier(&provider, lut.table_lo, &*accumulator, transcript)?;

                let significance_clamp_provider: OpLookupProvider<
                    SoftmaxSignificanceClampOperands,
                > = OpLookupProvider::new(shape.node.clone());
                let significance_clamp_verifier = significance_clamp_provider
                    .read_raf_verify::<F, T, SoftmaxClampTable<XLEN>, XLEN>(
                        accumulator,
                        transcript,
                    )?;

                let encoding = SoftmaxRaEncoding::exp_remainder(shape.idx(), shape.scale_bits);
                let [r_exp_ra_verifier, r_exp_bool_verifier] =
                    shout::ra_onehot_verifiers(&encoding, &*accumulator, transcript)?;

                Ok(vec![
                    hi_verifier,
                    lo_verifier,
                    Box::new(significance_clamp_verifier),
                    r_exp_ra_verifier,
                    r_exp_bool_verifier,
                ])
            }
            Self::ExpHiOneHot(shape) => {
                let lut = VerifierLookupTableData::new(shape.scale)?;
                let encoding = SoftmaxRaEncoding::exp_hi(shape.idx(), lut.table_hi.len().log_2());
                Ok(shout::ra_onehot_verifiers(&encoding, &*accumulator, transcript)?.into())
            }
            Self::ExpLoOneHot(shape) => {
                let lut = VerifierLookupTableData::new(shape.scale)?;
                let encoding = SoftmaxRaEncoding::exp_lo(shape.idx(), lut.table_lo.len().log_2());
                Ok(shout::ra_onehot_verifiers(&encoding, &*accumulator, transcript)?.into())
            }
            Self::ClampOneHot(shape) => {
                let provider: OpLookupProvider<SoftmaxSignificanceClampOperands> =
                    OpLookupProvider::new(shape.node.clone());
                Ok(
                    shout::ra_onehot_verifiers(&provider.encoding(), &*accumulator, transcript)?
                        .into(),
                )
            }
        }
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
    /// The decomposed exp tables at `scale`. `Err` unless `scale` is a power of two the
    /// semantics define the tables for.
    pub fn new(scale: i32) -> Result<Self, ProofVerifyError> {
        let decomp = generate_exp_lut_decomposed(scale)?;
        let mut table_hi = decomp.lut_hi;
        table_hi.resize(table_hi.len().next_power_of_two(), 0);
        let mut table_lo = decomp.lut_lo;
        table_lo.resize(table_lo.len().next_power_of_two(), 0);
        Ok(Self { table_hi, table_lo })
    }
}
