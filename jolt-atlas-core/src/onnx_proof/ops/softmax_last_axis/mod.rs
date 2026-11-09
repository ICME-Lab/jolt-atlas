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
//!
//! The prover walks the verifier's plan: each [`SoftmaxOpen`], [`SoftmaxRelation`] and
//! [`SoftmaxTie`] step has its prover function here, reading the node's
//! [`SoftmaxWitness`](atlas_trace::SoftmaxWitness) from the trace.
use crate::onnx_proof::op_lookups::{LookupWitness, OpLookupProve};
use crate::onnx_proof::trace_view::TraceView;
use crate::utils::opening_access::IntoProverProvider;
use atlas_sumcheck::relation::prover::DynRelationWitness;
pub use jolt_atlas_verifier::onnx_proof::ops::softmax_last_axis::*;

use crate::{
    onnx_proof::{
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
    },
    utils::{compute_lookup_indices_from_operands, opening_access::AccOpeningAccessor},
};
use atlas_lookup_tables::clamp::SoftmaxClampTable;
use atlas_sumcheck::opening_proof::prover::ProverOpeningAccumulator;
use atlas_trace::{SoftmaxWitness, Tensor, Trace};

use atlas_algebra::{
    field::JoltField,
    poly::{multilinear_polynomial::EvaluateMle, opening_point::OpeningPoint},
    utils::{lookup_bits::LookupBits, math::Math},
};
use atlas_sumcheck::{identity_range_check::prover::identity_rangecheck_prover, shout};
use atlas_transcript::Transcript;
use common::{consts::XLEN, VirtualPoly};

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

/// The node's softmax witness, captured by the executor.
fn witness<'a>(shape: &SoftmaxShape, trace: &'a Trace) -> &'a SoftmaxWitness {
    trace
        .softmax_witness(shape.idx())
        .unwrap_or_else(|| panic!("softmax node {} has no trace witness", shape.idx()))
}

/// The read addresses a softmax stage leaves for its one-hot checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SoftmaxAddresses {
    /// Stage 3's high exp-digit lookup.
    ExpHi,
    /// Stage 3's low exp-digit lookup.
    ExpLo,
    /// Stage 3's significance clamp.
    Clamp,
}

/// Append a [`SoftmaxOpen`]'s openings with the witness's claims.
///
/// The auxiliary scalars are sent as `F::from_u32(v as u32)` rather than `from_i32`: `max_k`
/// can be negative, and a two's-complement `u32` stays in `[0, 2^32)`, which the verifier
/// recovers with `to_u64() as i32`; `from_i32` of a negative value would be `p − |v|`, which
/// does not roundtrip through `to_u64()`.
pub fn prove_open<F: JoltField, T: Transcript>(
    open: &SoftmaxOpen,
    trace: &Trace,
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
) {
    let expect = "the prover holds every claim it reads";
    match open {
        SoftmaxOpen::Aux(shape) => {
            let w = witness(shape, trace);
            let [f, _] = shape.F_N;
            let mut provider = AccOpeningAccessor::new(accumulator, &shape.node)
                .into_provider(transcript, OpeningPoint::default());
            for k in 0..f {
                provider.append_advice(
                    |idx| VirtualPoly::SoftmaxSumOutput(idx, k),
                    F::from_u32(w.exp_sum_q[k] as u32),
                );
                provider.append_advice(
                    |idx| VirtualPoly::SoftmaxMaxOutput(idx, k),
                    F::from_u32(w.max_k[k] as u32),
                );
                provider.append_advice(
                    |idx| VirtualPoly::SoftmaxMaxIndex(idx, k),
                    F::from_u32(w.argmax_k[k] as u32),
                );
            }
        }
        SoftmaxOpen::ExpSum(shape) => {
            let w = witness(shape, trace);
            let accessor = AccOpeningAccessor::new(accumulator, &shape.node);
            let r0 = accessor.get_reduced_opening().expect(expect).0;
            let r_lead = r0.split_at(shape.F_N[0].log_2()).0;
            let eval = w.exp_sum_q.evaluate_mle(&r_lead.r);
            accessor
                .into_provider(transcript, r_lead)
                .append_advice(VirtualPoly::SoftmaxExpSum, eval);
        }
        SoftmaxOpen::RecipMultRemainder(shape) => {
            let w = witness(shape, trace);
            let accessor = AccOpeningAccessor::new(accumulator, &shape.node);
            let r0 = accessor.get_reduced_opening().expect(expect).0;
            let eval = w.r.evaluate_mle(&r0.r);
            accessor
                .into_provider(transcript, r0)
                .append_advice(VirtualPoly::SoftmaxRecipMultRemainder, eval);
        }
        SoftmaxOpen::ExpRemainder(shape) => {
            let w = witness(shape, trace);
            let accessor = AccOpeningAccessor::new(accumulator, &shape.node);
            let r1 = accessor
                .get_advice(VirtualPoly::SoftmaxExpQ)
                .expect(expect)
                .0;
            let eval = w.r_exp.evaluate_mle(&r1.r);
            accessor
                .into_provider(transcript, r1)
                .append_advice(VirtualPoly::SoftmaxExpRemainder, eval);
        }
        SoftmaxOpen::ZHiLo(shape) => {
            let w = witness(shape, trace);
            let accessor = AccOpeningAccessor::new(accumulator, &shape.node);
            let r2 = accessor
                .get_advice(VirtualPoly::SoftmaxExpHi)
                .expect(expect)
                .0;
            let z_hi_eval = w.z_hi.evaluate_mle(&r2.r);
            let z_lo_eval = w.z_lo.evaluate_mle(&r2.r);
            let mut provider = accessor.into_provider(transcript, r2);
            provider.append_advice(VirtualPoly::SoftmaxZHi, z_hi_eval);
            provider.append_advice(VirtualPoly::SoftmaxZLo, z_lo_eval);
        }
    }
}

/// A stage's provers and the read addresses it leaves for its one-hot checks.
pub type StageProvers<F, T> = (
    Vec<Box<dyn DynRelationWitness<F, T>>>,
    Vec<(SoftmaxAddresses, Vec<usize>)>,
);

/// The provers of a [`SoftmaxRelation`], set up in the stage's order; the read addresses a
/// stage leaves for its one-hot checks are returned beside them, and a one-hot stage takes
/// its addresses from `addresses`.
pub fn prove_relation<F: JoltField, T: Transcript>(
    relation: &SoftmaxRelation,
    trace: &Trace,
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
    addresses: &mut impl FnMut(SoftmaxAddresses) -> Option<Vec<usize>>,
) -> StageProvers<F, T> {
    let expect = "the prover holds every claim it reads";
    let taken = |addresses: &mut dyn FnMut(SoftmaxAddresses) -> Option<Vec<usize>>,
                 key: SoftmaxAddresses| {
        addresses(key).unwrap_or_else(|| {
            panic!("softmax: the {key:?} addresses were not left by their lookup")
        })
    };
    match relation {
        SoftmaxRelation::Stage1(shape) => {
            let w = witness(shape, trace);
            let recip_mult_params = RecipMultParams::new(
                shape.node.clone(),
                shape.scale,
                shape.F_N,
                &*accumulator,
                transcript,
            )
            .expect(expect);
            let recip_mult_prover =
                RecipMultProver::initialize(w.exp_q.clone(), w.inv_sum.clone(), recip_mult_params);

            let exp_sum_params =
                ExpSumParams::new(shape.node.clone(), shape.F_N, &*accumulator, transcript)
                    .expect(expect);
            let exp_sum_prover = ExpSumProver::initialize(w.exp_q.clone(), exp_sum_params);

            let provider = SoftmaxRCProvider::remainder(shape.node.clone(), shape.scale_bits);
            let r_lookup_bits = to_lookup_bits(&w.r, shape.scale_bits as usize);
            let rc_R_prover = identity_rangecheck_prover(&provider, r_lookup_bits, accumulator);

            (
                vec![
                    Box::new(recip_mult_prover),
                    Box::new(exp_sum_prover),
                    Box::new(rc_R_prover),
                ],
                Vec::new(),
            )
        }
        SoftmaxRelation::Stage2(shape) => {
            let w = witness(shape, trace);
            let [f, _n] = shape.F_N;
            let log_f = f.log_2();

            let accessor = AccOpeningAccessor::new(&*accumulator, &shape.node);
            let r1 = accessor
                .get_advice(VirtualPoly::SoftmaxExpQ)
                .expect(expect)
                .0;
            let r1_k = r1.split_at(log_f).0;

            let max_k_eval = w.max_k.evaluate_mle(&r1_k.r);
            let max_indicator_params = MaxIndicatorParams::new(
                shape.node.clone(),
                shape.F_N,
                w.argmax_k.clone(),
                max_k_eval,
                &*accumulator,
            )
            .expect(expect);
            let max_indicator_prover =
                MaxIndicatorProver::initialize(w.x.clone(), max_indicator_params);

            let exp_mult_params =
                MultParams::new(shape.node.clone(), shape.scale, &*accumulator).expect(expect);
            let exp_mult_prover =
                MultProver::initialize(w.exp_hi.clone(), w.exp_lo.clone(), exp_mult_params);

            let provider = SoftmaxRCProvider::exp_remainder(shape.node.clone(), shape.scale_bits);
            let r_exp_lookup_bits = to_lookup_bits(&w.r_exp, shape.scale_bits as usize);
            let exp_r_rc_prover =
                identity_rangecheck_prover(&provider, r_exp_lookup_bits, accumulator);

            let encoding = SoftmaxRaEncoding::remainder(shape.idx(), shape.scale_bits);
            let r_indices = to_indices(&w.r);
            let [R_ra_prover, R_bool_prover] =
                shout::prover::ra_onehot_provers(&encoding, &r_indices, &*accumulator, transcript);

            (
                vec![
                    Box::new(exp_mult_prover),
                    Box::new(max_indicator_prover),
                    Box::new(exp_r_rc_prover),
                    R_ra_prover,
                    R_bool_prover,
                ],
                Vec::new(),
            )
        }
        SoftmaxRelation::Stage3(shape) => {
            let w = witness(shape, trace);
            let lut = lookup_tables(shape, w);

            let hi_provider = ExpReadRafProvider {
                node: shape.node.clone(),
                table_size: lut.table_hi.len(),
                digit: ExpDigit::Hi,
            };
            let hi_prover = shout::prover::read_raf_prover(
                &hi_provider,
                &lut.z_hi_indices,
                &lut.table_hi,
                &*accumulator,
                transcript,
            );

            let lo_provider = ExpReadRafProvider {
                node: shape.node.clone(),
                table_size: lut.table_lo.len(),
                digit: ExpDigit::Lo,
            };
            let lo_prover = shout::prover::read_raf_prover(
                &lo_provider,
                &lut.z_lo_indices,
                &lut.table_lo,
                &*accumulator,
                transcript,
            );

            // z = max_k - x, the sat-clamp lookup's witness.
            let z_tensor = significance_clamp_witness(shape, w);
            let significance_clamp_indices: Vec<usize> =
                compute_lookup_indices_from_operands(&[&z_tensor.map(|v| v as i32)], false)
                    .iter()
                    .map(|&x| x.into())
                    .collect();
            let significance_clamp_provider: OpLookupProvider<SoftmaxSignificanceClampOperands> =
                OpLookupProvider::new(shape.node.clone());
            significance_clamp_provider.append_witness_claim_for(
                &z_tensor,
                accumulator,
                transcript,
            );
            let significance_clamp_prover = significance_clamp_provider
                .read_raf_prover_from_bits::<F, T, SoftmaxClampTable<XLEN>, XLEN>(
                    significance_clamp_provider.helper.lookup_bits(&z_tensor),
                    accumulator,
                    transcript,
                );

            let encoding = SoftmaxRaEncoding::exp_remainder(shape.idx(), shape.scale_bits);
            let r_exp_indices = to_indices(&w.r_exp);
            let [exp_r_ra_prover, exp_r_bool_prover] = shout::prover::ra_onehot_provers(
                &encoding,
                &r_exp_indices,
                &*accumulator,
                transcript,
            );

            (
                vec![
                    hi_prover,
                    lo_prover,
                    Box::new(significance_clamp_prover),
                    exp_r_ra_prover,
                    exp_r_bool_prover,
                ],
                vec![
                    (SoftmaxAddresses::ExpHi, lut.z_hi_indices),
                    (SoftmaxAddresses::ExpLo, lut.z_lo_indices),
                    (SoftmaxAddresses::Clamp, significance_clamp_indices),
                ],
            )
        }
        SoftmaxRelation::ExpHiOneHot(shape) => {
            let lut = VerifierLookupTableData::new(shape.scale)
                .expect("the softmax multiplier is a scale the semantics define");
            let encoding = SoftmaxRaEncoding::exp_hi(shape.idx(), lut.table_hi.len().log_2());
            let indices = taken(addresses, SoftmaxAddresses::ExpHi);
            (
                shout::prover::ra_onehot_provers(&encoding, &indices, &*accumulator, transcript)
                    .into(),
                Vec::new(),
            )
        }
        SoftmaxRelation::ExpLoOneHot(shape) => {
            let lut = VerifierLookupTableData::new(shape.scale)
                .expect("the softmax multiplier is a scale the semantics define");
            let encoding = SoftmaxRaEncoding::exp_lo(shape.idx(), lut.table_lo.len().log_2());
            let indices = taken(addresses, SoftmaxAddresses::ExpLo);
            (
                shout::prover::ra_onehot_provers(&encoding, &indices, &*accumulator, transcript)
                    .into(),
                Vec::new(),
            )
        }
        SoftmaxRelation::ClampOneHot(shape) => {
            let provider: OpLookupProvider<SoftmaxSignificanceClampOperands> =
                OpLookupProvider::new(shape.node.clone());
            let indices = taken(addresses, SoftmaxAddresses::Clamp);
            (
                shout::prover::ra_onehot_provers(
                    &provider.encoding(),
                    &indices,
                    &*accumulator,
                    transcript,
                )
                .into(),
                Vec::new(),
            )
        }
    }
}

/// A [`SoftmaxTie`] appends nothing on the prover's side: its openings are stage outputs.
pub fn prove_tie(_tie: &SoftmaxTie) {}

/// `z = max_k − x`, padded to a power of two: the significance clamp's lookup witness.
fn significance_clamp_witness(shape: &SoftmaxShape, w: &SoftmaxWitness) -> Tensor<i64> {
    let z = softmax_z(&w.x, &w.max_k, shape.F_N[1]);
    Tensor::from_slice(&z, &[z.len()])
        .expect("softmax_z tensor construction")
        .padded_pow2()
        .map(|v| v as i64)
}

/// The decomposed exp tables (padded) and the witness's digit addresses.
fn lookup_tables(shape: &SoftmaxShape, w: &SoftmaxWitness) -> LookupTableData {
    let mut lut = generate_exp_lut_decomposed(shape.scale)
        .expect("the softmax multiplier is a scale the semantics define");
    let mut table_hi = std::mem::take(&mut lut.lut_hi);
    pad_to_power_of_two(&mut table_hi);
    let mut table_lo = std::mem::take(&mut lut.lut_lo);
    pad_to_power_of_two(&mut table_lo);
    LookupTableData {
        table_hi,
        table_lo,
        z_hi_indices: to_indices(&w.z_hi),
        z_lo_indices: to_indices(&w.z_lo),
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

/// Pre-computed lookup table data shared between stage3 and stage4.
pub(crate) struct LookupTableData {
    pub(crate) table_hi: Vec<i32>,
    pub(crate) table_lo: Vec<i32>,
    pub(crate) z_hi_indices: Vec<usize>,
    pub(crate) z_lo_indices: Vec<usize>,
}
