//! # Neural Network Op Proofs Module
//!
//! The prover side of every supported ONNX operation. The verifier crate defines each op's
//! relations, its [`OpLayout`] (committed polynomials) and its [`OpPlan`]; this module walks
//! the plan with the relations' witnesses ([`prove_plan`]), so an op's transcript order is its
//! plan's step order on both sides.
//!
//! ## Op Categories
//!
//! **Arithmetic Operations**: [`add`], [`sub`], [`mul`], [`div`], [`square`], [`cube`], [`scalar_const_div`]
//!
//! **Activation Functions**: [`relu`], [`tanh`], [`softmax_last_axis`]
//!
//! **Tensor Operations**: [`einsum`], [`sum`], [`reshape`], [`broadcast`], [`moveaxis`], [`gather`]
//!
//! **Special Operations**: [`rsqrt`] (reciprocal square root), [`clamp`], [`iff`] (conditional select)
//!
//! **Logic Operations**: [`and`], [`is_nan`]
//!
//! **Infrastructure**: [`input`], [`constant`], [`identity`]
//!
//! ## Architecture
//!
//! Every operator's protocol is a relation (`onnx_proof::relation`): its claim algebra, from
//! which the verifier is derived. A generic relation is proven by
//! [`RelationProver`](crate::onnx_proof::relation::RelationProver) over its
//! [`RelationWitnesses`]; a specialized witness (the einsum dot engine, the mean-of-squares
//! reduction, the lookup provers) attaches at its [`OpRelation`] variant, or at its
//! [`OpRelKind`] for an op relation, in [`relation_witnesses`].

use crate::onnx_proof::ops::eval_reduction::NodeEvalReductionProve;
pub use jolt_atlas_verifier::onnx_proof::ops::*;

/// The clamped activations Erf, Sigmoid and Tanh.
pub mod activation_clamped;
/// Element-wise addition operation.
pub mod add;
/// Broadcast tensors to target shapes.
pub mod broadcast;
/// Clamp values to a specified range.
pub mod clamp;
/// Concatenate tensors along an axis.
pub mod concat;
/// Constant tensor nodes.
pub mod constant;
/// Element-wise division operation.
pub mod div;
/// Einstein summation for tensor contractions.
pub mod einsum;
/// Op-scoped evaluation reduction helpers.
pub mod eval_reduction;
/// Gather elements from input tensor using indices.
pub mod gather;
/// Conditional selection (if-then-else) operation.
pub mod iff;
/// Check for NaN values.
pub mod is_nan;
/// Fused mean-of-squares reduction operation.
pub mod mean_of_squares;
/// Reorder tensor axes.
pub mod moveaxis;
/// Element-wise multiplication operation.
pub mod mul;
/// Element-wise negation operation.
pub mod neg;
/// Reshape tensor dimensions.
pub mod reshape;
/// Reciprocal square root operation.
pub mod rsqrt;
/// Division by a scalar constant.
pub mod scalar_const_div;
/// Slice tensor along one axis.
pub mod slice;
/// Softmax operation along the last axis (axis = -1).
pub mod softmax_last_axis;
/// Sum reduction along axes.
pub mod sum;
/// Sin and Cos.
pub mod trig;

use crate::node_view::ComputationNode;

use atlas_algebra::{
    field::JoltField, poly::multilinear::EvaluateMle, utils::thread::drop_in_background_thread,
};
use atlas_sumcheck::evaluation_reduction::EvalReductionProof;
use atlas_transcript::Transcript;
use eval_reduction::NodeEvalReduction;

use crate::onnx_proof::{
    deferred_lookups::{DeferredBatch, DeferredSetup, DEFERRED_PROOF_IDX},
    fused_rebase::rebase_remainder,
    lookup,
    ops::{
        einsum::dot::EinsumDotProver, gather::GatherProver,
        mean_of_squares::MeanOfSquaresReductionProver,
    },
    plan::{OpOpen, OpPlan, OpRelation, Plan, Step},
    range_check::{self, ValueParams, ValueProver},
    relation::{
        op_rel_params, prove_tie, OpRel, OpRelKind, RelationParams, RelationProver, RelationSetup,
    },
    trace_view::TraceView,
    ProofId,
};
use crate::utils::opening_access::{AccOpeningAccessor, IntoProverProvider};
use atlas_algebra::poly::multilinear::MultilinearPolynomial;
use atlas_sumcheck::digit_range::DigitRangeParams;
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_sumcheck::relation::prover::{
    prove_relation, prove_relation_batch, prove_relation_batch_parallel, DynRelationWitness,
};
use atlas_sumcheck::shout::{self, RaOneHotParams};
use atlas_trace::Trace;
use common::{CommittedPoly, VirtualPoly};
use rayon::prelude::*;

use crate::onnx_proof::Prover;

/// Prover dispatcher for operator proofs: the node's operator's plan, walked with the
/// relations' witnesses.
pub struct OperatorProver;

impl OperatorProver {
    /// Prove `node`: build its operator's plan from the model and the public IO, then walk it.
    /// Returns the node's evaluation-reduction proof and its execution proofs.
    pub fn prove<F, T>(
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> (Option<EvalReductionProof<F>>, ExecutionProofs<F>)
    where
        F: JoltField,
        T: Transcript,
    {
        let plan = jolt_atlas_verifier::dispatch_operator!(node, |inner| OpPlan::plan(
            inner,
            node,
            prover.preprocessing.model(),
            &prover.io
        ))
        .expect("the prover's model has a plan for every node");
        prove_plan(node, prover, plan)
    }
}

/// The key a read leaves its addresses under for its one-hot checks: the opening of its
/// read-address polynomial ([`RaEncoding::ra_source`](lookup::RaEncoding::ra_source)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Addresses(pub OpeningId);

/// A queued deferred relation on the prover's side: its witnesses built in the node's turn,
/// or the relation waiting for its batch.
pub enum Deferred<F: JoltField, T: Transcript> {
    /// Built in the node's turn.
    Ready(Vec<Box<dyn DynRelationWitness<F, T>>>),
    /// Built when its batch runs.
    Pending(Box<OpRelation<F>>),
}

/// Walk `node`'s plan on the prover side: every step the verifier's interpreter
/// ([`Verifier::run_plan`](crate::onnx_proof::Verifier::run_plan)) runs, in the same order, with
/// the relations' witnesses.
#[tracing::instrument(skip_all, name = "prove_plan", fields(node = node.idx, op = %node.operator))]
pub fn prove_plan<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
    plan: Plan<F>,
) -> (Option<EvalReductionProof<F>>, ExecutionProofs<F>) {
    prove_steps(prover, node.idx, Some(node), plan.0)
}

/// Run `steps` whose sumcheck proofs are stored under `proof_idx`; `node` is the node of a
/// node plan (`None` for the post-loop plan, which has no evaluation reduction).
pub fn prove_steps<F: JoltField, T: Transcript>(
    prover: &mut Prover<F, T>,
    proof_idx: usize,
    node: Option<&ComputationNode>,
    steps: Vec<Step<F>>,
) -> (Option<EvalReductionProof<F>>, ExecutionProofs<F>) {
    let mut reduction = None;
    let mut proofs = Vec::new();
    for step in steps {
        match step {
            Step::Reduce => {
                let node = node.expect("the post-loop plan has no node to reduce");
                reduction = NodeEvalReduction::prove(prover, node);
            }
            Step::Open(open) => prove_open(&open, prover),
            Step::Relation { proof, relation } => {
                let mut witnesses = relation_witnesses(relation, prover);
                let [witness] = witnesses.as_mut_slice() else {
                    panic!("node {proof_idx}'s {proof:?} step is not a single relation")
                };
                let (sumcheck, _) = prove_relation(
                    witness.as_mut(),
                    &mut prover.accumulator,
                    &mut prover.transcript,
                );
                proofs.push((ProofId(proof_idx, proof), sumcheck));
            }
            Step::Batch { proof, relations } => {
                let prepared: Vec<(&'static str, Prepared<F, T>)> = relations
                    .into_iter()
                    .map(|relation| prepare(relation, prover))
                    .collect();
                let witnesses = build_all(prepared, prover);
                proofs.push((ProofId(proof_idx, proof), prove_batch(witnesses, prover)));
            }
            Step::Tie { relation } => prove_tie(
                &relation,
                &prover.trace,
                &mut prover.accumulator,
                &mut prover.transcript,
            ),
            Step::Defer { batch, relation } => {
                let item = match batch.setup() {
                    DeferredSetup::InNodeLoop => {
                        let _span = witness_span(&relation).entered();
                        Deferred::Ready(relation_witnesses(relation, prover))
                    }
                    DeferredSetup::AtBatch => Deferred::Pending(Box::new(relation)),
                };
                prover.deferred.entry(batch).or_default().push(item);
            }
        }
    }
    (reduction, proofs)
}

/// Prove the deferral queue: every batch in [`DeferredBatch`] order, as one batched sumcheck
/// each, stored under [`DEFERRED_PROOF_IDX`].
pub fn prove_deferred_batches<F: JoltField, T: Transcript>(
    prover: &mut Prover<F, T>,
) -> ExecutionProofs<F> {
    let mut proofs = Vec::new();
    let mut queue = std::mem::take(&mut prover.deferred);
    for batch in DeferredBatch::ALL {
        let Some(deferred) = queue.remove(&batch) else {
            continue;
        };
        let _span =
            tracing::span!(tracing::Level::INFO, "deferred::batch", kind = ?batch).entered();
        let prepared: Vec<(&'static str, Prepared<F, T>)> = deferred
            .into_iter()
            .map(|item| match item {
                Deferred::Ready(witnesses) => ("ready", Prepared::Ready(witnesses)),
                Deferred::Pending(relation) => prepare(*relation, prover),
            })
            .collect();
        let witnesses = build_all(prepared, prover);
        // No instance, no proof (the verifier skips the batch the same way).
        if !witnesses.is_empty() {
            proofs.push((
                ProofId(DEFERRED_PROOF_IDX, batch.proof_type()),
                prove_batch(witnesses, prover),
            ));
        }
    }
    proofs
}

/// Prove `witnesses` as one batched sumcheck, instance messages computed in parallel across
/// instances.
fn prove_batch<F: JoltField, T: Transcript>(
    mut witnesses: Vec<Box<dyn DynRelationWitness<F, T>>>,
    prover: &mut Prover<F, T>,
) -> atlas_sumcheck::sumcheck::SumcheckInstanceProof<F> {
    let (proof, _) = if witnesses.len() > 1 {
        prove_relation_batch_parallel(
            witnesses.iter_mut().map(|w| &mut **w as _).collect(),
            &mut prover.accumulator,
            &mut prover.transcript,
        )
    } else {
        prove_relation_batch(
            witnesses.iter_mut().map(|w| &mut **w as _).collect(),
            &mut prover.accumulator,
            &mut prover.transcript,
        )
    };
    drop_in_background_thread(witnesses);
    proof
}

/// A relation of a batch after its sequential part (claims read, challenges drawn, openings
/// appended): its witnesses, or the parameters its heavy construction runs from in parallel.
enum Prepared<F: JoltField, T: Transcript> {
    /// Witnesses built already.
    Ready(Vec<Box<dyn DynRelationWitness<F, T>>>),
    /// A one-hot check's parameters and its lookup's read addresses.
    OneHot(Box<RaOneHotParams<F>>, Vec<usize>),
    /// A range bucket's value sumcheck parameters.
    RemainderValue(ValueParams<F>, usize),
    /// A range bucket's digit range-check parameters.
    RemainderDigits(DigitRangeParams<F>, usize),
}

/// The span a deferred relation's witness build runs under, so a batch's time stays
/// attributable to the kinds of relation in it.
fn witness_span<F: JoltField>(relation: &OpRelation<F>) -> tracing::Span {
    kind_span(witness_kind(relation))
}

fn kind_span(kind: &'static str) -> tracing::Span {
    tracing::span!(tracing::Level::INFO, "deferred::witness", kind)
}

fn witness_kind<F: JoltField>(relation: &OpRelation<F>) -> &'static str {
    match relation {
        OpRelation::TableRead(read) => match read.table {
            lookup::PrefixSuffixTable::Relu => "relu",
            lookup::PrefixSuffixTable::Clamp => "clamp",
            lookup::PrefixSuffixTable::ActivationClamp | lookup::PrefixSuffixTable::SiluClamp => {
                "activation_clamp"
            }
            lookup::PrefixSuffixTable::RightShift => "trig_downscale",
            lookup::PrefixSuffixTable::SoftmaxClamp => "softmax",
        },
        OpRelation::Op(relation) if relation.kind == OpRelKind::SmallTable => "small_table",
        OpRelation::SoftmaxRead(_) => "softmax",
        OpRelation::OneHot(_) => "one_hot",
        OpRelation::GatherSmallOneHot(_) | OpRelation::GatherSmallHammingWeight(_) => {
            "gather_one_hot"
        }
        OpRelation::Op(_)
        | OpRelation::RemainderBucketValue(_)
        | OpRelation::RemainderBucketDigits(_) => "node",
    }
}

/// The sequential part of a batch relation's prover.
fn prepare<F: JoltField, T: Transcript>(
    relation: OpRelation<F>,
    prover: &mut Prover<F, T>,
) -> (&'static str, Prepared<F, T>) {
    let kind = witness_kind(&relation);
    let _span = kind_span(kind).entered();
    let expect = "the prover holds every claim it reads";
    let prepared = match relation {
        OpRelation::OneHot(encoding) => {
            let challenges = shout::ra_onehot_challenges::<F, T>(&encoding, &mut prover.transcript);
            let params = shout::ra_onehot_params(&encoding, &prover.accumulator, challenges)
                .expect("the prover holds the one-hot checks' claims");
            Prepared::OneHot(
                Box::new(params),
                prover.take_addresses(Addresses(encoding.ra_source), encoding.node),
            )
        }
        OpRelation::RemainderBucketValue(bucket) => Prepared::RemainderValue(
            range_check::remainder_value_params(
                &bucket,
                &prover.accumulator,
                &mut prover.transcript,
            )
            .expect(expect),
            bucket.idx,
        ),
        OpRelation::RemainderBucketDigits(bucket) => Prepared::RemainderDigits(
            range_check::remainder_digit_range_params(
                &bucket,
                &prover.accumulator,
                &mut prover.transcript,
            )
            .expect(expect),
            bucket.idx,
        ),
        other => Prepared::Ready(relation_witnesses(other, prover)),
    };
    (kind, prepared)
}

/// The heavy part of a batch's provers, in parallel across the batch.
fn build_all<F: JoltField, T: Transcript>(
    prepared: Vec<(&'static str, Prepared<F, T>)>,
    prover: &Prover<F, T>,
) -> Vec<Box<dyn DynRelationWitness<F, T>>> {
    let build = tracing::span!(tracing::Level::INFO, "deferred::build").entered();
    let buckets = &prover.buckets;
    prepared
        .into_par_iter()
        .flat_map_iter(|(kind, p)| match p {
            Prepared::Ready(witnesses) => witnesses,
            p => {
                let _span = tracing::span!(
                    parent: build.id(),
                    tracing::Level::INFO,
                    "deferred::witness",
                    kind
                )
                .entered();
                build_one(p, buckets)
            }
        })
        .collect()
}

/// One relation's provers from its parameters.
fn build_one<F: JoltField, T: Transcript>(
    prepared: Prepared<F, T>,
    buckets: &crate::onnx_proof::range_bucket::BucketWitnesses,
) -> Vec<Box<dyn DynRelationWitness<F, T>>> {
    match prepared {
        Prepared::Ready(witnesses) => witnesses,
        Prepared::OneHot(params, indices) => {
            shout::prover::ra_onehot_provers_from_params::<F, T>(*params, &indices).into()
        }
        Prepared::RemainderValue(params, bucket) => {
            vec![Box::new(ValueProver::new(
                params,
                buckets.remainder_value_column(bucket),
            )) as Box<dyn DynRelationWitness<F, T>>]
        }
        Prepared::RemainderDigits(params, bucket) => {
            range_check::digit_columns_provers::<F, T>(params, buckets.remainder_digits(bucket))
        }
    }
}

/// The provers of a plan relation (one, or the several of a stage or a one-hot check), set up
/// against the prover's accumulator and transcript. A lookup leaves its read addresses on the
/// prover for the one-hot checks that follow it.
pub fn relation_witnesses<F: JoltField, T: Transcript>(
    relation: OpRelation<F>,
    prover: &mut Prover<F, T>,
) -> Vec<Box<dyn DynRelationWitness<F, T>>> {
    match relation {
        OpRelation::Op(relation) => {
            let params = op_rel_params(
                relation,
                &prover.trace,
                &mut prover.accumulator,
                &mut prover.transcript,
            );
            // A read through committed address chunks leaves its addresses for their
            // one-hot checks, under its read-address opening.
            let relation = params.relation();
            let node = &relation.node;
            let addresses = match (relation.kind, &node.operator) {
                (OpRelKind::SmallTable, _) => {
                    Some(lookup::small_table_lookup_indices(node, &prover.trace))
                }
                (OpRelKind::Gather, atlas_trace::Op::GatherLarge(_)) => {
                    Some(gather::large::gather_lookup_indices(node, &prover.trace))
                }
                _ => None,
            };
            if let (Some(indices), Some(ra)) = (addresses, relation.produced.first()) {
                prover.leave_addresses(Addresses(*ra), node.idx, indices);
            }
            vec![op_rel_prover(params, &prover.trace, &prover.accumulator)]
        }
        OpRelation::TableRead(read) => {
            let (witness, indices) = lookup::table_read_prover(
                &read,
                &prover.trace,
                &mut prover.accumulator,
                &mut prover.transcript,
            );
            // Reads that share one committed address (a smooth ReLU's clamp and ReLU reads)
            // leave it once.
            let key = (Addresses(read.ra.ra_source), read.node.idx);
            if !prover.lookup_addresses.contains_key(&key) {
                prover.leave_addresses(key.0, key.1, indices);
            }
            vec![witness]
        }
        OpRelation::GatherSmallOneHot(shape) => {
            let (hamming_booleanity, booleanity) = gather::small::build_stage2_provers::<F>(
                &shape.node,
                &prover.trace,
                &prover.accumulator,
                &mut prover.transcript,
            );
            vec![Box::new(hamming_booleanity), Box::new(booleanity)]
        }
        OpRelation::GatherSmallHammingWeight(shape) => {
            vec![Box::new(gather::small::build_stage3_prover::<F>(
                &shape.node,
                &prover.trace,
                &prover.accumulator,
                &mut prover.transcript,
            ))]
        }
        OpRelation::SoftmaxRead(shape) => {
            let node_idx = shape.idx();
            let (witness, indices) = softmax_last_axis::read_prover(
                &shape,
                &prover.trace,
                &prover.accumulator,
                &mut prover.transcript,
            );
            let ra = softmax_last_axis::read_ra_encoding(&shape.node).ra_source;
            prover.leave_addresses(Addresses(ra), node_idx, indices);
            vec![witness]
        }
        one_hot @ (OpRelation::OneHot(_)
        | OpRelation::RemainderBucketValue(_)
        | OpRelation::RemainderBucketDigits(_)) => {
            let prepared = prepare(one_hot, prover);
            build_all(vec![prepared], prover)
        }
    }
}

/// Append a plan opening's claims from the trace.
fn prove_open<F: JoltField, T: Transcript>(open: &OpOpen, prover: &mut Prover<F, T>) {
    let expect = "the prover holds every claim it reads";
    match open {
        OpOpen::RescaleRemainder(node) => {
            let remainder = rebase_remainder(node, &prover.trace);
            let accessor = AccOpeningAccessor::new(&mut prover.accumulator, node);
            let r0 = accessor.get_reduced_opening().expect(expect).0;
            let eval = remainder.evaluate_mle(&r0.r);
            accessor
                .into_provider(&mut prover.transcript, r0)
                .append_advice(VirtualPoly::RescaleRemainder, eval);
        }
        OpOpen::TrigPeriodQuotient(node) => {
            let quotient = TraceView::trig_period_advice(&prover.trace, node)
                .quotient
                .clone();
            let accessor = AccOpeningAccessor::new(&mut prover.accumulator, node);
            let r0 = accessor.get_reduced_opening().expect(expect).0;
            let eval = quotient.evaluate_mle(&r0.r);
            accessor
                .into_provider(&mut prover.transcript, r0)
                .append_advice(CommittedPoly::TrigPeriodQuotient, eval);
        }
        OpOpen::TrigDownscaled(node) => {
            let downscaled = trig::downscaled(node, &prover.trace);
            let accessor = AccOpeningAccessor::new(&mut prover.accumulator, node);
            let r0 = accessor.get_reduced_opening().expect(expect).0;
            let eval = downscaled.evaluate_mle(&r0.r);
            accessor
                .into_provider(&mut prover.transcript, r0)
                .append_advice(VirtualPoly::TrigDownscaled, eval);
        }
        OpOpen::SmoothReluRelu(node) => {
            let relu = TraceView::operand_tensors(&prover.trace, node)[0].map(|x| x.max(0));
            let accessor = AccOpeningAccessor::new(&mut prover.accumulator, node);
            let r0 = accessor.get_reduced_opening().expect(expect).0;
            let eval = relu.evaluate_mle(&r0.r);
            accessor
                .into_provider(&mut prover.transcript, r0)
                .append_advice(VirtualPoly::SmoothReluRelu, eval);
        }
        OpOpen::Softmax(open) => softmax_last_axis::prove_open(
            open,
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        ),
    }
}

/// The witnesses of an engine-proven [`OpRel`], aligned with its produced openings: a
/// polynomial over the sumcheck's hypercube whose final bound value is the opening's claim.
pub fn op_rel_witnesses<F: JoltField>(
    relation: &OpRel<F>,
    trace: &Trace,
    setup: &RelationSetup<F>,
) -> Vec<MultilinearPolynomial<F>> {
    let node = &relation.node;
    match relation.kind {
        OpRelKind::And | OpRelKind::Mul | OpRelKind::Square | OpRelKind::Cube => {
            mul::product_witnesses(node, relation.kind, trace)
        }
        OpRelKind::Iff => iff::select_witnesses(node, trace),
        OpRelKind::Div => div::division_witnesses(node, trace),
        OpRelKind::ConstantDivision => scalar_const_div::constant_division_witnesses(node, trace),
        OpRelKind::Rsqrt => rsqrt::reciprocal_sqrt_witnesses(node, trace),
        OpRelKind::SumAxis => sum::axis::axis_sum_witnesses(relation, trace, &setup.incoming_point),
        OpRelKind::SoftmaxSums => softmax_last_axis::sums_witnesses(node, trace),
        OpRelKind::MeanOfSquares
        | OpRelKind::Contraction
        | OpRelKind::Gather
        | OpRelKind::SmallTable => {
            panic!(
                "{:?} has a specialized prover, not the engine's",
                relation.kind
            )
        }
        OpRelKind::AddAccumulation
        | OpRelKind::SubAccumulation
        | OpRelKind::DivQuotient
        | OpRelKind::Identity
        | OpRelKind::Neg
        | OpRelKind::ZeroOutput
        | OpRelKind::Constant
        | OpRelKind::Input
        | OpRelKind::AxisPermutation
        | OpRelKind::Broadcast
        | OpRelKind::Reshape
        | OpRelKind::Slice
        | OpRelKind::Concat
        | OpRelKind::TrigPeriod
        | OpRelKind::SoftmaxAddressLink => {
            panic!("{:?} is a tie: it has no round witnesses", relation.kind)
        }
    }
}

/// The prover's values of a tie's produced openings, aligned with them at their opening
/// points: read off the trace, or taken from claims the accumulator already holds where the
/// tie fixes them.
pub fn op_rel_tie_claims<F: JoltField>(
    relation: &OpRel<F>,
    trace: &Trace,
    setup: &RelationSetup<F>,
    accumulator: &dyn OpeningAccumulator<F>,
) -> Vec<F> {
    let node = &relation.node;
    match relation.kind {
        OpRelKind::AddAccumulation | OpRelKind::SubAccumulation => {
            add::accumulation_claims(node, trace, &setup.incoming_point)
        }
        OpRelKind::DivQuotient => div::division_quotient_claims(node, accumulator),
        OpRelKind::Identity | OpRelKind::Neg => {
            neg::signed_copy_claims(node, relation.kind, accumulator)
        }
        OpRelKind::ZeroOutput => is_nan::zero_output_claims(node, trace, &setup.incoming_point),
        OpRelKind::Constant | OpRelKind::Input => constant::public_value_claims(),
        OpRelKind::AxisPermutation => moveaxis::axis_permutation_claims(node, accumulator),
        OpRelKind::Broadcast => broadcast::broadcast_claims(relation, trace, &setup.incoming_point),
        OpRelKind::Reshape => reshape::flat_copy_claims(node, trace, &setup.incoming_point),
        OpRelKind::Slice => slice::subcube_select_claims(relation, trace, &setup.incoming_point),
        OpRelKind::Concat => concat::subcube_tiling_claims(relation, trace, &setup.incoming_point),
        OpRelKind::TrigPeriod => trig::trig_period_claims(node, accumulator),
        OpRelKind::SoftmaxAddressLink => {
            softmax_last_axis::address_link_claims(node, trace, &setup.incoming_point)
        }
        OpRelKind::And
        | OpRelKind::Mul
        | OpRelKind::Square
        | OpRelKind::Cube
        | OpRelKind::Iff
        | OpRelKind::Div
        | OpRelKind::ConstantDivision
        | OpRelKind::Rsqrt
        | OpRelKind::SumAxis
        | OpRelKind::SoftmaxSums
        | OpRelKind::MeanOfSquares
        | OpRelKind::Contraction
        | OpRelKind::Gather
        | OpRelKind::SmallTable => {
            panic!("{:?} is proven by a sumcheck, not a tie", relation.kind)
        }
    }
}

/// The prover of an op relation set up as `params`: its specialized witness, or the engine's
/// over its witnesses.
pub fn op_rel_prover<F: JoltField, T: Transcript>(
    params: RelationParams<F, OpRel<F>>,
    trace: &Trace,
    accumulator: &dyn OpeningAccumulator<F>,
) -> Box<dyn DynRelationWitness<F, T>> {
    match params.relation().kind {
        OpRelKind::MeanOfSquares => {
            Box::new(MeanOfSquaresReductionProver::initialize(trace, params))
        }
        OpRelKind::Contraction => Box::new(EinsumDotProver::initialize(trace, params)),
        OpRelKind::Gather => Box::new(GatherProver::initialize(trace, params, accumulator)),
        OpRelKind::SmallTable => Box::new(lookup::SmallTableProver::initialize(
            trace,
            params,
            accumulator,
        )),
        OpRelKind::And
        | OpRelKind::Mul
        | OpRelKind::Square
        | OpRelKind::Cube
        | OpRelKind::Iff
        | OpRelKind::Div
        | OpRelKind::ConstantDivision
        | OpRelKind::Rsqrt
        | OpRelKind::SumAxis
        | OpRelKind::SoftmaxSums
        | OpRelKind::AddAccumulation
        | OpRelKind::SubAccumulation
        | OpRelKind::DivQuotient
        | OpRelKind::Identity
        | OpRelKind::Neg
        | OpRelKind::ZeroOutput
        | OpRelKind::Constant
        | OpRelKind::Input
        | OpRelKind::AxisPermutation
        | OpRelKind::Broadcast
        | OpRelKind::Reshape
        | OpRelKind::Slice
        | OpRelKind::Concat
        | OpRelKind::TrigPeriod
        | OpRelKind::SoftmaxAddressLink => Box::new(RelationProver::initialize(trace, params)),
    }
}

/// The prover's claims of an [`OpRel`]'s setup opens, aligned with them at their coordinates
/// of the incoming point.
pub fn op_rel_setup_claims<F: JoltField>(
    relation: &OpRel<F>,
    trace: &Trace,
    setup: &RelationSetup<F>,
) -> Vec<F> {
    match relation.kind {
        OpRelKind::Gather => gather::index_claim(relation, trace, setup),
        OpRelKind::SmallTable => lookup::small_table_setup_claims(relation, trace, setup),
        OpRelKind::And
        | OpRelKind::Mul
        | OpRelKind::Square
        | OpRelKind::Cube
        | OpRelKind::Iff
        | OpRelKind::Div
        | OpRelKind::ConstantDivision
        | OpRelKind::Rsqrt
        | OpRelKind::SumAxis
        | OpRelKind::SoftmaxSums
        | OpRelKind::MeanOfSquares
        | OpRelKind::Contraction
        | OpRelKind::AddAccumulation
        | OpRelKind::SubAccumulation
        | OpRelKind::DivQuotient
        | OpRelKind::Identity
        | OpRelKind::Neg
        | OpRelKind::ZeroOutput
        | OpRelKind::Constant
        | OpRelKind::Input
        | OpRelKind::AxisPermutation
        | OpRelKind::Broadcast
        | OpRelKind::Reshape
        | OpRelKind::Slice
        | OpRelKind::Concat
        | OpRelKind::TrigPeriod
        | OpRelKind::SoftmaxAddressLink => {
            panic!("{:?} has no setup opens", relation.kind)
        }
    }
}
