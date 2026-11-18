//! # Neural Network Op Proofs Module
//!
//! The prover side of every supported ONNX operation. The verifier crate defines each op's
//! relations and its plan, the steps it writes to the model's schedule
//! (`onnx_proof::schedule`); this module walks the schedule with the
//! relations' witnesses ([`prove_schedule`]), so the transcript order is the schedule's step
//! order on both sides.
//!
//! ## Op Categories
//!
//! **Arithmetic Operations**: [`add`], [`sub`], [`neg`], [`mul`], [`square`], [`cube`],
//! [`scalar_const_div`], [`mean_of_squares`]
//!
//! **Activation Functions**: [`relu`], [`activation_clamped`], [`smooth_relu`], [`trig`],
//! [`softmax_last_axis`]
//!
//! **Tensor Operations**: [`einsum`], [`sum`], [`reshape`], [`broadcast`], [`moveaxis`],
//! [`slice`], [`concat`], [`gather`]
//!
//! **Special Operations**: [`rsqrt`] (reciprocal square root), [`iff`] (conditional select)
//!
//! **Logic Operations**: [`and`], [`is_nan`]
//!
//! **Infrastructure**: [`input`], [`constant`], [`identity`], [`eval_reduction`]
//!
//! ## Architecture
//!
//! Every operator's protocol is a relation (`onnx_proof::relation`): its claim algebra, from
//! which the verifier is derived. The prover side of an op relation ([`OpRel`]) is a match on
//! its [`OpRelKind`] with no wildcard arm: a relation of the engine's shape is proven by
//! [`RelationProver`](crate::onnx_proof::relation::RelationProver) over its witnesses
//! ([`op_rel_witnesses`]), a tie's claims come from [`op_rel_tie_claims`], the claims of a
//! relation's setup opens from [`op_rel_setup_claims`], and a specialized witness (the einsum
//! dot engine, the mean-of-squares reduction, the gather and small-table provers) attaches in
//! [`op_rel_prover`]. The other [`OpRelation`] variants (the prefix-suffix reads, the one-hot
//! checks, the range buckets) get their provers in [`relation_witnesses`].

use crate::onnx_proof::ops::eval_reduction::NodeEvalReductionProve;
pub use jolt_atlas_verifier::onnx_proof::ops::*;
use std::collections::{BTreeMap, BTreeSet};

/// The clamped activations Erf, Sigmoid and Tanh.
pub mod activation_clamped;
/// Element-wise addition operation.
pub mod add;
/// Broadcast tensors to target shapes.
pub mod broadcast;
/// Concatenate tensors along an axis.
pub mod concat;
/// Constant tensor nodes.
pub mod constant;
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
/// Sin and Cos.
/// Rescale: a wide value taken to the model's scale.
pub mod rescale;
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
pub mod trig;

use atlas_algebra::{
    field::{IntoOpening, JoltField},
    poly::multilinear::EvaluateMle,
    utils::thread::drop_in_background_thread,
};
use atlas_transcript::ProvingTranscript;
use eval_reduction::NodeEvalReduction;

use crate::node_view::ModelView;
use crate::onnx_proof::{
    fused_rebase::rebase_remainder,
    lookup,
    ops::{
        einsum::dot::EinsumDotProver, gather::GatherProver,
        mean_of_squares::MeanOfSquaresReductionProver,
    },
    range_check::{self, ValueParams, ValueProver},
    relation::{
        op_rel_params, prove_tie, OpRel, OpRelKind, RelationParams, RelationProver, RelationSetup,
    },
    schedule::{OpOpen, OpRelation, RelId, Schedule, Step},
    trace_view::TraceView,
};
use crate::utils::opening_access::{AccOpeningAccessor, IntoProverProvider};
use atlas_algebra::poly::multilinear::MultilinearPolynomial;
use atlas_sumcheck::evaluation_reduction::ReducedInstance;
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_sumcheck::relation::prover::{
    prove_relation, prove_relation_batch, prove_relation_batch_parallel, DynRelationWitness,
};
use atlas_sumcheck::shout::{self, RaOneHotParams};
use atlas_trace::Trace;
use common::{CommittedPoly, VirtualPoly};
use rayon::prelude::*;

use crate::onnx_proof::Prover;

/// The key a read leaves its addresses under for its one-hot checks: the opening of its
/// read-address polynomial ([`RaEncoding::ra_source`](lookup::RaEncoding::ra_source)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Addresses(pub OpeningId);

/// Walk `schedule` on the prover side: every step the verifier's interpreter
/// ([`Verifier::run`](crate::onnx_proof::Verifier::run)) runs, in the same order, with the
/// relations' witnesses. A relation set up by a `Setup` step keeps its witnesses until the
/// `Sumcheck` step that names it; a batched sumcheck builds the witnesses of its other members
/// in parallel. Every round message, claim and evaluation-reduction polynomial is sent through
/// the prover's transcript at the step the verifier reads it.
#[tracing::instrument(skip_all, name = "prove_schedule")]
pub fn prove_schedule<F: JoltField, T: ProvingTranscript>(
    schedule: &Schedule<F>,
    prover: &mut Prover<F, T>,
) {
    let relation = |id: &RelId| {
        schedule
            .relation(*id)
            .expect("a finished schedule names its own relations")
    };
    let mut set_up: Vec<Option<(&'static str, Prepared<F, T>)>> =
        schedule.relations().iter().map(|_| None).collect();
    // The span of the node whose steps are running: a node's turn starts with its reduction
    // or with a sumcheck of its own, and ends at a sumcheck over several nodes' relations.
    let mut turn: Option<(usize, tracing::span::EnteredSpan)> = None;
    // The points drawn whose claims are not yet sent, and the claims computed at them: each
    // node's output, and its rescale remainder where the wave opens one.
    let mut drawn: BTreeMap<usize, Vec<F>> = BTreeMap::new();
    let mut output_claims: BTreeMap<usize, F> = BTreeMap::new();
    let mut remainder_claims: BTreeMap<usize, F> = BTreeMap::new();
    let steps = schedule.steps();
    for (at, step) in steps.iter().enumerate() {
        let step_node = match step {
            Step::Reduce { node } | Step::Point { node, .. } => Some(Some(*node)),
            Step::Sumcheck { members, .. } => Some(schedule.sumcheck_node(members)),
            Step::Draw { .. } | Step::Open(_) | Step::Setup(_) | Step::Tie(_) => None,
        };
        if let Some(node) = step_node.filter(|node| turn.as_ref().map(|turn| turn.0) != *node) {
            drop(turn.take());
            if let Some(node) = node {
                let op = prover.preprocessing.model().view(node).operator;
                turn = Some((
                    node,
                    tracing::span!(tracing::Level::INFO, "prove_node", node, op = %op).entered(),
                ));
            }
        }
        match step {
            Step::Reduce { node } => {
                let node = prover.preprocessing.model().view(*node);
                NodeEvalReduction::prove(prover, &node);
            }
            Step::Draw { node, vars } => {
                let point: Vec<F> = prover
                    .transcript
                    .challenge_vector_optimized::<F>(*vars)
                    .into_opening();
                drawn.insert(*node, point);
                // The last point of a wave is drawn: every claim at the wave's points is
                // computed here, side by side, and sent at its own step.
                if !matches!(steps.get(at + 1), Some(Step::Draw { .. })) {
                    let _span = tracing::info_span!("wave_claims").entered();
                    let (model, trace) = (prover.preprocessing.model(), &prover.trace);
                    let remainders: BTreeSet<usize> = steps[at + 1..]
                        .iter()
                        .take_while(|step| !matches!(step, Step::Sumcheck { batched: true, .. }))
                        .filter_map(|step| match step {
                            Step::Open(OpOpen::RescaleRemainder(node)) => Some(node.idx),
                            _ => None,
                        })
                        .collect();
                    let claims: Vec<(usize, F, Option<F>)> = drawn
                        .par_iter()
                        .map(|(&node, point)| {
                            let view = model.view(node);
                            let output = TraceView::layer_data(trace, &view).output;
                            let claim: F = output.evaluate_mle(point);
                            let remainder = remainders
                                .contains(&node)
                                .then(|| rebase_remainder(&view, trace).evaluate_mle(point));
                            (node, claim, remainder)
                        })
                        .collect();
                    for (node, claim, remainder) in claims {
                        output_claims.insert(node, claim);
                        if let Some(remainder) = remainder {
                            remainder_claims.insert(node, remainder);
                        }
                    }
                }
            }
            Step::Point { node, .. } => {
                let expect = "a point step follows its draw";
                let point = drawn.remove(node).expect(expect);
                let claim = output_claims.remove(node).expect(expect);
                prover.transcript.send_scalar(&claim);
                prover
                    .accumulator
                    .reduced_evaluations
                    .insert(*node, ReducedInstance { r: point, claim });
                prover.accumulator.own_points.insert(*node);
            }
            Step::Open(open) => {
                let computed = match open {
                    OpOpen::RescaleRemainder(node) => remainder_claims.remove(&node.idx),
                    _ => None,
                };
                prove_open(open, prover, computed)
            }
            Step::Setup(id) => {
                set_up[id.0] = Some(prepare(relation(id).clone(), prover));
            }
            Step::Sumcheck {
                members,
                batched: false,
            } => {
                let mut witnesses: Witnesses<F, T> = members
                    .iter()
                    .flat_map(|id| relation_witnesses(relation(id).clone(), prover))
                    .collect();
                let [witness] = witnesses.as_mut_slice() else {
                    panic!("the unbatched sumcheck of {members:?} is not a single relation")
                };
                prove_relation(
                    witness.as_mut(),
                    &mut prover.accumulator,
                    &mut prover.transcript,
                );
            }
            Step::Sumcheck {
                members,
                batched: true,
            } => {
                let first = relation(&members[0]);
                let _span = tracing::span!(
                    tracing::Level::INFO,
                    "deferred::batch",
                    kind = batch_kind(first)
                )
                .entered();
                let prepared: Vec<(&'static str, Prepared<F, T>)> = members
                    .iter()
                    .map(|id| {
                        set_up[id.0]
                            .take()
                            .unwrap_or_else(|| prepare(relation(id).clone(), prover))
                    })
                    .collect();
                let witnesses = build_all(prepared, prover);
                prove_batch(witnesses, prover);
                // The value sumcheck is the last reader of the bucket witnesses.
                if matches!(first, OpRelation::RemainderBucketValue(_)) {
                    drop_in_background_thread(std::mem::take(&mut prover.buckets));
                }
            }
            Step::Tie(id) => {
                let OpRelation::Op(relation) = relation(id) else {
                    panic!("the tie {id:?} is not an op relation")
                };
                // An input's tie takes its public tensor from the trace's IO.
                let with_input;
                let relation = if relation.kind == OpRelKind::Input {
                    with_input = constant::with_public_input(relation, &prover.io)
                        .expect("the trace holds every input");
                    &with_input
                } else {
                    relation
                };
                prove_tie(
                    relation,
                    &prover.trace,
                    &mut prover.accumulator,
                    &mut prover.transcript,
                );
            }
        }
    }
}

/// What a batched sumcheck proves, by the kind of its first member (a batch's members are of
/// one kind): the name of its span.
fn batch_kind<F: JoltField>(first: &OpRelation<F>) -> &'static str {
    match first {
        OpRelation::RemainderBucketValue(_) => "BucketValues",
        OpRelation::TableRead(_) | OpRelation::SoftmaxRead(_) => "Lookups",
        OpRelation::OneHot(_)
        | OpRelation::GatherSmallOneHot(_)
        | OpRelation::GatherSmallHammingWeight(_) => "OneHot",
        OpRelation::Op(_) => "Op",
    }
}

/// The witnesses of one relation, set up.
type Witnesses<F, T> = Vec<Box<dyn DynRelationWitness<F, T>>>;

/// Prove `witnesses` as one batched sumcheck, instance messages computed in parallel across
/// instances.
fn prove_batch<F: JoltField, T: ProvingTranscript>(
    mut witnesses: Vec<Box<dyn DynRelationWitness<F, T>>>,
    prover: &mut Prover<F, T>,
) {
    if witnesses.len() > 1 {
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
}

/// A relation of a batch after its sequential part (claims read, challenges drawn, openings
/// appended): its witnesses, or the parameters its heavy construction runs from in parallel.
enum Prepared<F: JoltField, T: ProvingTranscript> {
    /// Witnesses built already.
    Ready(Vec<Box<dyn DynRelationWitness<F, T>>>),
    /// A one-hot check's parameters and its lookup's read addresses.
    OneHot(Box<RaOneHotParams<F>>, Vec<usize>),
    /// A range bucket's value sumcheck parameters.
    RemainderValue(ValueParams<F>, usize),
    /// An op relation set up, whose witness is built from the trace, with the input claim
    /// its build is checked against ([`op_params`]).
    Op(Box<RelationParams<F, OpRel<F>>>, Option<F>),
    /// A relation set up, with the build of its witness.
    Deferred(lookup::DeferredWitness<F, T>),
}

fn kind_span(kind: &'static str) -> tracing::Span {
    tracing::span!(tracing::Level::INFO, "deferred::witness", kind)
}

fn witness_kind<F: JoltField>(relation: &OpRelation<F>) -> &'static str {
    match relation {
        OpRelation::TableRead(read) => match read.table {
            lookup::PrefixSuffixTable::Relu => "relu",
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
        OpRelation::Op(_) | OpRelation::RemainderBucketValue(_) => "node",
    }
}

/// The sequential part of a batch relation's prover.
fn prepare<F: JoltField, T: ProvingTranscript>(
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
        OpRelation::Op(relation) => {
            let (params, input_claim) = op_params(relation, prover);
            Prepared::Op(Box::new(params), input_claim)
        }
        OpRelation::TableRead(read) => Prepared::Deferred(table_read_setup(&read, prover)),
        other => Prepared::Ready(relation_witnesses(other, prover)),
    };
    (kind, prepared)
}

/// The heavy part of a batch's provers, in parallel across the batch.
fn build_all<F: JoltField, T: ProvingTranscript>(
    prepared: Vec<(&'static str, Prepared<F, T>)>,
    prover: &Prover<F, T>,
) -> Vec<Box<dyn DynRelationWitness<F, T>>> {
    let build = tracing::span!(tracing::Level::INFO, "deferred::build").entered();
    let (buckets, trace) = (&prover.buckets, &prover.trace);
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
                build_one(p, buckets, trace)
            }
        })
        .collect()
}

/// One relation's provers from its parameters.
fn build_one<F: JoltField, T: ProvingTranscript>(
    prepared: Prepared<F, T>,
    buckets: &crate::onnx_proof::range_bucket::BucketWitnesses,
    trace: &Trace,
) -> Vec<Box<dyn DynRelationWitness<F, T>>> {
    match prepared {
        Prepared::Ready(witnesses) => witnesses,
        Prepared::Op(params, input_claim) => vec![op_rel_prover(*params, trace, input_claim)],
        Prepared::Deferred(build) => vec![build()],
        Prepared::OneHot(params, indices) => {
            shout::prover::ra_onehot_provers_from_params::<F, T>(*params, &indices).into()
        }
        Prepared::RemainderValue(params, bucket) => {
            vec![Box::new(ValueProver::new(
                params,
                buckets.remainder_value_column(bucket),
                buckets.remainder_digits(bucket),
            )) as Box<dyn DynRelationWitness<F, T>>]
        }
    }
}

/// The sequential part of a table read's prover: its operand claim appended and its
/// challenge drawn. Reads that share one committed address (a smooth ReLU's clamp and ReLU
/// reads) leave it once for their one-hot checks.
fn table_read_setup<F: JoltField, T: ProvingTranscript>(
    read: &lookup::TableRead,
    prover: &mut Prover<F, T>,
) -> lookup::DeferredWitness<F, T> {
    let (build, indices) = lookup::table_read_prover(
        read,
        &prover.trace,
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    let key = (Addresses(read.ra.ra_source), read.node.idx);
    if !prover.lookup_addresses.contains_key(&key) {
        prover.leave_addresses(key.0, key.1, indices);
    }
    build
}

/// The sequential part of an op relation's prover: its setup. A read through committed
/// address chunks leaves its addresses for their one-hot checks, under its read-address
/// opening. In a test build the setup comes with the relation's input claim, which the
/// specialized provers that sum their own witness check it against.
fn op_params<F: JoltField, T: ProvingTranscript>(
    relation: OpRel<F>,
    prover: &mut Prover<F, T>,
) -> (RelationParams<F, OpRel<F>>, Option<F>) {
    let params = op_rel_params(
        relation,
        &prover.trace,
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    let relation = params.relation();
    let node = &relation.node;
    let addresses = if relation.kind == OpRelKind::SmallTable {
        let def = relation
            .small_table()
            .expect("the relation was built for a small table");
        Some(lookup::small_table_lookup_indices(node, def, &prover.trace))
    } else if relation.kind == OpRelKind::Gather
        && matches!(node.operator, atlas_trace::Op::GatherLarge(_))
    {
        Some(gather::large::gather_lookup_indices(node, &prover.trace))
    } else {
        None
    };
    if let (Some(indices), Some(ra)) = (addresses, relation.produced.first()) {
        prover.leave_addresses(Addresses(*ra), node.idx, indices);
    }
    let input_claim = cfg!(test).then(|| {
        crate::onnx_proof::relation::relation_input_claim(
            relation,
            params.setup(),
            &prover.accumulator,
        )
        .expect("the prover holds every claim it reads")
    });
    (params, input_claim)
}

/// The provers of a schedule's relation (one, or the several of a stage or a one-hot check), set up
/// against the prover's accumulator and transcript. A lookup leaves its read addresses on the
/// prover for the one-hot checks that follow it.
pub fn relation_witnesses<F: JoltField, T: ProvingTranscript>(
    relation: OpRelation<F>,
    prover: &mut Prover<F, T>,
) -> Vec<Box<dyn DynRelationWitness<F, T>>> {
    match relation {
        OpRelation::Op(relation) => {
            let (params, input_claim) = op_params(relation, prover);
            vec![op_rel_prover(params, &prover.trace, input_claim)]
        }
        OpRelation::TableRead(read) => vec![table_read_setup(&read, prover)()],
        OpRelation::GatherSmallOneHot(shape) => {
            vec![Box::new(gather::small::build_stage2_prover::<F>(
                &shape.node,
                &prover.trace,
                &prover.accumulator,
                &mut prover.transcript,
            ))]
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
        one_hot @ (OpRelation::OneHot(_) | OpRelation::RemainderBucketValue(_)) => {
            let prepared = prepare(one_hot, prover);
            build_all(vec![prepared], prover)
        }
    }
}

/// Append an open step's claims from the trace. `computed` is a rescale remainder's claim
/// where its wave computed it with its point.
fn prove_open<F: JoltField, T: ProvingTranscript>(
    open: &OpOpen,
    prover: &mut Prover<F, T>,
    computed: Option<F>,
) {
    let expect = "the prover holds every claim it reads";
    match open {
        OpOpen::RescaleRemainder(node) => {
            let accessor = AccOpeningAccessor::new(&mut prover.accumulator, node);
            let r0 = accessor.get_reduced_opening().expect(expect).0;
            let eval = computed
                .unwrap_or_else(|| rebase_remainder(node, &prover.trace).evaluate_mle(&r0.r));
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
            mul::product_witnesses(relation, trace)
        }
        OpRelKind::Iff => iff::select_witnesses(node, trace),
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
        | OpRelKind::Rescale
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
        OpRelKind::Identity | OpRelKind::Neg => {
            neg::signed_copy_claims(node, relation.kind, accumulator)
        }
        OpRelKind::ZeroOutput => is_nan::zero_output_claims(node, trace, &setup.incoming_point),
        OpRelKind::Constant | OpRelKind::Input => constant::public_value_claims(),
        OpRelKind::AxisPermutation => moveaxis::axis_permutation_claims(node, accumulator),
        OpRelKind::Broadcast => broadcast::broadcast_claims(relation, trace, &setup.incoming_point),
        OpRelKind::Reshape => reshape::flat_copy_claims(node, accumulator),
        OpRelKind::Slice => slice::subcube_select_claims(relation, trace, &setup.incoming_point),
        OpRelKind::Concat => concat::subcube_tiling_claims(relation, trace, &setup.incoming_point),
        OpRelKind::TrigPeriod => trig::trig_period_claims(node, accumulator),
        OpRelKind::Rescale => rescale::rescale_claims(node, trace, &setup.incoming_point),
        OpRelKind::SoftmaxAddressLink => {
            softmax_last_axis::address_link_claims(node, trace, &setup.incoming_point)
        }
        OpRelKind::And
        | OpRelKind::Mul
        | OpRelKind::Square
        | OpRelKind::Cube
        | OpRelKind::Iff
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
/// over its witnesses, built from the trace and the setup. A gather's and a small-table
/// read's check the sum of their witness against `input_claim` when given one.
pub fn op_rel_prover<F: JoltField, T: ProvingTranscript>(
    params: RelationParams<F, OpRel<F>>,
    trace: &Trace,
    input_claim: Option<F>,
) -> Box<dyn DynRelationWitness<F, T>> {
    match params.relation().kind {
        OpRelKind::MeanOfSquares => {
            Box::new(MeanOfSquaresReductionProver::initialize(trace, params))
        }
        OpRelKind::Contraction => Box::new(EinsumDotProver::initialize(trace, params)),
        OpRelKind::Gather => Box::new(GatherProver::initialize(trace, params, input_claim)),
        OpRelKind::SmallTable => Box::new(lookup::SmallTableProver::initialize(
            trace,
            params,
            input_claim,
        )),
        OpRelKind::And
        | OpRelKind::Mul
        | OpRelKind::Square
        | OpRelKind::Cube
        | OpRelKind::Iff
        | OpRelKind::ConstantDivision
        | OpRelKind::Rsqrt
        | OpRelKind::SumAxis
        | OpRelKind::SoftmaxSums
        | OpRelKind::AddAccumulation
        | OpRelKind::SubAccumulation
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
        | OpRelKind::Rescale
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
        | OpRelKind::ConstantDivision
        | OpRelKind::Rsqrt
        | OpRelKind::SumAxis
        | OpRelKind::SoftmaxSums
        | OpRelKind::MeanOfSquares
        | OpRelKind::Contraction
        | OpRelKind::AddAccumulation
        | OpRelKind::SubAccumulation
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
        | OpRelKind::Rescale
        | OpRelKind::SoftmaxAddressLink => {
            panic!("{:?} has no setup opens", relation.kind)
        }
    }
}
