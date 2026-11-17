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
//! reduction, the lookup provers) attaches at its [`OpRelation`] variant in
//! [`relation_witnesses`].

use crate::onnx_proof::ops::eval_reduction::NodeEvalReductionProve;
pub use jolt_atlas_verifier::onnx_proof::ops::*;

/// Shared plumbing for the clamped Erf/Sigmoid/Tanh variants.
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

use crate::node_view::ComputationNode;

use atlas_algebra::{
    field::JoltField, poly::multilinear::EvaluateMle, utils::thread::drop_in_background_thread,
};
use atlas_sumcheck::evaluation_reduction::EvalReductionProof;
use atlas_transcript::Transcript;
use eval_reduction::NodeEvalReduction;

use crate::onnx_proof::{
    clamp_split::{self, ValueParams, ValueProver},
    deferred_lookups::{DeferredBatch, DeferredSetup, DEFERRED_PROOF_IDX},
    fused_rebase::rebase_remainder,
    neural_teleport::{
        cos::CosTable,
        sin::SinTable,
        small_table::{SmallTable, SmallTableRelation, TrigTableProver},
        trig_downscale::TrigDownscaleOperands,
    },
    op_lookups::{DefaultLookupOperands, OpLookupEncoding, OpLookupProve, OpLookupProvider},
    ops::softmax_last_axis::SoftmaxRelation,
    ops::{
        activation_clamped::{
            small_table_params, ActivationClampOperands, SmallTableProver, SmallTableRaEncoding,
        },
        clamp::SymmetricClampOperands,
        cos::CosRaEncoding,
        einsum::dot::EinsumDotProver,
        erf::ErfTableMarker,
        gather::GatherProver,
        mean_of_squares::MeanOfSquaresReductionProver,
        sigmoid::SigmoidTableMarker,
        sin::SinRaEncoding,
        smooth_relu::{GeluErfTable, GeluTanhTable, SiluTable, SmoothReluOperands},
        softmax_last_axis::SoftmaxAddresses,
        tanh::TanhTableMarker,
    },
    plan::{OpActivation, OpLookup, OpOpen, OpPlan, OpRelation, OpTie, OpTrig, Plan, Step},
    relation::{prove_tie, RelationParams, RelationProver, RelationWitnesses},
    trace_view::TraceView,
    ProofId,
};
use crate::utils::opening_access::{AccOpeningAccessor, IntoProverProvider, Target};
use atlas_lookup_tables::{
    clamp::{ActivationClampTable, ClampTable, SiluClampTable},
    relu::ReluTable,
    right_shift::RightShiftTable,
};
use atlas_sumcheck::digit_range::DigitRangeParams;
use atlas_sumcheck::relation::prover::{
    prove_relation, prove_relation_batch, prove_relation_batch_parallel, DynRelationWitness,
};
use atlas_sumcheck::shout::{self, RaOneHotEncoding, RaOneHotParams};
use common::consts::{TRIG_DOWNSCALE_BITS, XLEN};
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

/// The read addresses a lookup leaves for its one-hot checks, keyed by the lookup and its
/// node (`usize::MAX` for a model-level one).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Addresses {
    /// An [`OpRelation::Lookup`].
    Lookup(OpLookup),
    /// An [`OpRelation::SmallActivation`].
    SmallActivation,
    /// An [`OpRelation::Trig`].
    Trig,
    /// A softmax stage's lookup.
    Softmax(SoftmaxAddresses),
}

/// A queued deferred relation on the prover's side: its witnesses built in the node's turn,
/// or the relation waiting for its batch.
pub enum Deferred<F: JoltField, T: Transcript> {
    /// Built in the node's turn.
    Ready(Vec<Box<dyn DynRelationWitness<F, T>>>),
    /// Built when its batch runs.
    Pending(Box<OpRelation>),
}

/// Walk `node`'s plan on the prover side: every step the verifier's interpreter
/// ([`Verifier::run_plan`](crate::onnx_proof::Verifier::run_plan)) runs, in the same order, with
/// the relations' witnesses.
#[tracing::instrument(skip_all, name = "prove_plan", fields(node = node.idx, op = %node.operator))]
pub fn prove_plan<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
    plan: Plan,
) -> (Option<EvalReductionProof<F>>, ExecutionProofs<F>) {
    prove_steps(prover, node.idx, Some(node), plan.0)
}

/// Run `steps` whose sumcheck proofs are stored under `proof_idx`; `node` is the node of a
/// node plan (`None` for the post-loop plan, which has no evaluation reduction).
pub fn prove_steps<F: JoltField, T: Transcript>(
    prover: &mut Prover<F, T>,
    proof_idx: usize,
    node: Option<&ComputationNode>,
    steps: Vec<Step>,
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
            Step::Tie { relation } => prove_op_tie(&relation, prover),
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
fn witness_span(relation: &OpRelation) -> tracing::Span {
    kind_span(witness_kind(relation))
}

fn kind_span(kind: &'static str) -> tracing::Span {
    tracing::span!(tracing::Level::INFO, "deferred::witness", kind)
}

fn witness_kind(relation: &OpRelation) -> &'static str {
    match relation {
        OpRelation::Lookup(OpLookup::Relu, _) => "relu",
        OpRelation::Lookup(OpLookup::SymmetricClamp, _) => "clamp",
        OpRelation::Lookup(OpLookup::ActivationClamp, _) => "activation_clamp",
        OpRelation::Lookup(OpLookup::TrigDownscale, _) => "trig_downscale",
        OpRelation::SmallActivation(..) => "small_activation",
        OpRelation::Trig(..) => "trig",
        OpRelation::Softmax(_) => "softmax",
        OpRelation::LookupOneHot(..)
        | OpRelation::SmallActivationOneHot(_)
        | OpRelation::TrigOneHot(..) => "one_hot",
        OpRelation::GatherLargeOneHot(_)
        | OpRelation::GatherSmallOneHot(_)
        | OpRelation::GatherSmallHammingWeight(_) => "gather_one_hot",
        _ => "node",
    }
}

/// The sequential part of a batch relation's prover.
fn prepare<F: JoltField, T: Transcript>(
    relation: OpRelation,
    prover: &mut Prover<F, T>,
) -> (&'static str, Prepared<F, T>) {
    let kind = witness_kind(&relation);
    let _span = kind_span(kind).entered();
    let expect = "the prover holds every claim it reads";
    fn onehot<F: JoltField, T: Transcript>(
        encoding: &impl RaOneHotEncoding,
        key: Addresses,
        node_idx: usize,
        prover: &mut Prover<F, T>,
    ) -> Prepared<F, T> {
        let challenges = shout::ra_onehot_challenges::<F, T>(encoding, &mut prover.transcript);
        let params = shout::ra_onehot_params(encoding, &prover.accumulator, challenges)
            .expect("the prover holds the one-hot checks' claims");
        Prepared::OneHot(Box::new(params), prover.take_addresses(key, node_idx))
    }
    let prepared = match relation {
        OpRelation::LookupOneHot(kind @ OpLookup::Relu, node) => onehot(
            &OpLookupEncoding::<DefaultLookupOperands>::new(&node),
            Addresses::Lookup(kind),
            node.idx,
            prover,
        ),
        OpRelation::LookupOneHot(kind @ OpLookup::SymmetricClamp, node) => onehot(
            &OpLookupEncoding::<SymmetricClampOperands>::new(&node),
            Addresses::Lookup(kind),
            node.idx,
            prover,
        ),
        // A smooth ReLU's clamp and ReLU reads share one committed address, left under the
        // clamp's key.
        OpRelation::LookupOneHot(
            OpLookup::ActivationClamp | OpLookup::SiluClamp | OpLookup::SmoothRelu,
            node,
        ) => onehot(
            &OpLookupEncoding::<ActivationClampOperands>::new(&node),
            Addresses::Lookup(OpLookup::ActivationClamp),
            node.idx,
            prover,
        ),
        OpRelation::LookupOneHot(kind @ OpLookup::TrigDownscale, node) => onehot(
            &OpLookupEncoding::<TrigDownscaleOperands>::new(&node),
            Addresses::Lookup(kind),
            node.idx,
            prover,
        ),
        OpRelation::SmallActivationOneHot(node) => onehot(
            &SmallTableRaEncoding::for_node(&node),
            Addresses::SmallActivation,
            node.idx,
            prover,
        ),
        OpRelation::TrigOneHot(OpTrig::Cos, node) => onehot(
            &CosRaEncoding::new(&node),
            Addresses::Trig,
            node.idx,
            prover,
        ),
        OpRelation::TrigOneHot(OpTrig::Sin, node) => onehot(
            &SinRaEncoding::new(&node),
            Addresses::Trig,
            node.idx,
            prover,
        ),
        OpRelation::RemainderBucketValue(bucket) => Prepared::RemainderValue(
            clamp_split::remainder_value_params(
                &bucket,
                &prover.accumulator,
                &mut prover.transcript,
            )
            .expect(expect),
            bucket.idx,
        ),
        OpRelation::RemainderBucketDigits(bucket) => Prepared::RemainderDigits(
            clamp_split::remainder_digit_range_params(
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
    buckets: &crate::onnx_proof::global_clamp::BucketWitnesses,
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
            clamp_split::digit_columns_provers::<F, T>(params, buckets.remainder_digits(bucket))
        }
    }
}

/// The provers of a plan relation (one, or the several of a stage or a one-hot check), set up
/// against the prover's accumulator and transcript. A lookup leaves its read addresses on the
/// prover for the one-hot checks that follow it.
pub fn relation_witnesses<F: JoltField, T: Transcript>(
    relation: OpRelation,
    prover: &mut Prover<F, T>,
) -> Vec<Box<dyn DynRelationWitness<F, T>>> {
    fn engine<F: JoltField, T: Transcript, R: RelationWitnesses<F> + 'static>(
        relation: R,
        prover: &mut Prover<F, T>,
    ) -> Vec<Box<dyn DynRelationWitness<F, T>>> {
        let params = RelationParams::new(relation, &prover.accumulator, &mut prover.transcript);
        vec![Box::new(RelationProver::initialize(&prover.trace, params))]
    }
    fn lookup<F: JoltField, T: Transcript, H, LUT>(
        kind: OpLookup,
        node: ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<Box<dyn DynRelationWitness<F, T>>>
    where
        H: Default
            + crate::onnx_proof::op_lookups::LookupWitness
            + crate::onnx_proof::op_lookups::LookupRaPolys,
        LUT: atlas_lookup_tables::JoltLookupTable
            + atlas_lookup_tables::PrefixSuffixDecompositionTrait<XLEN>
            + Default
            + 'static,
    {
        let node_idx = node.idx;
        let (witness, indices) = OpLookupProvider::<H>::new(node)
            .read_raf_prove::<F, T, LUT, XLEN>(
                &prover.trace,
                &mut prover.accumulator,
                &mut prover.transcript,
            );
        prover.leave_addresses(Addresses::Lookup(kind), node_idx, indices);
        vec![Box::new(witness)]
    }
    fn small_activation<F: JoltField, T: Transcript, Table: SmallTable>(
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<Box<dyn DynRelationWitness<F, T>>> {
        let params =
            small_table_params::<F, Table>(node, &prover.accumulator, &mut prover.transcript)
                .expect("the prover holds every claim it reads");
        let exec = SmallTableProver::<F, Table>::initialize(
            params,
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        prover.leave_addresses(
            Addresses::SmallActivation,
            node.idx,
            exec.lookup_indices.clone(),
        );
        vec![Box::new(exec)]
    }
    fn trig<
        F: JoltField,
        T: Transcript,
        Table: crate::onnx_proof::neural_teleport::small_table::SmallTable,
    >(
        node: ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<Box<dyn DynRelationWitness<F, T>>> {
        let node_idx = node.idx;
        let downscaled = trig_downscaled(&node, &prover.trace);
        let params = RelationParams::new(
            SmallTableRelation::<Table>::new(node),
            &prover.accumulator,
            &mut prover.transcript,
        );
        let (exec, indices) =
            TrigTableProver::initialize(&downscaled, params, &mut prover.accumulator);
        prover.leave_addresses(Addresses::Trig, node_idx, indices);
        vec![Box::new(exec)]
    }
    match relation {
        OpRelation::And(relation) => engine(relation, prover),
        OpRelation::ConstantDivision(relation) => engine(relation, prover),
        OpRelation::Contraction(relation) => {
            let params = RelationParams::new(relation, &prover.accumulator, &mut prover.transcript);
            vec![Box::new(EinsumDotProver::initialize(&prover.trace, params))]
        }
        OpRelation::Cube(relation) => engine(relation, prover),
        OpRelation::Div(relation) => engine(relation, prover),
        OpRelation::Gather(relation) => {
            let params = RelationParams::new(relation, &prover.accumulator, &mut prover.transcript);
            vec![Box::new(GatherProver::initialize(
                &prover.trace,
                params,
                &mut prover.accumulator,
                &mut prover.transcript,
            ))]
        }
        OpRelation::GatherLargeOneHot(node) => {
            let encoding = gather::large::GatherRaEncoding::new(&node)
                .expect("the node is a large gather with an index operand");
            let indices = gather::large::gather_lookup_indices(&node, &prover.trace);
            shout::prover::ra_onehot_provers(
                &encoding,
                &indices,
                &prover.accumulator,
                &mut prover.transcript,
            )
            .into()
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
        OpRelation::Iff(relation) => engine(relation, prover),
        OpRelation::Lookup(kind @ OpLookup::Relu, node) => {
            lookup::<F, T, DefaultLookupOperands, ReluTable<XLEN>>(kind, node, prover)
        }
        OpRelation::Lookup(kind @ OpLookup::SymmetricClamp, node) => {
            lookup::<F, T, SymmetricClampOperands, ClampTable<XLEN>>(kind, node, prover)
        }
        OpRelation::Lookup(kind @ OpLookup::ActivationClamp, node) => {
            lookup::<F, T, ActivationClampOperands, ActivationClampTable<XLEN>>(kind, node, prover)
        }
        OpRelation::Lookup(kind @ OpLookup::TrigDownscale, node) => {
            lookup::<F, T, TrigDownscaleOperands, RightShiftTable<XLEN>>(kind, node, prover)
        }
        OpRelation::Lookup(OpLookup::SiluClamp, node) => {
            lookup::<F, T, ActivationClampOperands, SiluClampTable<XLEN>>(
                OpLookup::ActivationClamp,
                node,
                prover,
            )
        }
        OpRelation::Lookup(OpLookup::SmoothRelu, node) => {
            // The same addresses as the node's clamp read, which leaves them for the one-hot
            // checks.
            let (witness, _) = OpLookupProvider::<SmoothReluOperands>::new(node)
                .read_raf_prove::<F, T, ReluTable<XLEN>, XLEN>(
                    &prover.trace,
                    &mut prover.accumulator,
                    &mut prover.transcript,
                );
            vec![Box::new(witness)]
        }
        OpRelation::MeanOfSquares(relation) => {
            let params = RelationParams::new(relation, &prover.accumulator, &mut prover.transcript);
            vec![Box::new(MeanOfSquaresReductionProver::initialize(
                &prover.trace,
                params,
            ))]
        }
        OpRelation::Mul(relation) => engine(relation, prover),
        OpRelation::Rsqrt(relation) => engine(relation, prover),
        OpRelation::SmallActivation(OpActivation::Erf, node) => {
            small_activation::<F, T, ErfTableMarker>(&node, prover)
        }
        OpRelation::SmallActivation(OpActivation::Sigmoid, node) => {
            small_activation::<F, T, SigmoidTableMarker>(&node, prover)
        }
        OpRelation::SmallActivation(OpActivation::Tanh, node) => {
            small_activation::<F, T, TanhTableMarker>(&node, prover)
        }
        OpRelation::SmallActivation(OpActivation::GeluErf, node) => {
            small_activation::<F, T, GeluErfTable>(&node, prover)
        }
        OpRelation::SmallActivation(OpActivation::GeluTanh, node) => {
            small_activation::<F, T, GeluTanhTable>(&node, prover)
        }
        OpRelation::SmallActivation(OpActivation::Silu, node) => {
            small_activation::<F, T, SiluTable>(&node, prover)
        }
        OpRelation::Softmax(relation) => {
            let Prover {
                trace,
                accumulator,
                transcript,
                lookup_addresses,
                ..
            } = prover;
            let node_idx = match &relation {
                SoftmaxRelation::Sums(s)
                | SoftmaxRelation::Read(s)
                | SoftmaxRelation::Clamp(s)
                | SoftmaxRelation::ReadOneHot(s)
                | SoftmaxRelation::ClampOneHot(s) => s.idx(),
            };
            let (witnesses, left) = softmax_last_axis::prove_relation(
                relation,
                trace,
                accumulator,
                transcript,
                &mut |key| lookup_addresses.remove(&(Addresses::Softmax(key), node_idx)),
            );
            for (key, indices) in left {
                prover.leave_addresses(Addresses::Softmax(key), node_idx, indices);
            }
            witnesses
        }
        OpRelation::Square(relation) => engine(relation, prover),
        OpRelation::SumAxis(relation) => engine(relation, prover),
        OpRelation::Trig(OpTrig::Cos, node) => trig::<F, T, CosTable>(node, prover),
        OpRelation::Trig(OpTrig::Sin, node) => trig::<F, T, SinTable>(node, prover),
        one_hot @ (OpRelation::LookupOneHot(..)
        | OpRelation::SmallActivationOneHot(_)
        | OpRelation::TrigOneHot(..)
        | OpRelation::RemainderBucketValue(_)
        | OpRelation::RemainderBucketDigits(_)) => {
            let prepared = prepare(one_hot, prover);
            build_all(vec![prepared], prover)
        }
    }
}

/// A trig node's downscaled teleport remainder, `(input mod P) >> TRIG_DOWNSCALE_BITS`.
fn trig_downscaled(node: &ComputationNode, trace: &atlas_trace::Trace) -> atlas_trace::Tensor<i32> {
    TraceView::teleport_advice(trace, node)
        .remainder
        .map(|v| v >> TRIG_DOWNSCALE_BITS)
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
        OpOpen::TeleportQuotient(node) => {
            let quotient = TraceView::teleport_advice(&prover.trace, node)
                .quotient
                .clone();
            let accessor = AccOpeningAccessor::new(&mut prover.accumulator, node);
            let r0 = accessor.get_reduced_opening().expect(expect).0;
            let eval = quotient.evaluate_mle(&r0.r);
            accessor
                .into_provider(&mut prover.transcript, r0)
                .append_advice(CommittedPoly::TeleportNodeQuotient, eval);
        }
        OpOpen::TrigDownscaled(node) => {
            let downscaled = trig_downscaled(node, &prover.trace);
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

/// Append a plan tie's produced openings with the prover's claims.
fn prove_op_tie<F: JoltField, T: Transcript>(relation: &OpTie, prover: &mut Prover<F, T>) {
    let expect = "the prover holds every claim it reads";
    let Prover {
        trace,
        accumulator,
        transcript,
        ..
    } = prover;
    match relation {
        OpTie::AddAccumulation(relation) => prove_tie(relation, trace, accumulator, transcript),
        OpTie::AxisPermutation(relation) => prove_tie(relation, trace, accumulator, transcript),
        OpTie::Broadcast(relation) => prove_tie(relation, trace, accumulator, transcript),
        OpTie::Concat(relation) => prove_tie(relation, trace, accumulator, transcript),
        OpTie::Constant(relation) => prove_tie(relation, trace, accumulator, transcript),
        OpTie::Input(relation) => prove_tie(relation, trace, accumulator, transcript),
        OpTie::DivQuotient(relation) => prove_tie(relation, trace, accumulator, transcript),
        OpTie::Identity(relation) => prove_tie(relation, trace, accumulator, transcript),
        OpTie::Neg(relation) => prove_tie(relation, trace, accumulator, transcript),
        OpTie::Reshape(relation) => prove_tie(relation, trace, accumulator, transcript),
        OpTie::Slice(relation) => prove_tie(relation, trace, accumulator, transcript),
        OpTie::SubAccumulation(relation) => prove_tie(relation, trace, accumulator, transcript),
        OpTie::ZeroOutput(relation) => prove_tie(relation, trace, accumulator, transcript),
        OpTie::TeleportInput(relation) => {
            let node = &relation.node;
            let accessor = AccOpeningAccessor::new(&*accumulator, node);
            let (r_node_output, _) = accessor.get_reduced_opening().expect(expect);
            let q_claim = accessor
                .get_advice(CommittedPoly::TeleportNodeQuotient)
                .expect(expect)
                .1;
            let r_claim = accessor
                .get_advice(VirtualPoly::TeleportRemainder)
                .expect(expect)
                .1;
            let a_claim = F::from_i32(relation.tau) * q_claim + r_claim;
            AccOpeningAccessor::new(accumulator, node)
                .into_provider(transcript, r_node_output)
                .append_nodeio(Target::Input(0), a_claim);
        }
        OpTie::Softmax(tie) => softmax_last_axis::prove_tie(tie, trace, accumulator, transcript),
    }
}
