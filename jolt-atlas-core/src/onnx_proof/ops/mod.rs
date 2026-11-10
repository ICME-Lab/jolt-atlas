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
    field::JoltField,
    poly::{multilinear_polynomial::EvaluateMle, opening_point::OpeningPoint},
    utils::thread::drop_in_background_thread,
};
use atlas_sumcheck::evaluation_reduction::EvalReductionProof;
use atlas_transcript::Transcript;
use eval_reduction::NodeEvalReduction;

use crate::onnx_proof::{
    clamp_lookups::{clamp_intermediate, SaturatingAccClampOperands},
    clamp_split::{self, SplitParams, SplitProver, ValueParams, ValueProver},
    deferred_lookups::{DeferredBatch, DeferredSetup, DEFERRED_PROOF_IDX},
    fused_rebase::rebase_remainder,
    neural_teleport::{
        cos::CosTable,
        division::compute_division,
        sin::SinTable,
        small_table::{SmallTableRelation, TrigTableProver},
        trig_downscale::TrigDownscaleOperands,
    },
    op_lookups::{DefaultLookupOperands, OpLookupEncoding, OpLookupProve, OpLookupProvider},
    ops::softmax_last_axis::SoftmaxRelation,
    ops::{
        activation_clamped::{
            small_table_params, ActivationClampOperands, SmallActivationTable, SmallTableProver,
            SmallTableRaEncoding,
        },
        clamp::SymmetricClampOperands,
        cos::CosRaEncoding,
        einsum::dot::EinsumDotProver,
        erf::ErfTableMarker,
        gather::GatherProver,
        mean_of_squares::MeanOfSquaresReductionProver,
        sigmoid::SigmoidTableMarker,
        sin::SinRaEncoding,
        softmax_last_axis::SoftmaxAddresses,
        tanh::TanhTableMarker,
    },
    plan::{
        OpActivation, OpLookup, OpOpen, OpPlan, OpRangeCheck, OpRelation, OpTie, OpTrig, Plan, Step,
    },
    range_checking::{
        range_check_operands::RangeCheckWitness,
        range_check_operands::{
            DivRangeCheckOperands, MeanOfSquaresRangeCheckOperands, RangeCheckOperands,
            RangeCheckingOperandsTrait, RiRangeCheckOperands, RsRangeCheckOperands,
            TeleportRangeCheckOperands,
        },
        RangeCheckProve, RangeCheckProvider,
    },
    relation::{prove_tie, RelationParams, RelationProver, RelationWitnesses},
    trace_view::TraceView,
    ProofId,
};
use crate::utils::opening_access::{AccOpeningAccessor, IntoProverProvider, Target};
use atlas_lookup_tables::{
    clamp::{ActivationClampTable, ClampTable},
    relu::ReluTable,
    right_shift::RightShiftTable,
    unsigned_less_than::UnsignedLessThanTable,
};
use atlas_sumcheck::booleanity::BooleanitySumcheckParams;
use atlas_sumcheck::digit_range::DigitRangeParams;
use atlas_sumcheck::relation::prover::{
    prove_relation, prove_relation_batch, prove_relation_batch_parallel, DynRelationWitness,
};
use atlas_sumcheck::shout::{self, RaOneHotEncoding, RaOneHotParams};
use atlas_trace::trace::LayerData;
use common::consts::{TRIG_DOWNSCALE_BITS, TRIG_PERIOD_MODULUS, XLEN};
use common::{CommittedPoly, VirtualPoly};
use jolt_claims::SymbolicSumcheck;
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
    /// An [`OpRelation::RangeCheck`].
    RangeCheck(OpRangeCheck),
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
                let prepared: Vec<Prepared<F, T>> = relations
                    .into_iter()
                    .map(|relation| prepare(relation, prover))
                    .collect();
                let witnesses = build_all(prepared, prover);
                // No instance, no proof (the verifier skips the batch the same way).
                if !witnesses.is_empty() {
                    proofs.push((ProofId(proof_idx, proof), prove_batch(witnesses, prover)));
                }
            }
            Step::Tie { relation } => prove_op_tie(&relation, prover),
            Step::Defer { batch, relation } => {
                let item = match batch.setup() {
                    DeferredSetup::InNodeLoop => {
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
        let prepared: Vec<Prepared<F, T>> = deferred
            .into_iter()
            .map(|item| match item {
                Deferred::Ready(witnesses) => Prepared::Ready(witnesses),
                Deferred::Pending(relation) => prepare(*relation, prover),
            })
            .collect();
        let witnesses = build_all(prepared, prover);
        proofs.push((
            ProofId(DEFERRED_PROOF_IDX, batch.proof_type()),
            prove_batch(witnesses, prover),
        ));
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
    OneHot(RaOneHotParams<F>, Vec<usize>),
    /// A clamp bucket's split (or, when exact, value) sumcheck parameters.
    ClampValue(ClampValueParams<F>, usize),
    /// A clamp bucket's digit range-check parameters.
    ClampDigits(DigitRangeParams<F>, usize),
    /// A saturated clamp bucket's slack chunk-check parameters.
    ClampSlack(BooleanitySumcheckParams<F>, usize),
    /// A remainder bucket's value sumcheck parameters.
    RemainderValue(ValueParams<F>, usize),
    /// A remainder bucket's chunk-check parameters.
    RemainderChunks(BooleanitySumcheckParams<F>, usize),
}

enum ClampValueParams<F: JoltField> {
    Split(SplitParams<F>),
    Exact(ValueParams<F>),
}

/// The sequential part of a batch relation's prover.
fn prepare<F: JoltField, T: Transcript>(
    relation: OpRelation,
    prover: &mut Prover<F, T>,
) -> Prepared<F, T> {
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
        Prepared::OneHot(params, prover.take_addresses(key, node_idx))
    }
    fn range_check_onehot<F: JoltField, T: Transcript, H: RangeCheckingOperandsTrait>(
        kind: OpRangeCheck,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Prepared<F, T> {
        let encoding = RangeCheckOperands::<H>::new(node)
            .expect("the prover's model has a range check for the node")
            .get_encoding(node);
        onehot(&encoding, Addresses::RangeCheck(kind), node.idx, prover)
    }
    match relation {
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
        OpRelation::LookupOneHot(kind @ OpLookup::ActivationClamp, node) => onehot(
            &OpLookupEncoding::<ActivationClampOperands>::new(&node),
            Addresses::Lookup(kind),
            node.idx,
            prover,
        ),
        OpRelation::LookupOneHot(kind @ OpLookup::TrigDownscale, node) => onehot(
            &OpLookupEncoding::<TrigDownscaleOperands>::new(&node),
            Addresses::Lookup(kind),
            node.idx,
            prover,
        ),
        OpRelation::RangeCheckOneHot(kind @ OpRangeCheck::Div, node) => {
            range_check_onehot::<F, T, DivRangeCheckOperands>(kind, &node, prover)
        }
        OpRelation::RangeCheckOneHot(kind @ OpRangeCheck::MeanOfSquares, node) => {
            range_check_onehot::<F, T, MeanOfSquaresRangeCheckOperands>(kind, &node, prover)
        }
        OpRelation::RangeCheckOneHot(kind @ OpRangeCheck::RsqrtDiv, node) => {
            range_check_onehot::<F, T, RiRangeCheckOperands>(kind, &node, prover)
        }
        OpRelation::RangeCheckOneHot(kind @ OpRangeCheck::RsqrtSqrt, node) => {
            range_check_onehot::<F, T, RsRangeCheckOperands>(kind, &node, prover)
        }
        OpRelation::RangeCheckOneHot(kind @ OpRangeCheck::Teleport, node) => {
            range_check_onehot::<F, T, TeleportRangeCheckOperands>(kind, &node, prover)
        }
        OpRelation::SmallActivationOneHot(node) => onehot(
            &SmallTableRaEncoding { node_idx: node.idx },
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
        OpRelation::ClampBucketValue(bucket) => {
            let params = if prover.clamp_split(bucket.idx).saturated() {
                ClampValueParams::Split(
                    SplitParams::draw(&bucket, &prover.accumulator, &mut prover.transcript)
                        .expect(expect),
                )
            } else {
                ClampValueParams::Exact(
                    clamp_split::exact_value_params(
                        &bucket,
                        &prover.accumulator,
                        &mut prover.transcript,
                    )
                    .expect(expect),
                )
            };
            Prepared::ClampValue(params, bucket.idx)
        }
        OpRelation::ClampBucketDigits(bucket) => Prepared::ClampDigits(
            clamp_split::digit_range_params(&bucket, &prover.accumulator, &mut prover.transcript)
                .expect(expect),
            bucket.idx,
        ),
        OpRelation::ClampBucketSlack(bucket) => {
            if !prover.clamp_split(bucket.idx).saturated() {
                // Exact bucket: no slack check (the verifier reads the exact flag).
                return Prepared::Ready(Vec::new());
            }
            Prepared::ClampSlack(
                clamp_split::chunk_check_params::<F, T>(
                    &bucket,
                    &prover.accumulator,
                    &mut prover.transcript,
                )
                .expect(expect),
                bucket.idx,
            )
        }
        OpRelation::RemainderBucketValue(bucket) => Prepared::RemainderValue(
            clamp_split::remainder_value_params(
                &bucket,
                &prover.accumulator,
                &mut prover.transcript,
            )
            .expect(expect),
            bucket.idx,
        ),
        OpRelation::RemainderBucketChunks(bucket) => Prepared::RemainderChunks(
            clamp_split::remainder_chunk_check_params::<F, T>(
                &bucket,
                &prover.accumulator,
                &mut prover.transcript,
            )
            .expect(expect),
            bucket.idx,
        ),
        other => Prepared::Ready(relation_witnesses(other, prover)),
    }
}

/// The heavy part of a batch's provers, in parallel across the batch.
fn build_all<F: JoltField, T: Transcript>(
    prepared: Vec<Prepared<F, T>>,
    prover: &Prover<F, T>,
) -> Vec<Box<dyn DynRelationWitness<F, T>>> {
    let _span = tracing::span!(tracing::Level::INFO, "deferred::build").entered();
    let buckets = &prover.buckets;
    prepared
        .into_par_iter()
        .flat_map_iter(|p| match p {
            Prepared::Ready(witnesses) => witnesses,
            Prepared::OneHot(params, indices) => {
                shout::prover::ra_onehot_provers_from_params::<F, T>(params, &indices).into()
            }
            Prepared::ClampValue(params, bucket) => {
                let split = buckets.clamp_split(bucket);
                vec![match params {
                    ClampValueParams::Split(p) => {
                        Box::new(SplitProver::new(p, split)) as Box<dyn DynRelationWitness<F, T>>
                    }
                    ClampValueParams::Exact(p) => Box::new(ValueProver::new(p, split.out_values()))
                        as Box<dyn DynRelationWitness<F, T>>,
                }]
            }
            Prepared::ClampDigits(params, bucket) => {
                clamp_split::digit_range_provers::<F, T>(params, buckets.clamp_split(bucket))
            }
            Prepared::ClampSlack(params, bucket) => {
                clamp_split::chunk_check_provers::<F, T>(params, buckets.clamp_split(bucket))
            }
            Prepared::RemainderValue(params, bucket) => {
                let (values, _) = buckets.remainder_values(bucket);
                vec![Box::new(ValueProver::new(params, values.to_vec()))
                    as Box<dyn DynRelationWitness<F, T>>]
            }
            Prepared::RemainderChunks(params, bucket) => {
                let (values, width) = buckets.remainder_values(bucket);
                clamp_split::remainder_chunk_check_provers::<F, T>(params, values, width)
            }
        })
        .collect()
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
    fn range_check<F: JoltField, T: Transcript, H: RangeCheckWitness>(
        kind: OpRangeCheck,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<Box<dyn DynRelationWitness<F, T>>> {
        let (witness, indices) = RangeCheckProvider::<H>::new(node)
            .expect("the prover's model has a range check for the node")
            .read_raf_prove::<F, T, UnsignedLessThanTable<XLEN>>(
                &prover.trace,
                &mut prover.accumulator,
                &mut prover.transcript,
            );
        prover.leave_addresses(Addresses::RangeCheck(kind), node.idx, indices);
        vec![Box::new(witness)]
    }
    fn small_activation<F: JoltField, T: Transcript, Table: SmallActivationTable>(
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
        OpRelation::Concat(relation) => engine(relation, prover),
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
        OpRelation::MeanOfSquares(relation) => {
            let params = RelationParams::new(relation, &prover.accumulator, &mut prover.transcript);
            vec![Box::new(MeanOfSquaresReductionProver::initialize(
                &prover.trace,
                params,
            ))]
        }
        OpRelation::Mul(relation) => engine(relation, prover),
        OpRelation::RangeCheck(kind @ OpRangeCheck::Div, node) => {
            range_check::<F, T, DivRangeCheckOperands>(kind, &node, prover)
        }
        OpRelation::RangeCheck(kind @ OpRangeCheck::MeanOfSquares, node) => {
            range_check::<F, T, MeanOfSquaresRangeCheckOperands>(kind, &node, prover)
        }
        OpRelation::RangeCheck(kind @ OpRangeCheck::RsqrtDiv, node) => {
            range_check::<F, T, RiRangeCheckOperands>(kind, &node, prover)
        }
        OpRelation::RangeCheck(kind @ OpRangeCheck::RsqrtSqrt, node) => {
            range_check::<F, T, RsRangeCheckOperands>(kind, &node, prover)
        }
        OpRelation::RangeCheck(kind @ OpRangeCheck::Teleport, node) => {
            range_check::<F, T, TeleportRangeCheckOperands>(kind, &node, prover)
        }
        OpRelation::Reshape(relation) => engine(relation, prover),
        OpRelation::Rsqrt(relation) => engine(relation, prover),
        OpRelation::Slice(relation) => engine(relation, prover),
        OpRelation::SmallActivation(OpActivation::Erf, node) => {
            small_activation::<F, T, ErfTableMarker>(&node, prover)
        }
        OpRelation::SmallActivation(OpActivation::Sigmoid, node) => {
            small_activation::<F, T, SigmoidTableMarker>(&node, prover)
        }
        OpRelation::SmallActivation(OpActivation::Tanh, node) => {
            small_activation::<F, T, TanhTableMarker>(&node, prover)
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
                SoftmaxRelation::Stage1(s)
                | SoftmaxRelation::Stage2(s)
                | SoftmaxRelation::Stage3(s)
                | SoftmaxRelation::ExpHiOneHot(s)
                | SoftmaxRelation::ExpLoOneHot(s)
                | SoftmaxRelation::ClampOneHot(s) => s.idx(),
            };
            let (witnesses, left) = softmax_last_axis::prove_relation(
                &relation,
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
        | OpRelation::RangeCheckOneHot(..)
        | OpRelation::SmallActivationOneHot(_)
        | OpRelation::TrigOneHot(..)
        | OpRelation::ClampBucketValue(_)
        | OpRelation::ClampBucketDigits(_)
        | OpRelation::ClampBucketSlack(_)
        | OpRelation::RemainderBucketValue(_)
        | OpRelation::RemainderBucketChunks(_)) => {
            let prepared = prepare(one_hot, prover);
            build_all(vec![prepared], prover)
        }
    }
}

/// A trig node's downscaled teleport remainder, `(input mod P) >> TRIG_DOWNSCALE_BITS`.
fn trig_downscaled(node: &ComputationNode, trace: &atlas_trace::Trace) -> atlas_trace::Tensor<i32> {
    let LayerData { operands, .. } = TraceView::layer_data(trace, node);
    let (_quotient, remainder) = compute_division(operands[0], TRIG_PERIOD_MODULUS as i32);
    remainder.map(|v| v >> TRIG_DOWNSCALE_BITS)
}

/// Append a plan opening's claims from the trace.
fn prove_open<F: JoltField, T: Transcript>(open: &OpOpen, prover: &mut Prover<F, T>) {
    let expect = "the prover holds every claim it reads";
    match open {
        OpOpen::ClampAcc(node) => {
            let witness = clamp_intermediate(node, &prover.trace);
            OpLookupProvider::with_helper(node.clone(), SaturatingAccClampOperands::for_node(node))
                .append_witness_claim_for(
                    &witness,
                    &mut prover.accumulator,
                    &mut prover.transcript,
                );
        }
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
            let LayerData { operands, .. } = TraceView::layer_data(&prover.trace, node);
            let (quotient, _remainder) = compute_division(operands[0], TRIG_PERIOD_MODULUS as i32);
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
        OpOpen::Softmax(open) => softmax_last_axis::prove_open(
            open,
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        ),
        OpOpen::ClampExactFlag(bucket_idx) => {
            let exact = !prover.clamp_split(*bucket_idx).saturated();
            prover.accumulator.append_virtual(
                &mut prover.transcript,
                clamp_split::exact_flag_id(*bucket_idx),
                OpeningPoint::new(Vec::<F>::new()),
                if exact { F::one() } else { F::zero() },
            );
        }
    }
}

/// Append a plan tie's produced openings with the prover's claims. An in-the-clear check
/// (a scalar clamp, an exact output, a bucket's exact flag) produces nothing here: its
/// openings are earlier steps'.
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
        OpTie::Constant(relation) => prove_tie(relation, trace, accumulator, transcript),
        OpTie::Input(relation) => prove_tie(relation, trace, accumulator, transcript),
        OpTie::DivQuotient(relation) => prove_tie(relation, trace, accumulator, transcript),
        OpTie::Identity(relation) => prove_tie(relation, trace, accumulator, transcript),
        OpTie::Neg(relation) => prove_tie(relation, trace, accumulator, transcript),
        OpTie::SubAccumulation(relation) => prove_tie(relation, trace, accumulator, transcript),
        OpTie::ZeroOutput(relation) => prove_tie(relation, trace, accumulator, transcript),
        OpTie::ScalarAccumulation { node, .. } => {
            let (opening_point, _claim) = AccOpeningAccessor::new(&*accumulator, node)
                .get_reduced_opening()
                .expect(expect);
            let LayerData { operands, .. } = TraceView::layer_data(trace, node);
            let [left, right] = operands[..] else {
                panic!("Expected two operands for {}", node.operator)
            };
            let left_claim = left.evaluate_mle(&opening_point.r);
            let right_claim = right.evaluate_mle(&opening_point.r);
            let mut provider =
                AccOpeningAccessor::new(accumulator, node).into_provider(transcript, opening_point);
            provider.append_nodeio(Target::Input(0), left_claim);
            provider.append_nodeio(Target::Input(1), right_claim);
        }
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
        OpTie::Softmax(tie) => softmax_last_axis::prove_tie(tie),
        OpTie::ScalarClamp(_)
        | OpTie::ScalarRemainder { .. }
        | OpTie::ExactClampOutput(_)
        | OpTie::ClampBucketExact(_) => {}
    }
}
