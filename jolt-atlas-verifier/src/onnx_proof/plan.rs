//! Verification plans: what the verifier checks for an op, in order, as data.
//!
//! An op's [`OpPlan::plan`] returns its [`Plan`], an ordered list of [`Step`]s over relations,
//! built from the model and the public IO before the transcript runs. The interpreter
//! ([`Verifier::run_plan`](super::Verifier::run_plan)) executes the steps; `jolt-atlas-core`'s
//! prover walks the same plan with the relations' witnesses. The transcript order of an op is
//! therefore its plan's step order on both sides.
//!
//! A step holds relations, never closures. A relation whose parameters depend on earlier claims
//! or on challenges reads and draws them in its
//! [`SetupRelation::setup`](super::relation::SetupRelation::setup), which runs when the
//! interpreter reaches the step. The post-loop work (the packed clamp and remainder buckets,
//! the deferred batches) is a plan too: [`deferred_lookups::plan`](super::deferred_lookups::plan)
//! builds it from the model and the public IO once the node loop is done.

use crate::node_view::ComputationNode;
use crate::onnx_proof::{
    deferred_lookups::DeferredBatch,
    fused_rebase,
    lookup::SmallTableRaEncoding,
    op_lookups::{DefaultLookupOperands, OpLookupEncoding, OpLookupProvider},
    ops::{
        activation_clamped::ActivationClampOperands,
        clamp::SymmetricClampOperands,
        gather::{self, small::GatherSmallShape},
        smooth_relu::SmoothReluOperands,
        softmax_last_axis::{SoftmaxOpen, SoftmaxRelation},
        trig::TrigDownscaleOperands,
    },
    range_bucket::RangeBucket,
    range_check::{self, ValueParams},
    relation::{op_rel_verifier, OpRel},
    ProofType,
};
use crate::utils::opening_access::AccOpeningAccessor;
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_lookup_tables::{
    clamp::{ActivationClampTable, ClampTable, SiluClampTable},
    relu::ReluTable,
    right_shift::RightShiftTable,
};
use atlas_sumcheck::{
    opening_proof::{PolynomialId, VerifierOpeningAccumulator},
    relation::DynRelationVerifier,
    shout,
};
use atlas_trace::{trace::ModelExecutionIO, Model};
use atlas_transcript::Transcript;
use common::{consts::XLEN, CommittedPoly, VirtualPoly};

/// One thing the verifier does for a node.
pub enum Step<F: JoltField> {
    /// Reduce the claims on the node's output to one (the node's evaluation reduction).
    Reduce,
    /// Append advice openings at a stated point; their claims come from the proof.
    Open(OpOpen),
    /// Verify `relation`'s sumcheck against the node's proof stored under `proof`.
    Relation {
        /// The proof slot of the sumcheck.
        proof: ProofType,
        /// The relation proven.
        relation: OpRelation<F>,
    },
    /// Verify several relations as one batched sumcheck stored under `proof`, their setups run
    /// in list order.
    Batch {
        /// The proof slot of the batched sumcheck.
        proof: ProofType,
        /// The relations of the batch.
        relations: Vec<OpRelation<F>>,
    },
    /// Check a zero-round relation: append its openings and require the identity to hold.
    Tie {
        /// The relation checked.
        relation: OpRel<F>,
    },
    /// Queue `relation` for the post-loop batch `batch`, proven as one batched sumcheck with
    /// every other relation queued there. The relation is set up when the batch says
    /// ([`DeferredBatch::setup`]): in the node's turn, or when the batch runs.
    Defer {
        /// The batch the relation joins.
        batch: DeferredBatch,
        /// The relation queued.
        relation: OpRelation<F>,
    },
}

/// A node's verification, in transcript order.
pub struct Plan<F: JoltField>(pub Vec<Step<F>>);

/// An op's verification plan for one node.
pub trait OpPlan {
    /// The steps that verify `node`, from the model and the public IO alone.
    fn plan<F: JoltField>(
        &self,
        node: &ComputationNode,
        model: &Model,
        io: &ModelExecutionIO,
    ) -> Result<Plan<F>, ProofVerifyError>;
}

/// A relation proven by a sumcheck, as a plan step holds it.
pub enum OpRelation<F: JoltField> {
    /// An op relation.
    Op(OpRel<F>),
    /// `GatherLarge`: the one-hot checks of its read addresses (two relations).
    GatherLargeOneHot(ComputationNode),
    /// `GatherSmall`: the Hamming-Booleanity and Booleanity checks of its one-hot `ra`.
    GatherSmallOneHot(GatherSmallShape),
    /// `GatherSmall`: the Hamming-weight check of its one-hot `ra`.
    GatherSmallHammingWeight(GatherSmallShape),
    /// A prefix-suffix table lookup of the node's operand: the operand's opening is appended,
    /// then the read-raf relation is set up.
    Lookup(OpLookup, ComputationNode),
    /// The one-hot checks of a [`Self::Lookup`]'s read addresses: two relations (the address
    /// polynomial as the product of its chunks; the chunks' Booleanity and Hamming weight).
    LookupOneHot(OpLookup, ComputationNode),
    /// The one-hot checks of a small table's read addresses (two relations).
    SmallTableOneHot(ComputationNode),
    /// A softmax stage, as `ops::softmax_last_axis` lists them.
    Softmax(SoftmaxRelation),
    /// A packed range bucket's value relation: the range claims of its members
    /// ([`range_bucket::RangeClaim`](crate::onnx_proof::range_bucket::RangeClaim)).
    RemainderBucketValue(RangeBucket),
    /// A packed range bucket's digit range check.
    RemainderBucketDigits(RangeBucket),
}

/// The table a lookup op reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum OpLookup {
    /// `ReLU`'s table.
    Relu,
    /// `Clamp`'s symmetric clamp table.
    SymmetricClamp,
    /// A clamped activation's input clamp (`clamped = ActivationClampTable[input]`).
    ActivationClamp,
    /// A trig op's downscale (right shift) of its remainder.
    TrigDownscale,
    /// SiLU's input clamp, twice as wide as the clamped activations'.
    SiluClamp,
    /// A smooth ReLU's `ReLU` read, through its clamp lookup's committed address.
    SmoothRelu,
}

impl<F: JoltField> OpRelation<F> {
    /// The relation's verifiers (one, or the several of a stage or a one-hot check), set up
    /// against the accumulator and transcript.
    pub fn verifiers<T: Transcript>(
        self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<Vec<Box<dyn DynRelationVerifier<F, T>>>, ProofVerifyError> {
        fn boxed<F: JoltField, T: Transcript>(
            verifier: impl DynRelationVerifier<F, T> + 'static,
        ) -> Vec<Box<dyn DynRelationVerifier<F, T>>> {
            vec![Box::new(verifier)]
        }
        match self {
            Self::Op(relation) => Ok(boxed(op_rel_verifier(relation, accumulator, transcript)?)),
            Self::GatherLargeOneHot(node) => Ok(shout::ra_onehot_verifiers(
                &gather::large::GatherRaEncoding::new(&node)?,
                accumulator,
                transcript,
            )?
            .into()),
            Self::GatherSmallOneHot(shape) => {
                let (hamming_booleanity, booleanity) =
                    gather::small::build_stage2_verifiers(&shape, accumulator, transcript)?;
                Ok(vec![Box::new(hamming_booleanity), Box::new(booleanity)])
            }
            Self::GatherSmallHammingWeight(shape) => Ok(boxed(
                gather::small::build_stage3_verifier(&shape, accumulator, transcript)?,
            )),
            Self::Lookup(OpLookup::Relu, node) => Ok(boxed(
                OpLookupProvider::<DefaultLookupOperands>::new(node)
                    .read_raf_verify::<F, T, ReluTable<XLEN>, XLEN>(accumulator, transcript)?,
            )),
            Self::Lookup(OpLookup::SymmetricClamp, node) => Ok(boxed(
                OpLookupProvider::<SymmetricClampOperands>::new(node)
                    .read_raf_verify::<F, T, ClampTable<XLEN>, XLEN>(accumulator, transcript)?,
            )),
            Self::Lookup(OpLookup::ActivationClamp, node) => Ok(boxed(
                OpLookupProvider::<ActivationClampOperands>::new(node)
                    .read_raf_verify::<F, T, ActivationClampTable<XLEN>, XLEN>(
                        accumulator,
                        transcript,
                    )?,
            )),
            Self::Lookup(OpLookup::TrigDownscale, node) => Ok(boxed(
                OpLookupProvider::<TrigDownscaleOperands>::new(node)
                    .read_raf_verify::<F, T, RightShiftTable<XLEN>, XLEN>(
                        accumulator,
                        transcript,
                    )?,
            )),
            Self::Lookup(OpLookup::SiluClamp, node) => Ok(boxed(
                OpLookupProvider::<ActivationClampOperands>::new(node)
                    .read_raf_verify::<F, T, SiluClampTable<XLEN>, XLEN>(accumulator, transcript)?,
            )),
            Self::Lookup(OpLookup::SmoothRelu, node) => Ok(boxed(
                OpLookupProvider::<SmoothReluOperands>::new(node)
                    .read_raf_verify::<F, T, ReluTable<XLEN>, XLEN>(accumulator, transcript)?,
            )),
            Self::LookupOneHot(OpLookup::Relu, node) => Ok(shout::ra_onehot_verifiers(
                &OpLookupEncoding::<DefaultLookupOperands>::new(&node),
                accumulator,
                transcript,
            )?
            .into()),
            Self::LookupOneHot(OpLookup::SymmetricClamp, node) => Ok(shout::ra_onehot_verifiers(
                &OpLookupEncoding::<SymmetricClampOperands>::new(&node),
                accumulator,
                transcript,
            )?
            .into()),
            // One committed address serves a smooth ReLU's clamp and ReLU reads; its checks
            // are the clamp's.
            Self::LookupOneHot(
                OpLookup::ActivationClamp | OpLookup::SiluClamp | OpLookup::SmoothRelu,
                node,
            ) => Ok(shout::ra_onehot_verifiers(
                &OpLookupEncoding::<ActivationClampOperands>::new(&node),
                accumulator,
                transcript,
            )?
            .into()),
            Self::LookupOneHot(OpLookup::TrigDownscale, node) => Ok(shout::ra_onehot_verifiers(
                &OpLookupEncoding::<TrigDownscaleOperands>::new(&node),
                accumulator,
                transcript,
            )?
            .into()),
            Self::SmallTableOneHot(node) => Ok(shout::ra_onehot_verifiers(
                &SmallTableRaEncoding::for_node(&node)?,
                accumulator,
                transcript,
            )?
            .into()),
            Self::Softmax(relation) => relation.verifiers(accumulator, transcript),
            Self::RemainderBucketValue(bucket) => Ok(boxed(ValueParams::into_verifier(
                range_check::remainder_value_params(&bucket, accumulator, transcript)?,
            ))),
            Self::RemainderBucketDigits(bucket) => Ok(boxed(
                range_check::remainder_digit_range_params(&bucket, accumulator, transcript)?
                    .into_verifier(),
            )),
        }
    }
}

/// Advice openings a plan step appends; the point is stated, the claim is the proof's.
pub enum OpOpen {
    /// A fused node's rescale remainder at the reduced output point.
    RescaleRemainder(ComputationNode),
    /// A trig op's period quotient at the reduced output point.
    TrigPeriodQuotient(ComputationNode),
    /// A trig op's downscaled remainder at the reduced output point.
    TrigDownscaled(ComputationNode),
    /// A smooth ReLU's `ReLU(x)` at the reduced output point.
    SmoothReluRelu(ComputationNode),
    /// A softmax opening, as `ops::softmax_last_axis` lists them.
    Softmax(SoftmaxOpen),
}

impl OpOpen {
    /// Append the openings.
    pub fn verify<F: JoltField, T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<(), ProofVerifyError> {
        /// Append the node's advice `poly` at its reduced output point.
        fn at_reduced_output<F: JoltField, T: Transcript, Poly: Into<PolynomialId>>(
            node: &ComputationNode,
            accumulator: &mut VerifierOpeningAccumulator<F>,
            transcript: &mut T,
            poly: fn(usize) -> Poly,
        ) -> Result<(), ProofVerifyError> {
            let accessor = AccOpeningAccessor::new(accumulator, node);
            let r0 = accessor.get_reduced_opening()?.0;
            accessor.into_provider(transcript, r0).append_advice(poly)
        }
        match self {
            Self::RescaleRemainder(node) => {
                at_reduced_output(node, accumulator, transcript, VirtualPoly::RescaleRemainder)
            }
            Self::TrigPeriodQuotient(node) => at_reduced_output(
                node,
                accumulator,
                transcript,
                CommittedPoly::TrigPeriodQuotient,
            ),
            Self::TrigDownscaled(node) => {
                at_reduced_output(node, accumulator, transcript, VirtualPoly::TrigDownscaled)
            }
            Self::SmoothReluRelu(node) => {
                at_reduced_output(node, accumulator, transcript, VirtualPoly::SmoothReluRelu)
            }
            Self::Softmax(open) => open.verify(accumulator, transcript),
        }
    }
}

/// The fused-rescale skeleton around an op's arithmetic relation (`Einsum`, `Mul`, `Square`,
/// `Cube`): the remainder advice, then the arithmetic sumcheck under `proof`, whose input claim
/// is `out·2^S + rem`. The remainder, and the output unless the node is a model output, are
/// range-checked in the packed buckets after the node loop. A node without a rescale
/// (`bits == 0`) proves the arithmetic relation alone against its reduced output.
pub fn fused_rescale_plan<F: JoltField>(
    node: &ComputationNode,
    proof: ProofType,
    relation: OpRelation<F>,
) -> Plan<F> {
    let mut steps = vec![Step::Reduce];
    if fused_rebase::rebase_bits(node).is_some_and(|bits| bits > 0) {
        steps.push(Step::Open(OpOpen::RescaleRemainder(node.clone())));
    }
    steps.push(Step::Relation { proof, relation });
    Plan(steps)
}

/// Bind a plan-verified op with no committed polynomials: `$relation` (a relation of the plan)
/// is checked to be a relation of that op, and its [`OpLayout`](super::ops::OpLayout) is the
/// default. An op that commits polynomials writes its own `OpLayout` and uses `bind_relation!`.
#[macro_export]
macro_rules! impl_plan_op {
    ($inner_ty:ty, $relation:ty) => {
        $crate::bind_relation!($inner_ty, $relation);

        impl $crate::onnx_proof::ops::OpLayout for $inner_ty {}
    };
}
