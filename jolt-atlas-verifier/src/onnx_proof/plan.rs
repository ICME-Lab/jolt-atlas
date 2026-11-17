//! Verification plans: what the verifier checks for an op, in order, as data.
//!
//! An op's [`OpPlan::plan`] returns its [`Plan`], an ordered list of [`Step`]s over relations,
//! built from the model and the public IO before the transcript runs. The interpreter
//! ([`Verifier::run_plan`](super::Verifier::run_plan)) executes the steps; `jolt-atlas-core`'s
//! prover walks the same plan with the relations' witnesses. The transcript order of an op is
//! therefore its plan's step order on both sides.
//!
//! A step holds values, never closures. A sumcheck step holds an [`OpRelation`], one variant
//! per kind of verifier construction: an op relation ([`OpRel`]), a prefix-suffix table read
//! ([`TableRead`]), the one-hot checks of a read's addresses ([`RaEncoding`]), and the stages
//! with their own shape. A tie step holds a zero-round [`OpRel`], checked by
//! [`verify_tie`](super::relation::verify_tie); an open step an [`OpOpen`]. A relation whose
//! parameters depend on earlier claims or on challenges reads and draws them in its
//! [`SetupRelation::setup`](super::relation::SetupRelation::setup), which runs when the
//! interpreter reaches the step. The post-loop work (the packed range buckets, the deferred
//! batches) is a plan too: [`deferred_lookups::plan`](super::deferred_lookups::plan)
//! builds it from the model and the public IO once the node loop is done.

use crate::node_view::ComputationNode;
use crate::onnx_proof::{
    deferred_lookups::DeferredBatch,
    fused_rebase,
    lookup::{table_read_verifier, RaEncoding, TableRead},
    ops::{
        gather::{self, small::GatherSmallShape},
        softmax_last_axis::{self, SoftmaxOpen, SoftmaxShape},
    },
    range_bucket::RangeBucket,
    range_check::{self, ValueParams},
    relation::{op_rel_verifier, OpRel},
    ProofType,
};
use crate::utils::opening_access::AccOpeningAccessor;
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_sumcheck::{
    opening_proof::{PolynomialId, VerifierOpeningAccumulator},
    relation::DynRelationVerifier,
    shout,
};
use atlas_trace::{trace::ModelExecutionIO, Model};
use atlas_transcript::Transcript;
use common::{CommittedPoly, VirtualPoly};

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

/// A relation proven by a sumcheck, as a plan step holds it: one variant per kind of verifier
/// construction.
pub enum OpRelation<F: JoltField> {
    /// An op relation.
    Op(OpRel<F>),
    /// A prefix-suffix table read: the operand's opening is appended at the read's cycle
    /// point, then the read-raf relation is set up.
    TableRead(TableRead),
    /// The one-hot checks of a read's addresses: two relations (the address polynomial as the
    /// product of its chunks; the chunks' Booleanity and Hamming weight).
    OneHot(RaEncoding),
    /// `GatherSmall`: the Hamming-Booleanity and Booleanity checks of its one-hot `ra`.
    GatherSmallOneHot(GatherSmallShape),
    /// `GatherSmall`: the Hamming-weight check of its one-hot `ra`.
    GatherSmallHammingWeight(GatherSmallShape),
    /// `SoftmaxLastAxis`: the read of the output, sum and shifted sum tables at the reads'
    /// point.
    SoftmaxRead(SoftmaxShape),
    /// A packed range bucket's value relation: the range claims of its members
    /// ([`range_bucket::RangeClaim`](crate::onnx_proof::range_bucket::RangeClaim)).
    RemainderBucketValue(RangeBucket),
    /// A packed range bucket's digit range check.
    RemainderBucketDigits(RangeBucket),
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
            Self::TableRead(read) => Ok(vec![table_read_verifier(&read, accumulator, transcript)?]),
            Self::OneHot(encoding) => {
                Ok(shout::ra_onehot_verifiers(&encoding, accumulator, transcript)?.into())
            }
            Self::GatherSmallOneHot(shape) => {
                let (hamming_booleanity, booleanity) =
                    gather::small::build_stage2_verifiers(&shape, accumulator, transcript)?;
                Ok(vec![Box::new(hamming_booleanity), Box::new(booleanity)])
            }
            Self::GatherSmallHammingWeight(shape) => Ok(boxed(
                gather::small::build_stage3_verifier(&shape, accumulator, transcript)?,
            )),
            Self::SoftmaxRead(shape) => Ok(vec![softmax_last_axis::read_verifier(
                shape,
                accumulator,
                transcript,
            )?]),
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
