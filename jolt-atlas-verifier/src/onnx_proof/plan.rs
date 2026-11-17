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
    clamp_split::{self, ValueParams},
    deferred_lookups::DeferredBatch,
    fused_rebase,
    global_clamp::ClampBucket,
    neural_teleport::{
        cos::CosTable, division::TeleportInputRelation, sin::SinTable,
        small_table::SmallTableRelation, trig_downscale::TrigDownscaleOperands,
    },
    op_lookups::{DefaultLookupOperands, OpLookupEncoding, OpLookupProvider},
    ops::{
        activation_clamped::{small_table_verifier, ActivationClampOperands, SmallTableRaEncoding},
        broadcast::BroadcastRelation,
        clamp::SymmetricClampOperands,
        concat::SubcubeTilingRelation,
        constant::PublicRelation,
        cos::CosRaEncoding,
        div::{DivQuotientRelation, DivRelation},
        einsum::ContractionRelation,
        erf::ErfTableMarker,
        gather::{self, small::GatherSmallShape, GatherRelation},
        is_nan::ZeroOutputRelation,
        mean_of_squares::MeanOfSquaresRelation,
        moveaxis::AxisPermutationRelation,
        neg::SignedCopyRelation,
        reshape::FlatCopyRelation,
        rsqrt::RsqrtRelation,
        scalar_const_div::ConstantDivisionRelation,
        sigmoid::SigmoidTableMarker,
        sin::SinRaEncoding,
        slice::SubcubeSelectRelation,
        smooth_relu::{GeluErfTable, GeluTanhTable, SiluTable, SmoothReluOperands},
        softmax_last_axis::{SoftmaxOpen, SoftmaxRelation, SoftmaxTie},
        sum::axis::SumAxisRelation,
        tanh::TanhTableMarker,
    },
    relation::{verify_tie, OpRel, RelationVerifier, SetupRelation},
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
use atlas_trace::{
    op::{Constant, Identity, Input, Neg},
    trace::ModelExecutionIO,
    Model,
};
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
        relation: OpTie<F>,
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

/// The small activation table a clamped activation reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpActivation {
    /// `Erf`.
    Erf,
    /// `Sigmoid`.
    Sigmoid,
    /// `Tanh`.
    Tanh,
    /// The erf-form GELU's correction to `ReLU`.
    GeluErf,
    /// The tanh-form GELU's correction to `ReLU`.
    GeluTanh,
    /// SiLU's correction to `ReLU`.
    Silu,
}

/// The trigonometric table a teleported trig op reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpTrig {
    /// `Cos`.
    Cos,
    /// `Sin`.
    Sin,
}

/// A relation proven by a sumcheck, as a plan step holds it.
pub enum OpRelation<F: JoltField> {
    /// An op relation.
    Op(OpRel<F>),
    /// `ScalarConstDiv`: the quotient by a constant.
    ConstantDivision(ConstantDivisionRelation),
    /// `Einsum`: the contraction, its input claim the fused rescale's `q·2^S + rem`.
    Contraction(ContractionRelation),
    /// `Div`: the floor division identity, at a point drawn from the transcript.
    Div(DivRelation),
    /// `GatherSmall` / `GatherLarge`: the dictionary read; the index operand's opening is
    /// appended at the lookup variables of the setup's point.
    Gather(GatherRelation),
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
    /// `MeanOfSquares`: the squared-sum reduction, its input claim `q·D + rem`.
    MeanOfSquares(MeanOfSquaresRelation),
    /// `Rsqrt`: the fused division and square-root identities.
    Rsqrt(RsqrtRelation),
    /// A clamped activation's small-table execution: `γ` drawn, the clamped input's opening
    /// appended at the reduced output point.
    SmallActivation(OpActivation, ComputationNode),
    /// The one-hot checks of a [`Self::SmallActivation`]'s read addresses (two relations).
    SmallActivationOneHot(ComputationNode),
    /// A softmax stage, as `ops::softmax_last_axis` lists them.
    Softmax(SoftmaxRelation),
    /// `Sum`: the axis reduction of the output.
    SumAxis(SumAxisRelation),
    /// A trig op's table execution over the downscaled input.
    Trig(OpTrig, ComputationNode),
    /// The one-hot checks of a [`Self::Trig`]'s read addresses (two relations).
    TrigOneHot(OpTrig, ComputationNode),
    /// A packed range bucket's value relation: the range claims of its members
    /// ([`global_clamp::RangeClaim`](crate::onnx_proof::global_clamp::RangeClaim)).
    RemainderBucketValue(ClampBucket),
    /// A packed range bucket's digit range check.
    RemainderBucketDigits(ClampBucket),
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
        fn one<F: JoltField, T: Transcript, R: SetupRelation<F> + 'static>(
            relation: R,
            accumulator: &VerifierOpeningAccumulator<F>,
            transcript: &mut T,
        ) -> Result<Vec<Box<dyn DynRelationVerifier<F, T>>>, ProofVerifyError> {
            Ok(vec![Box::new(RelationVerifier::new(
                relation,
                accumulator,
                transcript,
            )?)])
        }
        fn boxed<F: JoltField, T: Transcript>(
            verifier: impl DynRelationVerifier<F, T> + 'static,
        ) -> Vec<Box<dyn DynRelationVerifier<F, T>>> {
            vec![Box::new(verifier)]
        }
        match self {
            Self::Op(relation) => one(relation, accumulator, transcript),
            Self::ConstantDivision(relation) => one(relation, accumulator, transcript),
            Self::Contraction(relation) => one(relation, accumulator, transcript),
            Self::Div(relation) => one(relation, accumulator, transcript),
            Self::Gather(relation) => Ok(boxed(gather::gather_verifier(
                relation,
                accumulator,
                transcript,
            )?)),
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
            Self::MeanOfSquares(relation) => one(relation, accumulator, transcript),
            Self::Rsqrt(relation) => one(relation, accumulator, transcript),
            Self::SmallActivation(OpActivation::Erf, node) => {
                Ok(boxed(small_table_verifier::<F, ErfTableMarker>(
                    &node,
                    accumulator,
                    transcript,
                )?))
            }
            Self::SmallActivation(OpActivation::Sigmoid, node) => Ok(boxed(
                small_table_verifier::<F, SigmoidTableMarker>(&node, accumulator, transcript)?,
            )),
            Self::SmallActivation(OpActivation::Tanh, node) => {
                Ok(boxed(small_table_verifier::<F, TanhTableMarker>(
                    &node,
                    accumulator,
                    transcript,
                )?))
            }
            Self::SmallActivation(OpActivation::GeluErf, node) => Ok(boxed(
                small_table_verifier::<F, GeluErfTable>(&node, accumulator, transcript)?,
            )),
            Self::SmallActivation(OpActivation::GeluTanh, node) => Ok(boxed(
                small_table_verifier::<F, GeluTanhTable>(&node, accumulator, transcript)?,
            )),
            Self::SmallActivation(OpActivation::Silu, node) => {
                Ok(boxed(small_table_verifier::<F, SiluTable>(
                    &node,
                    accumulator,
                    transcript,
                )?))
            }
            Self::SmallActivationOneHot(node) => Ok(shout::ra_onehot_verifiers(
                &SmallTableRaEncoding::for_node(&node),
                accumulator,
                transcript,
            )?
            .into()),
            Self::Softmax(relation) => relation.verifiers(accumulator, transcript),
            Self::SumAxis(relation) => one(relation, accumulator, transcript),
            Self::Trig(OpTrig::Cos, node) => one(
                SmallTableRelation::<CosTable>::new(node),
                accumulator,
                transcript,
            ),
            Self::Trig(OpTrig::Sin, node) => one(
                SmallTableRelation::<SinTable>::new(node),
                accumulator,
                transcript,
            ),
            Self::TrigOneHot(OpTrig::Cos, node) => Ok(shout::ra_onehot_verifiers(
                &CosRaEncoding::new(&node),
                accumulator,
                transcript,
            )?
            .into()),
            Self::TrigOneHot(OpTrig::Sin, node) => Ok(shout::ra_onehot_verifiers(
                &SinRaEncoding::new(&node),
                accumulator,
                transcript,
            )?
            .into()),
            Self::RemainderBucketValue(bucket) => Ok(boxed(ValueParams::into_verifier(
                clamp_split::remainder_value_params(&bucket, accumulator, transcript)?,
            ))),
            Self::RemainderBucketDigits(bucket) => Ok(boxed(
                clamp_split::remainder_digit_range_params(&bucket, accumulator, transcript)?
                    .into_verifier(),
            )),
        }
    }
}

/// A zero-round relation or an in-the-clear check, as a plan step holds it.
pub enum OpTie<F: JoltField> {
    /// An op relation with no rounds.
    Op(OpRel<F>),
    /// `MoveAxis`: the operand at the permuted point.
    AxisPermutation(AxisPermutationRelation),
    /// `Broadcast`: the operand replicated along the broadcast axes.
    Broadcast(BroadcastRelation),
    /// `Concat`: the operands' blocks tile the output's axis.
    Concat(SubcubeTilingRelation),
    /// `Constant`: the output is the model's constant.
    Constant(PublicRelation<Constant>),
    /// `Input`: the output is the public input.
    Input(PublicRelation<Input>),
    /// `Div`: the committed quotient is the node's reduced output.
    DivQuotient(DivQuotientRelation),
    /// `Identity`: the operand is the output.
    Identity(SignedCopyRelation<Identity>),
    /// `Neg`: the operand is the output, negated.
    Neg(SignedCopyRelation<Neg>),
    /// `Reshape`: the operand at the same point.
    Reshape(FlatCopyRelation),
    /// `Slice`: the operand's subcube the slice selects.
    Slice(SubcubeSelectRelation),
    /// `IsNan`: the output is zero.
    ZeroOutput(ZeroOutputRelation),
    /// A trig op's input is `P·q + r` over its teleport quotient and remainder claims.
    TeleportInput(TeleportInputRelation),
    /// A softmax in-the-clear check, as `ops::softmax_last_axis` lists them.
    Softmax(SoftmaxTie),
}

impl<F: JoltField> OpTie<F> {
    /// Append the tie's produced openings and check its identity.
    pub fn verify<T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<(), ProofVerifyError> {
        match self {
            Self::Op(relation) => verify_tie(relation, accumulator, transcript),
            Self::AxisPermutation(relation) => verify_tie(relation, accumulator, transcript),
            Self::Broadcast(relation) => verify_tie(relation, accumulator, transcript),
            Self::Concat(relation) => verify_tie(relation, accumulator, transcript),
            Self::Constant(relation) => verify_tie(relation, accumulator, transcript),
            Self::Input(relation) => verify_tie(relation, accumulator, transcript),
            Self::DivQuotient(relation) => verify_tie(relation, accumulator, transcript),
            Self::Identity(relation) => verify_tie(relation, accumulator, transcript),
            Self::Neg(relation) => verify_tie(relation, accumulator, transcript),
            Self::Reshape(relation) => verify_tie(relation, accumulator, transcript),
            Self::Slice(relation) => verify_tie(relation, accumulator, transcript),
            Self::ZeroOutput(relation) => verify_tie(relation, accumulator, transcript),
            Self::TeleportInput(relation) => relation.verify(accumulator, transcript),
            Self::Softmax(tie) => tie.verify(accumulator, transcript),
        }
    }
}

/// Advice openings a plan step appends; the point is stated, the claim is the proof's.
pub enum OpOpen {
    /// A fused node's rescale remainder at the reduced output point.
    RescaleRemainder(ComputationNode),
    /// A trig op's teleport quotient at the reduced output point.
    TeleportQuotient(ComputationNode),
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
            Self::TeleportQuotient(node) => at_reduced_output(
                node,
                accumulator,
                transcript,
                CommittedPoly::TeleportNodeQuotient,
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
