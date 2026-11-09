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
    clamp_lookups::{
        acc_opening_id, recover_small_int, verify_scalar_clamp, SaturatingAccClampOperands,
    },
    clamp_split::{self, SplitParams, ValueParams},
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
        add::AccumulationRelation,
        and::AndRelation,
        broadcast::BroadcastRelation,
        clamp::SymmetricClampOperands,
        concat::ConcatRelation,
        constant::PublicRelation,
        cos::CosRaEncoding,
        cube::CubeRelation,
        div::{DivQuotientRelation, DivRelation},
        einsum::ContractionRelation,
        erf::ErfTableMarker,
        gather::{self, small::GatherSmallShape, GatherRelation},
        iff::IffRelation,
        is_nan::ZeroOutputRelation,
        mean_of_squares::MeanOfSquaresRelation,
        moveaxis::AxisPermutationRelation,
        mul::MulRelation,
        neg::SignedCopyRelation,
        reshape::ReshapeRelation,
        rsqrt::RsqrtRelation,
        scalar_const_div::ConstantDivisionRelation,
        sigmoid::SigmoidTableMarker,
        sin::SinRaEncoding,
        slice::SliceRelation,
        softmax_last_axis::{SoftmaxOpen, SoftmaxRelation, SoftmaxTie},
        square::SquareRelation,
        sum::axis::SumAxisRelation,
        tanh::TanhTableMarker,
    },
    range_checking::{
        range_check_operands::{
            DivRangeCheckOperands, MeanOfSquaresRangeCheckOperands, RangeCheckOperands,
            RangeCheckingOperandsTrait, RiRangeCheckOperands, RsRangeCheckOperands,
            TeleportRangeCheckOperands,
        },
        RangeCheckProvider,
    },
    relation::{verify_tie, RelationVerifier, SetupRelation},
    ProofType,
};
use crate::utils::opening_access::{AccOpeningAccessor, Target};
use atlas_algebra::{
    field::JoltField, poly::opening_point::OpeningPoint, utils::errors::ProofVerifyError,
};
use atlas_lookup_tables::{
    clamp::{ActivationClampTable, ClampTable},
    relu::ReluTable,
    right_shift::RightShiftTable,
    unsigned_less_than::UnsignedLessThanTable,
};
use atlas_sumcheck::{
    opening_proof::{OpeningAccumulator, PolynomialId, VerifierOpeningAccumulator},
    relation::DynRelationVerifier,
    shout,
};
use atlas_trace::{
    op::{Add, Constant, Identity, Input, Neg, Sub},
    trace::ModelExecutionIO,
    Model,
};
use atlas_transcript::Transcript;
use common::{consts::XLEN, CommittedPoly, VirtualPoly};
use jolt_claims::SymbolicSumcheck;

/// One thing the verifier does for a node.
pub enum Step {
    /// Reduce the claims on the node's output to one (the node's evaluation reduction).
    Reduce,
    /// Append advice openings at a stated point; their claims come from the proof.
    Open(OpOpen),
    /// Verify `relation`'s sumcheck against the node's proof stored under `proof`.
    Relation {
        /// The proof slot of the sumcheck.
        proof: ProofType,
        /// The relation proven.
        relation: OpRelation,
    },
    /// Verify several relations as one batched sumcheck stored under `proof`, their setups run
    /// in list order.
    Batch {
        /// The proof slot of the batched sumcheck.
        proof: ProofType,
        /// The relations of the batch.
        relations: Vec<OpRelation>,
    },
    /// Check a zero-round relation: append its openings and require the identity to hold.
    Tie {
        /// The relation checked.
        relation: OpTie,
    },
    /// Queue `relation` for the post-loop batch `batch`, proven as one batched sumcheck with
    /// every other relation queued there. The relation is set up when the batch says
    /// ([`DeferredBatch::setup`]): in the node's turn, or when the batch runs.
    Defer {
        /// The batch the relation joins.
        batch: DeferredBatch,
        /// The relation queued.
        relation: OpRelation,
    },
}

/// A node's verification, in transcript order.
pub struct Plan(pub Vec<Step>);

/// An op's verification plan for one node.
pub trait OpPlan {
    /// The steps that verify `node`, from the model and the public IO alone.
    fn plan(
        &self,
        node: &ComputationNode,
        model: &Model,
        io: &ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError>;
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
pub enum OpRelation {
    /// `And`: the product of two booleans.
    And(AndRelation),
    /// `ScalarConstDiv`: the quotient by a constant.
    ConstantDivision(ConstantDivisionRelation),
    /// `Concat`: each operand selected into its slot.
    Concat(ConcatRelation),
    /// `Einsum`: the contraction, its input claim the fused rescale's `q·2^S + rem`.
    Contraction(ContractionRelation),
    /// `Cube`: `x³`.
    Cube(CubeRelation),
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
    /// `Iff`: the masked select.
    Iff(IffRelation),
    /// A prefix-suffix table lookup of the node's operand: the operand's opening is appended,
    /// then the read-raf relation is set up.
    Lookup(OpLookup, ComputationNode),
    /// The one-hot checks of a [`Self::Lookup`]'s read addresses: two relations (the address
    /// polynomial as the product of its chunks; the chunks' Booleanity and Hamming weight).
    LookupOneHot(OpLookup, ComputationNode),
    /// `MeanOfSquares`: the squared-sum reduction, its input claim `q·D + rem`.
    MeanOfSquares(MeanOfSquaresRelation),
    /// `Mul`: `left·right`.
    Mul(MulRelation),
    /// A remainder's range check against its bound: a prefix-suffix lookup into the
    /// unsigned less-than table.
    RangeCheck(OpRangeCheck, ComputationNode),
    /// The one-hot checks of a [`Self::RangeCheck`]'s read addresses (two relations).
    RangeCheckOneHot(OpRangeCheck, ComputationNode),
    /// `Reshape`: the operand selected into the output's layout.
    Reshape(ReshapeRelation),
    /// `Rsqrt`: the fused division and square-root identities.
    Rsqrt(RsqrtRelation),
    /// `Slice`: the operand's window selected.
    Slice(SliceRelation),
    /// A clamped activation's small-table execution: `γ` drawn, the clamped input's opening
    /// appended at the reduced output point.
    SmallActivation(OpActivation, ComputationNode),
    /// The one-hot checks of a [`Self::SmallActivation`]'s read addresses (two relations).
    SmallActivationOneHot(ComputationNode),
    /// A softmax stage, as `ops::softmax_last_axis` lists them.
    Softmax(SoftmaxRelation),
    /// `Square`: `x²`.
    Square(SquareRelation),
    /// `Sum`: the axis reduction of the pre-clamp accumulation.
    SumAxis(SumAxisRelation),
    /// A trig op's table execution over the downscaled input.
    Trig(OpTrig, ComputationNode),
    /// The one-hot checks of a [`Self::Trig`]'s read addresses (two relations).
    TrigOneHot(OpTrig, ComputationNode),
    /// A packed clamp bucket's value relation: the split sumcheck, or the exact value sumcheck
    /// when the bucket's exact flag is set.
    ClampBucketValue(ClampBucket),
    /// A packed clamp bucket's chunk checks (one-hot checks of its committed chunks).
    ClampBucketChunks(ClampBucket),
    /// A packed remainder bucket's value relation.
    RemainderBucketValue(ClampBucket),
    /// A packed remainder bucket's chunk checks.
    RemainderBucketChunks(ClampBucket),
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
}

/// The remainder a range check bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum OpRangeCheck {
    /// `Div`'s remainder, below the divisor.
    Div,
    /// `MeanOfSquares`'s rescale remainder, below `N·2^S`.
    MeanOfSquares,
    /// `Rsqrt`'s division remainder, below the input.
    RsqrtDiv,
    /// `Rsqrt`'s square-root remainder, at most `2·out`.
    RsqrtSqrt,
    /// A trig op's teleport remainder, below the period.
    Teleport,
}

impl OpRelation {
    /// The relation's verifiers (one, or the several of a stage or a one-hot check), set up
    /// against the accumulator and transcript.
    pub fn verifiers<F: JoltField, T: Transcript>(
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
        fn range_check<F: JoltField, T: Transcript, H: RangeCheckingOperandsTrait>(
            node: &ComputationNode,
            accumulator: &mut VerifierOpeningAccumulator<F>,
            transcript: &mut T,
        ) -> Result<Vec<Box<dyn DynRelationVerifier<F, T>>>, ProofVerifyError> {
            Ok(boxed(
                RangeCheckProvider::<H>::new(node)?
                    .read_raf_verify::<F, T, UnsignedLessThanTable<XLEN>>(
                        accumulator,
                        transcript,
                    )?,
            ))
        }
        fn range_check_onehot<F: JoltField, T: Transcript, H: RangeCheckingOperandsTrait>(
            node: &ComputationNode,
            accumulator: &VerifierOpeningAccumulator<F>,
            transcript: &mut T,
        ) -> Result<Vec<Box<dyn DynRelationVerifier<F, T>>>, ProofVerifyError> {
            Ok(shout::ra_onehot_verifiers(
                &RangeCheckOperands::<H>::new(node)?.get_encoding(node),
                accumulator,
                transcript,
            )?
            .into())
        }
        match self {
            Self::And(relation) => one(relation, accumulator, transcript),
            Self::ConstantDivision(relation) => one(relation, accumulator, transcript),
            Self::Concat(relation) => one(relation, accumulator, transcript),
            Self::Contraction(relation) => one(relation, accumulator, transcript),
            Self::Cube(relation) => one(relation, accumulator, transcript),
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
            Self::Iff(relation) => one(relation, accumulator, transcript),
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
            Self::LookupOneHot(OpLookup::ActivationClamp, node) => Ok(shout::ra_onehot_verifiers(
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
            Self::Mul(relation) => one(relation, accumulator, transcript),
            Self::RangeCheck(OpRangeCheck::Div, node) => {
                range_check::<F, T, DivRangeCheckOperands>(&node, accumulator, transcript)
            }
            Self::RangeCheck(OpRangeCheck::MeanOfSquares, node) => {
                range_check::<F, T, MeanOfSquaresRangeCheckOperands>(&node, accumulator, transcript)
            }
            Self::RangeCheck(OpRangeCheck::RsqrtDiv, node) => {
                range_check::<F, T, RiRangeCheckOperands>(&node, accumulator, transcript)
            }
            Self::RangeCheck(OpRangeCheck::RsqrtSqrt, node) => {
                range_check::<F, T, RsRangeCheckOperands>(&node, accumulator, transcript)
            }
            Self::RangeCheck(OpRangeCheck::Teleport, node) => {
                range_check::<F, T, TeleportRangeCheckOperands>(&node, accumulator, transcript)
            }
            Self::RangeCheckOneHot(OpRangeCheck::Div, node) => {
                range_check_onehot::<F, T, DivRangeCheckOperands>(&node, accumulator, transcript)
            }
            Self::RangeCheckOneHot(OpRangeCheck::MeanOfSquares, node) => {
                range_check_onehot::<F, T, MeanOfSquaresRangeCheckOperands>(
                    &node,
                    accumulator,
                    transcript,
                )
            }
            Self::RangeCheckOneHot(OpRangeCheck::RsqrtDiv, node) => {
                range_check_onehot::<F, T, RiRangeCheckOperands>(&node, accumulator, transcript)
            }
            Self::RangeCheckOneHot(OpRangeCheck::RsqrtSqrt, node) => {
                range_check_onehot::<F, T, RsRangeCheckOperands>(&node, accumulator, transcript)
            }
            Self::RangeCheckOneHot(OpRangeCheck::Teleport, node) => {
                range_check_onehot::<F, T, TeleportRangeCheckOperands>(
                    &node,
                    accumulator,
                    transcript,
                )
            }
            Self::Reshape(relation) => one(relation, accumulator, transcript),
            Self::Rsqrt(relation) => one(relation, accumulator, transcript),
            Self::Slice(relation) => one(relation, accumulator, transcript),
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
            Self::SmallActivationOneHot(node) => Ok(shout::ra_onehot_verifiers(
                &SmallTableRaEncoding { node_idx: node.idx },
                accumulator,
                transcript,
            )?
            .into()),
            Self::Softmax(relation) => relation.verifiers(accumulator, transcript),
            Self::Square(relation) => one(relation, accumulator, transcript),
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
            Self::ClampBucketValue(bucket) => Ok(boxed_value(
                if clamp_split::exact_flag(&bucket, accumulator)? {
                    Box::new(ValueParams::into_verifier(clamp_split::exact_value_params(
                        &bucket,
                        accumulator,
                        transcript,
                    )?))
                } else {
                    Box::new(SplitParams::into_verifier(SplitParams::draw(
                        &bucket,
                        accumulator,
                        transcript,
                    )?)?)
                },
            )),
            Self::ClampBucketChunks(bucket) => {
                let params = if clamp_split::exact_flag(&bucket, accumulator)? {
                    clamp_split::exact_chunk_check_params::<F, T>(&bucket, accumulator, transcript)?
                } else {
                    clamp_split::chunk_check_params::<F, T>(&bucket, accumulator, transcript)?
                };
                Ok(clamp_split::chunk_check_verifiers::<F, T>(params))
            }
            Self::RemainderBucketValue(bucket) => Ok(boxed(ValueParams::into_verifier(
                clamp_split::remainder_value_params(&bucket, accumulator, transcript)?,
            ))),
            Self::RemainderBucketChunks(bucket) => Ok(clamp_split::chunk_check_verifiers::<F, T>(
                clamp_split::remainder_chunk_check_params::<F, T>(
                    &bucket,
                    accumulator,
                    transcript,
                )?,
            )),
        }
    }
}

fn boxed_value<F: JoltField, T: Transcript>(
    verifier: Box<dyn DynRelationVerifier<F, T>>,
) -> Vec<Box<dyn DynRelationVerifier<F, T>>> {
    vec![verifier]
}

/// A zero-round relation or an in-the-clear check, as a plan step holds it.
pub enum OpTie {
    /// `Add`: the pre-clamp accumulation is `left + right`.
    AddAccumulation(AccumulationRelation<Add>),
    /// `MoveAxis`: the operand at the permuted point.
    AxisPermutation(AxisPermutationRelation),
    /// `Broadcast`: the operand replicated along the broadcast axes.
    Broadcast(BroadcastRelation),
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
    /// `Sub`: the pre-clamp accumulation is `left − right`.
    SubAccumulation(AccumulationRelation<Sub>),
    /// `IsNan`: the output is zero.
    ZeroOutput(ZeroOutputRelation),
    /// A scalar `Add`/`Sub`: both operands open at the reduced output point, and the output is
    /// the saturating clamp of `left ± right`, recomputed in the clear.
    ScalarAccumulation {
        /// The node.
        node: ComputationNode,
        /// `+1` for `Add`, `−1` for `Sub`.
        sign: i64,
    },
    /// A scalar clamped node: the output is the saturating clamp of its `ClampAcc` opening,
    /// recomputed in the clear.
    ScalarClamp(ComputationNode),
    /// A scalar fused node's remainder, opened in the clear, lies in `[0, bound)`.
    ScalarRemainder {
        /// The node.
        node: ComputationNode,
        /// The exclusive bound.
        bound: i64,
    },
    /// A model output node whose public output is unsaturated: the pre-clamp accumulation
    /// equals the output.
    ExactClampOutput(ComputationNode),
    /// A trig op's input is `P·q + r` over its teleport quotient and remainder claims.
    TeleportInput(TeleportInputRelation),
    /// A softmax in-the-clear check, as `ops::softmax_last_axis` lists them.
    Softmax(SoftmaxTie),
    /// A packed clamp bucket's exact flag is a bit, and when set every live node of the bucket
    /// has its pre-clamp accumulation equal to its output.
    ClampBucketExact(ClampBucket),
}

impl OpTie {
    /// Append the tie's produced openings and check its identity.
    pub fn verify<F: JoltField, T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<(), ProofVerifyError> {
        match self {
            Self::AddAccumulation(relation) => verify_tie(relation, accumulator, transcript),
            Self::AxisPermutation(relation) => verify_tie(relation, accumulator, transcript),
            Self::Broadcast(relation) => verify_tie(relation, accumulator, transcript),
            Self::Constant(relation) => verify_tie(relation, accumulator, transcript),
            Self::Input(relation) => verify_tie(relation, accumulator, transcript),
            Self::DivQuotient(relation) => verify_tie(relation, accumulator, transcript),
            Self::Identity(relation) => verify_tie(relation, accumulator, transcript),
            Self::Neg(relation) => verify_tie(relation, accumulator, transcript),
            Self::SubAccumulation(relation) => verify_tie(relation, accumulator, transcript),
            Self::ZeroOutput(relation) => verify_tie(relation, accumulator, transcript),
            Self::ScalarAccumulation { node, sign } => {
                let (opening_point, output_claim) =
                    AccOpeningAccessor::new(&*accumulator, node).get_reduced_opening()?;
                let mut provider = AccOpeningAccessor::new(accumulator, node)
                    .into_provider(transcript, opening_point);
                provider.append_nodeio(Target::Input(0))?;
                provider.append_nodeio(Target::Input(1))?;
                let left_claim = provider.get_nodeio(Target::Input(0))?.1;
                let right_claim = provider.get_nodeio(Target::Input(1))?.1;
                let acc = left_claim + F::from_i64(*sign) * right_claim;
                verify_scalar_clamp(acc, output_claim, &node.operator.to_string())
            }
            Self::ScalarClamp(node) => {
                let accessor = AccOpeningAccessor::new(&*accumulator, node);
                let acc = accessor.get_advice(VirtualPoly::ClampAcc)?.1;
                let output_claim = accessor.get_reduced_opening()?.1;
                verify_scalar_clamp(acc, output_claim, &node.operator.to_string())
            }
            Self::ScalarRemainder { node, bound } => {
                let remainder = AccOpeningAccessor::new(&*accumulator, node)
                    .get_advice(VirtualPoly::RescaleRemainder)?
                    .1;
                let value = recover_small_int(remainder).ok_or_else(|| {
                    ProofVerifyError::InvalidOpeningProof(format!(
                        "{} (scalar): remainder claim is not a small integer",
                        node.operator
                    ))
                })?;
                if value < 0 || value >= *bound {
                    return Err(ProofVerifyError::InvalidOpeningProof(format!(
                        "{} (scalar): remainder must lie in [0, {bound})",
                        node.operator
                    )));
                }
                Ok(())
            }
            Self::ExactClampOutput(node) => {
                let acc = accumulator
                    .get_virtual_polynomial_opening(acc_opening_id(node.idx))?
                    .1;
                let out = accumulator.get_node_output_opening(node.idx)?.1;
                if acc != out {
                    return Err(ProofVerifyError::InvalidOpeningProof(format!(
                        "output node {}: unsaturated public output must equal the pre-clamp accumulation",
                        node.idx
                    )));
                }
                Ok(())
            }
            Self::TeleportInput(relation) => relation.verify(accumulator, transcript),
            Self::Softmax(tie) => tie.verify(accumulator, transcript),
            Self::ClampBucketExact(bucket) => {
                if clamp_split::exact_flag(bucket, accumulator)? {
                    clamp_split::verify_exact_nodes(bucket, accumulator)?;
                }
                Ok(())
            }
        }
    }
}

/// Advice openings a plan step appends; the point is stated, the claim is the proof's.
pub enum OpOpen {
    /// The node's pre-clamp accumulation (`ClampAcc`) at the node's output point: the clamp
    /// lookup's `raf`, or the in-the-clear accumulation of a scalar node.
    ClampAcc(ComputationNode),
    /// A fused node's rescale remainder at the reduced output point.
    RescaleRemainder(ComputationNode),
    /// A trig op's teleport quotient at the reduced output point.
    TeleportQuotient(ComputationNode),
    /// A trig op's downscaled remainder at the reduced output point.
    TrigDownscaled(ComputationNode),
    /// A softmax opening, as `ops::softmax_last_axis` lists them.
    Softmax(SoftmaxOpen),
    /// A packed clamp bucket's exact flag, at the empty point.
    ClampExactFlag(usize),
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
            Self::ClampAcc(node) => OpLookupProvider::with_helper(
                node.clone(),
                SaturatingAccClampOperands::for_node(node),
            )
            .append_witness_claim_verifier(accumulator, transcript),
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
            Self::Softmax(open) => open.verify(accumulator, transcript),
            Self::ClampExactFlag(bucket_idx) => accumulator.append_virtual(
                transcript,
                clamp_split::exact_flag_id(*bucket_idx),
                OpeningPoint::new(Vec::<F>::new()),
            ),
        }
    }
}

/// The fused-rescale skeleton around an op's arithmetic relation (`Einsum`, `Mul`, `Square`,
/// `Cube`): the remainder advice, the clamp of the pre-clamp quotient, the arithmetic
/// sumcheck under `proof`, and the remainder's range check. A node without a rescale
/// (`bits == 0`) proves the arithmetic relation alone against its reduced output.
pub fn fused_rescale_plan(
    node: &ComputationNode,
    model: &Model,
    io: &ModelExecutionIO,
    proof: ProofType,
    relation: OpRelation,
) -> Result<Plan, ProofVerifyError> {
    let mut steps = vec![Step::Reduce];
    let bits = fused_rebase::rebase_bits(node).filter(|&bits| bits > 0);
    if let Some(bits) = bits {
        steps.push(Step::Open(OpOpen::RescaleRemainder(node.clone())));
        steps.extend(clamp_steps(node, model, io));
        steps.push(Step::Relation { proof, relation });
        if clamp_lookups_is_scalar(node) {
            steps.push(Step::Tie {
                relation: OpTie::ScalarRemainder {
                    node: node.clone(),
                    bound: 1i64 << bits,
                },
            });
            steps.push(Step::Tie {
                relation: OpTie::ScalarClamp(node.clone()),
            });
        }
        // A non-scalar remainder is range-checked in its packed bucket after the node loop.
    } else {
        steps.push(Step::Relation { proof, relation });
    }
    Ok(Plan(steps))
}

/// The steps that discharge `output = SatClamp(ClampAcc)` for a clamped node: the accumulation
/// is opened; a scalar node is checked in the clear by the caller's [`OpTie::ScalarClamp`], a
/// model output node with an unsaturated public output ties `acc = out`, and every other node
/// is proven in its packed clamp bucket after the node loop.
pub fn clamp_steps(node: &ComputationNode, model: &Model, io: &ModelExecutionIO) -> Vec<Step> {
    let mut steps = vec![Step::Open(OpOpen::ClampAcc(node.clone()))];
    if !clamp_lookups_is_scalar(node) && clamp_split::exact_output_verifier(model, io, node.idx) {
        steps.push(Step::Tie {
            relation: OpTie::ExactClampOutput(node.clone()),
        });
    }
    steps
}

fn clamp_lookups_is_scalar(node: &ComputationNode) -> bool {
    crate::onnx_proof::clamp_lookups::is_scalar(node)
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
