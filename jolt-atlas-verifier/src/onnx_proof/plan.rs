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
//! interpreter reaches the step.

use crate::node_view::ComputationNode;
use crate::onnx_proof::{
    op_lookups::{DefaultLookupOperands, OpLookupEncoding, OpLookupProvider},
    ops::{
        and::AndRelation,
        broadcast::BroadcastRelation,
        clamp::SymmetricClampOperands,
        concat::ConcatRelation,
        constant::PublicRelation,
        div::{DivQuotientRelation, DivRelation},
        iff::IffRelation,
        is_nan::ZeroOutputRelation,
        moveaxis::AxisPermutationRelation,
        neg::SignedCopyRelation,
        reshape::ReshapeRelation,
        scalar_const_div::ConstantDivisionRelation,
        slice::SliceRelation,
    },
    range_checking::{
        range_check_operands::{DivRangeCheckOperands, RangeCheckOperands},
        RangeCheckProvider,
    },
    relation::{verify_tie, RelationVerifier, SetupRelation},
    ProofType,
};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_lookup_tables::{
    clamp::ClampTable, relu::ReluTable, unsigned_less_than::UnsignedLessThanTable,
};
use atlas_sumcheck::{
    opening_proof::VerifierOpeningAccumulator, relation::DynRelationVerifier, shout,
};
use atlas_trace::{
    op::{Constant, Identity, Input, Neg},
    trace::ModelExecutionIO,
    Model,
};
use atlas_transcript::Transcript;
use common::consts::XLEN;

/// One thing the verifier does for a node.
pub enum Step {
    /// Reduce the claims on the node's output to one (the node's evaluation reduction).
    Reduce,
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

/// A relation proven by a sumcheck, as a plan step holds it.
pub enum OpRelation {
    /// `And`: the product of two booleans.
    And(AndRelation),
    /// `ScalarConstDiv`: the quotient by a constant.
    ConstantDivision(ConstantDivisionRelation),
    /// `Concat`: each operand selected into its slot.
    Concat(ConcatRelation),
    /// `Div`: the floor division identity, at a point drawn from the transcript.
    Div(DivRelation),
    /// `Iff`: the masked select.
    Iff(IffRelation),
    /// A prefix-suffix table lookup of the node's operand: the operand's opening is appended,
    /// then the read-raf relation is set up.
    Lookup(OpLookup, ComputationNode),
    /// The one-hot checks of a [`Self::Lookup`]'s read addresses: two relations (the address
    /// polynomial as the product of its chunks; the chunks' Booleanity and Hamming weight).
    LookupOneHot(OpLookup, ComputationNode),
    /// A remainder's range check against its bound: a prefix-suffix lookup into the
    /// unsigned less-than table.
    RangeCheck(OpRangeCheck, ComputationNode),
    /// The one-hot checks of a [`Self::RangeCheck`]'s read addresses (two relations).
    RangeCheckOneHot(OpRangeCheck, ComputationNode),
    /// `Reshape`: the operand selected into the output's layout.
    Reshape(ReshapeRelation),
    /// `Slice`: the operand's window selected.
    Slice(SliceRelation),
}

/// The table a lookup op reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpLookup {
    /// `ReLU`'s table.
    Relu,
    /// `Clamp`'s symmetric clamp table.
    SymmetricClamp,
}

/// The remainder a range check bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpRangeCheck {
    /// `Div`'s remainder, below the divisor.
    Div,
}

impl OpRelation {
    /// The relation's verifiers (one, or the two of a one-hot check), set up against the
    /// accumulator and transcript.
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
        match self {
            Self::And(relation) => one(relation, accumulator, transcript),
            Self::ConstantDivision(relation) => one(relation, accumulator, transcript),
            Self::Concat(relation) => one(relation, accumulator, transcript),
            Self::Div(relation) => one(relation, accumulator, transcript),
            Self::Iff(relation) => one(relation, accumulator, transcript),
            Self::Lookup(OpLookup::Relu, node) => Ok(vec![Box::new(
                OpLookupProvider::<DefaultLookupOperands>::new(node)
                    .read_raf_verify::<F, T, ReluTable<XLEN>, XLEN>(accumulator, transcript)?,
            )]),
            Self::Lookup(OpLookup::SymmetricClamp, node) => Ok(vec![Box::new(
                OpLookupProvider::<SymmetricClampOperands>::new(node)
                    .read_raf_verify::<F, T, ClampTable<XLEN>, XLEN>(accumulator, transcript)?,
            )]),
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
            Self::RangeCheck(OpRangeCheck::Div, node) => Ok(vec![Box::new(
                RangeCheckProvider::<DivRangeCheckOperands>::new(&node)?
                    .read_raf_verify::<F, T, UnsignedLessThanTable<XLEN>>(
                        accumulator,
                        transcript,
                    )?,
            )]),
            Self::RangeCheckOneHot(OpRangeCheck::Div, node) => Ok(shout::ra_onehot_verifiers(
                &RangeCheckOperands::<DivRangeCheckOperands>::new(&node)?.get_encoding(&node),
                accumulator,
                transcript,
            )?
            .into()),
            Self::Reshape(relation) => one(relation, accumulator, transcript),
            Self::Slice(relation) => one(relation, accumulator, transcript),
        }
    }
}

/// A zero-round relation, as a plan step holds it.
pub enum OpTie {
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
    /// `IsNan`: the output is zero.
    ZeroOutput(ZeroOutputRelation),
}

impl OpTie {
    /// Append the tie's produced openings and check its identity.
    pub fn verify<F: JoltField, T: Transcript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<(), ProofVerifyError> {
        match self {
            Self::AxisPermutation(relation) => verify_tie(relation, accumulator, transcript),
            Self::Broadcast(relation) => verify_tie(relation, accumulator, transcript),
            Self::Constant(relation) => verify_tie(relation, accumulator, transcript),
            Self::Input(relation) => verify_tie(relation, accumulator, transcript),
            Self::DivQuotient(relation) => verify_tie(relation, accumulator, transcript),
            Self::Identity(relation) => verify_tie(relation, accumulator, transcript),
            Self::Neg(relation) => verify_tie(relation, accumulator, transcript),
            Self::ZeroOutput(relation) => verify_tie(relation, accumulator, transcript),
        }
    }
}

/// Bind a plan-verified op: its [`OpPlan`] is its whole verification, evaluation reduction
/// included, and `$relation` (a relation of the plan) is checked to be a relation of that op.
/// The `@verify` arm alone is for an op that writes its own `OpLayout` (committed polynomials).
#[macro_export]
macro_rules! impl_plan_op {
    ($inner_ty:ty, $relation:ty) => {
        $crate::bind_relation!($inner_ty, $relation);

        impl $crate::onnx_proof::ops::OpLayout for $inner_ty {
            fn reduction_flow(&self) -> $crate::onnx_proof::ops::ReductionFlow {
                $crate::onnx_proof::ops::ReductionFlow::Custom
            }
        }

        $crate::impl_plan_op!(@verify $inner_ty);
    };
    (@verify $inner_ty:ty) => {
        impl<F: atlas_algebra::field::JoltField, T: atlas_transcript::Transcript>
            $crate::onnx_proof::ops::VerifyOp<F, T> for $inner_ty
        {
            fn verify_with_reduction(
                &self,
                node: &$crate::node_view::ComputationNode,
                verifier: &mut $crate::onnx_proof::Verifier<'_, F, T>,
            ) -> Result<(), atlas_algebra::utils::errors::ProofVerifyError> {
                let plan = $crate::onnx_proof::plan::OpPlan::plan(
                    self,
                    node,
                    verifier.preprocessing.model(),
                    verifier.io,
                )?;
                verifier.run_plan(node, plan)
            }
        }
    };
}
