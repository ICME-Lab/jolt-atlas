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
    ops::{
        and::AndRelation, concat::ConcatRelation, iff::IffRelation, is_nan::ZeroOutputRelation,
        moveaxis::AxisPermutationRelation, neg::SignedCopyRelation, reshape::ReshapeRelation,
        slice::SliceRelation,
    },
    relation::{verify_tie, RelationVerifier},
    ProofType,
};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_sumcheck::{opening_proof::VerifierOpeningAccumulator, relation::DynRelationVerifier};
use atlas_trace::{
    op::{Identity, Neg},
    trace::ModelExecutionIO,
    Model,
};
use atlas_transcript::Transcript;

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
    /// `Concat`: each operand selected into its slot.
    Concat(ConcatRelation),
    /// `Iff`: the masked select.
    Iff(IffRelation),
    /// `Reshape`: the operand selected into the output's layout.
    Reshape(ReshapeRelation),
    /// `Slice`: the operand's window selected.
    Slice(SliceRelation),
}

impl OpRelation {
    /// The relation's verifier, set up against the accumulator and transcript.
    pub fn verifier<F: JoltField, T: Transcript>(
        self,
        accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<Box<dyn DynRelationVerifier<F, T>>, ProofVerifyError> {
        Ok(match self {
            Self::And(relation) => {
                Box::new(RelationVerifier::new(relation, accumulator, transcript)?)
            }
            Self::Concat(relation) => {
                Box::new(RelationVerifier::new(relation, accumulator, transcript)?)
            }
            Self::Iff(relation) => {
                Box::new(RelationVerifier::new(relation, accumulator, transcript)?)
            }
            Self::Reshape(relation) => {
                Box::new(RelationVerifier::new(relation, accumulator, transcript)?)
            }
            Self::Slice(relation) => {
                Box::new(RelationVerifier::new(relation, accumulator, transcript)?)
            }
        })
    }
}

/// A zero-round relation, as a plan step holds it.
pub enum OpTie {
    /// `MoveAxis`: the operand at the permuted point.
    AxisPermutation(AxisPermutationRelation),
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
            Self::Identity(relation) => verify_tie(relation, accumulator, transcript),
            Self::Neg(relation) => verify_tie(relation, accumulator, transcript),
            Self::ZeroOutput(relation) => verify_tie(relation, accumulator, transcript),
        }
    }
}

/// Bind a plan-verified op: its [`OpPlan`] is its whole verification, evaluation reduction
/// included, and `$relation` (a relation of the plan) is checked to be a relation of that op.
#[macro_export]
macro_rules! impl_plan_op {
    ($inner_ty:ty, $relation:ty) => {
        $crate::bind_relation!($inner_ty, $relation);

        impl $crate::onnx_proof::ops::OpLayout for $inner_ty {
            fn reduction_flow(&self) -> $crate::onnx_proof::ops::ReductionFlow {
                $crate::onnx_proof::ops::ReductionFlow::Custom
            }
        }

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
