//! Temporary: a node's steps as the per-node interpreters run them.
//!
//! The per-op plan functions write to a [`ScheduleBuilder`](super::schedule::ScheduleBuilder).
//! Until the prover walks the model's schedule as the verifier does, [`Plan::of_node`] reads
//! one node's steps back out of a builder in the form `jolt-atlas-core`'s `prove_plan`
//! executes: a table read is a [`Step::Defer`] to [`DeferredBatch::Lookups`], a one-hot check a
//! [`Step::Defer`] to [`DeferredBatch::OneHot`]. This module and
//! [`deferred_lookups`](super::deferred_lookups) go away with that walk.

use crate::node_view::ComputationNode;
use crate::onnx_proof::{
    deferred_lookups::DeferredBatch,
    relation::OpRel,
    schedule::{self, OpOpen, OpPlan, OpRelation, ScheduleBuilder},
    ProofType,
};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_trace::{trace::ModelExecutionIO, Model};

/// One thing the prover does for a node.
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

impl<F: JoltField> Plan<F> {
    /// The steps `node`'s operator writes, as the prover's per-node walk runs them. The one-hot
    /// checks, which have no step in the node's turn, are queued after the node's steps in the
    /// order they were registered.
    pub fn of_node(
        node: &ComputationNode,
        model: &Model,
        io: &ModelExecutionIO,
    ) -> Result<Self, ProofVerifyError> {
        let mut builder = ScheduleBuilder::new();
        builder.enter_node(node.idx);
        crate::dispatch_operator!(node, |inner| OpPlan::plan(
            inner,
            node,
            model,
            io,
            &mut builder
        ))?;
        let ScheduleBuilder {
            relations,
            steps,
            one_hot,
            ..
        } = builder;
        let mut relations: Vec<Option<OpRelation<F>>> = relations.into_iter().map(Some).collect();
        let mut take = |id: schedule::RelId| {
            relations
                .get_mut(id.0)
                .and_then(Option::take)
                .ok_or_else(|| {
                    ProofVerifyError::Malformed(format!(
                        "plan: node {}'s relation {} is used twice",
                        node.idx, id.0
                    ))
                })
        };
        let mut plan = Vec::new();
        for step in steps {
            plan.push(match step {
                schedule::Step::Reduce { .. } => Step::Reduce,
                schedule::Step::Open(open) => Step::Open(open),
                schedule::Step::Setup(id) => Step::Defer {
                    batch: DeferredBatch::Lookups,
                    relation: take(id)?,
                },
                schedule::Step::Sumcheck {
                    proof,
                    members,
                    batched,
                } => {
                    let mut relations = members
                        .into_iter()
                        .map(&mut take)
                        .collect::<Result<Vec<_>, _>>()?;
                    match (batched, relations.pop()) {
                        (false, Some(relation)) if relations.is_empty() => Step::Relation {
                            proof: proof.1,
                            relation,
                        },
                        (_, last) => {
                            relations.extend(last);
                            Step::Batch {
                                proof: proof.1,
                                relations,
                            }
                        }
                    }
                }
                schedule::Step::Tie(id) => match take(id)? {
                    OpRelation::Op(relation) => Step::Tie { relation },
                    _ => {
                        return Err(ProofVerifyError::Malformed(format!(
                            "plan: node {}'s tie is not an op relation",
                            node.idx
                        )))
                    }
                },
            });
        }
        for id in one_hot {
            plan.push(Step::Defer {
                batch: DeferredBatch::OneHot,
                relation: take(id)?,
            });
        }
        Ok(Self(plan))
    }
}
