//! The verification schedule: what the verifier checks, in transcript order, as data.
//!
//! A [`Schedule`] is a list of [`Step`]s over a table of relations ([`OpRelation`]), each
//! named by its [`RelId`]. [`Schedule::compile`] builds a model's: each node's steps in reverse
//! topological order, written by its op's [`OpPlan::plan`] to a [`ScheduleBuilder`], then the
//! steps [`ScheduleBuilder::finish`] adds for the whole model: the packed range buckets' value
//! sumcheck, the lookups sumcheck (every table read) and the one-hot
//! sumcheck (every one-hot check of a read's addresses). `finish` checks that every relation
//! is set up at most once and verified exactly once, and that no sumcheck is empty. The
//! interpreter ([`Verifier::run`](super::Verifier::run)) executes the steps in order.
//!
//! A table read is set up in its node's turn (a `Setup` step), where it draws its challenges
//! and appends its operand claim, and is proven in the lookups sumcheck. A one-hot check
//! reads the read-address claim that sumcheck (or its node's own read) produces, so it is set
//! up only when the one-hot sumcheck runs.
//!
//! A step holds values, never closures. A relation is one variant of [`OpRelation`] per kind
//! of verifier construction: an op relation ([`OpRel`]), a prefix-suffix table read
//! ([`TableRead`]), the one-hot checks of a read's addresses ([`RaEncoding`]), and the stages
//! with their own shape. A tie step names a zero-round [`OpRel`], checked by
//! [`verify_tie`](super::relation::verify_tie); an open step holds an [`OpOpen`]. A relation
//! whose parameters depend on earlier claims or on challenges reads and draws them in its
//! [`SetupRelation::setup`](super::relation::SetupRelation::setup), which runs when the
//! relation is set up: at its `Setup` step, or at its `Sumcheck` step when it has none.

use crate::node_view::{ComputationNode, ModelView};
use crate::onnx_proof::{
    fused_rebase,
    lookup::{table_read_verifier, RaEncoding, TableRead},
    ops::{
        gather::{self, small::GatherSmallShape},
        softmax_last_axis::{self, SoftmaxOpen, SoftmaxShape},
    },
    range_bucket::{is_digit_backed, range_buckets, RangeBucket},
    range_check::{self, ValueParams},
    relation::{op_rel_verifier, OpRel},
};
use crate::utils::opening_access::AccOpeningAccessor;
use atlas_algebra::{
    field::JoltField,
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_sumcheck::{
    opening_proof::{OpeningId, PolynomialId, VerifierOpeningAccumulator},
    relation::DynRelationVerifier,
    shout,
};
use atlas_trace::Model;
use atlas_transcript::VerifyingTranscript;
use common::{CommittedPoly, VirtualPoly};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// The steps of one node's plan that are not yet in the schedule.
struct NodePlan {
    node: usize,
    /// Whether the node takes its claim at a point of its own ([`Step::Point`]) and so waits
    /// for no consumer.
    digit_backed: bool,
    steps: VecDeque<Step>,
}

/// The most rows (`Σ 2^rounds` over the members) one batched sumcheck of a wave holds: the
/// members' witnesses are live together, so a wave with more is proven as several sumchecks.
/// A relation whose size the schedule does not know is a sumcheck of its own.
const WAVE_BUDGET: usize = 1 << 24;

/// The nodes' plans as one step list, in waves.
///
/// A node is *complete* when every step of its plan is in the list, and *ready* when it is
/// digit-backed or every consumer of its output is complete: the claims its plan reads then
/// exist. Each wave takes the nodes ready at its start, in descending order, writes each
/// one's steps up to its next sumcheck, and then writes those sumchecks as one batched
/// sumcheck (several, over [`WAVE_BUDGET`]). The nodes of a wave do not read each other's
/// claims, so their sumchecks share challenges. `Err` if no node is ready while one is
/// incomplete.
fn waves<F: JoltField>(
    mut plans: Vec<NodePlan>,
    consumers: &[Vec<usize>],
    relations: &[OpRelation<F>],
) -> Result<Vec<Step>, ProofVerifyError> {
    let rows = |id: &RelId| match relations.get(id.0) {
        Some(OpRelation::Op(relation)) => 1usize << relation.rounds.min(usize::BITS as usize - 1),
        _ => WAVE_BUDGET,
    };
    let mut complete = vec![false; consumers.len()];
    let mut steps = Vec::new();
    while plans.iter().any(|plan| !plan.steps.is_empty()) || complete.iter().any(|done| !done) {
        let is_ready = |plan: &NodePlan| {
            let waits_for_nobody = plan.digit_backed
                || consumers.get(plan.node).is_none_or(|readers| {
                    readers
                        .iter()
                        .all(|&reader| complete.get(reader).copied().unwrap_or(true))
                });
            waits_for_nobody && !complete.get(plan.node).copied().unwrap_or(true)
        };
        let ready: Vec<usize> = (0..plans.len())
            .filter(|&at| plans.get(at).is_some_and(is_ready))
            .collect();
        if ready.is_empty() {
            return Err(ProofVerifyError::Malformed(
                "schedule: a node waits for a consumer that waits for it".to_string(),
            ));
        }
        // The points of the wave's digit-backed nodes, before any step that reads a claim.
        for &at in &ready {
            let Some(plan) = plans.get(at) else {
                continue;
            };
            for step in &plan.steps {
                match step {
                    Step::Sumcheck { .. } => break,
                    Step::Point { node, vars } => steps.push(Step::Draw {
                        node: *node,
                        vars: *vars,
                    }),
                    _ => {}
                }
            }
        }
        let mut members: Vec<RelId> = Vec::new();
        for &at in &ready {
            let Some(plan) = plans.get_mut(at) else {
                continue;
            };
            while let Some(step) = plan.steps.pop_front() {
                match step {
                    Step::Sumcheck { members: own, .. } => {
                        members.extend(own);
                        break;
                    }
                    other => steps.push(other),
                }
            }
        }
        let mut batch: Vec<RelId> = Vec::new();
        let mut filled = 0usize;
        for id in members {
            let size = rows(&id);
            if !batch.is_empty() && filled.saturating_add(size) > WAVE_BUDGET {
                steps.push(Step::Sumcheck {
                    members: std::mem::take(&mut batch),
                    batched: true,
                });
                filled = 0;
            }
            filled = filled.saturating_add(size);
            batch.push(id);
        }
        if !batch.is_empty() {
            steps.push(Step::Sumcheck {
                members: batch,
                batched: true,
            });
        }
        for &at in &ready {
            if let Some(plan) = plans.get(at) {
                if plan.steps.is_empty() {
                    if let Some(done) = complete.get_mut(plan.node) {
                        *done = true;
                    }
                }
            }
        }
    }
    Ok(steps)
}

/// The name of a relation in a schedule: its index in the schedule's relation table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RelId(pub usize);

/// One thing the verifier does.
#[derive(Clone)]
pub enum Step {
    /// Reduce the claims on `node`'s output to one (the node's evaluation reduction).
    Reduce {
        /// The node whose output claims are reduced.
        node: usize,
    },
    /// Draw the point a digit-backed node's output is opened at ([`is_digit_backed`]). A
    /// wave draws the points of all its digit-backed nodes before any of their claims is
    /// read: the outputs are committed, as their buckets' digits, before any point is drawn,
    /// so a claim at one point is fixed whatever the other points are, and the prover may
    /// compute the wave's claims together.
    Draw {
        /// The node whose output is opened.
        node: usize,
        /// The number of coordinates of the point: log₂ of the node's padded output size.
        vars: usize,
    },
    /// Take the claim of a digit-backed node's output at the point its [`Step::Draw`] drew:
    /// the node's relation runs on that claim, and the bucket holding the output's digits
    /// checks it with every claim the node's consumers appended.
    Point {
        /// The node whose output is opened.
        node: usize,
        /// The number of coordinates of the point: log₂ of the node's padded output size.
        vars: usize,
    },
    /// Append advice openings at a stated point; their claims are read from the transcript.
    Open(OpOpen),
    /// Set a relation up (its draws and appends); a later `Sumcheck` verifies it.
    Setup(RelId),
    /// Set up each member not yet set up, in order, then verify the members as one sumcheck,
    /// its round messages and the members' produced claims read from the transcript.
    Sumcheck {
        /// The relations proven, in instance order.
        members: Vec<RelId>,
        /// Whether the sumcheck is a batch, which draws a coefficient per instance (even for
        /// one instance); otherwise it is the sumcheck of its single member's one instance.
        batched: bool,
    },
    /// Check a zero-round relation (an [`OpRel`]): append its openings and require the
    /// identity to hold.
    Tie(RelId),
}

/// A verification schedule: the relations checked and the steps that check them, in transcript
/// order. A function of the model: the one step that reads the public IO is an `Input` node's
/// tie, whose public tensor the interpreter fills in. Built by a [`ScheduleBuilder`], whose
/// [`finish`](ScheduleBuilder::finish) checks it.
#[derive(Clone)]
pub struct Schedule<F: JoltField> {
    relations: Vec<OpRelation<F>>,
    steps: Vec<Step>,
}

impl<F: JoltField> std::fmt::Debug for Schedule<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Schedule")
            .field("relations", &self.relations.len())
            .field("steps", &self.steps.len())
            .finish()
    }
}

/// One line per step: where it runs (`n<node>`, or `model` for a sumcheck over relations of
/// several nodes or of none), its kind (`Sumcheck` is one relation's sumcheck, `Batch` a
/// batched one) and what it checks: an op relation by its [`OpRelKind`](super::relation::OpRelKind), any other relation or
/// an opening by name, a member of a `model` step with its node. For people and for the
/// schedule pins; nothing parses it.
impl<F: JoltField> std::fmt::Display for Schedule<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let relation = |id: &RelId| self.relations.get(id.0);
        let name = |id: &RelId| relation(id).map_or_else(|| "?".to_string(), OpRelation::name);
        let node = |id: &RelId| relation(id).and_then(OpRelation::node);
        let at = |node: Option<usize>| node.map_or_else(|| "n?".to_string(), |n| format!("n{n}"));
        for step in &self.steps {
            match step {
                Step::Reduce { node } => writeln!(f, "n{node} Reduce")?,
                Step::Draw { node, .. } => writeln!(f, "n{node} Draw")?,
                Step::Point { node, .. } => writeln!(f, "n{node} Point")?,
                Step::Open(open) => writeln!(f, "n{} Open [{}]", open.node(), open.name())?,
                Step::Setup(id) => writeln!(f, "{} Setup [{}]", at(node(id)), name(id))?,
                Step::Tie(id) => writeln!(f, "{} Tie [{}]", at(node(id)), name(id))?,
                Step::Sumcheck {
                    members, batched, ..
                } => {
                    let kind = if *batched { "Batch" } else { "Sumcheck" };
                    if let Some(shared) = self.sumcheck_node(members) {
                        let names: Vec<String> = members.iter().map(name).collect();
                        writeln!(f, "{} {kind} [{}]", at(Some(shared)), names.join(", "))?;
                    } else {
                        let names: Vec<String> = members
                            .iter()
                            .map(|id| match node(id) {
                                Some(node) => format!("n{node}:{}", name(id)),
                                None => name(id),
                            })
                            .collect();
                        writeln!(f, "model {kind} [{}]", names.join(", "))?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl<F: JoltField> Schedule<F> {
    /// The schedule of `model`. `Err` if a node does not have the shape its operator proves.
    pub fn compile(model: &Model) -> Result<Self, ProofVerifyError> {
        let mut builder = ScheduleBuilder::new();
        let views = model.views();
        let mut plans: Vec<NodePlan> = Vec::with_capacity(views.len());
        for node in views.iter().rev() {
            builder.enter_node(node.idx);
            let digit_backed = is_digit_backed(model, node);
            if digit_backed {
                builder.fresh_point(node.pow2_padded_num_output_elements().log_2());
            }
            crate::dispatch_operator!(node, |inner| OpPlan::plan(
                inner,
                node,
                model,
                &mut builder
            ))?;
            plans.push(NodePlan {
                node: node.idx,
                digit_backed,
                steps: std::mem::take(&mut builder.steps).into(),
            });
        }
        // Who reads each node's output.
        let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); views.len()];
        for node in views.iter() {
            for &operand in &node.inputs {
                if let Some(readers) = consumers.get_mut(operand) {
                    readers.push(node.idx);
                }
            }
        }
        builder.steps = waves(plans, &consumers, &builder.relations)?;
        builder.finish(range_buckets(model)?)
    }

    /// The committed polynomials the schedule opens, sorted and without repeats: those of every
    /// `Open` step and of every relation.
    pub fn committed(&self) -> Vec<CommittedPoly> {
        let mut polys = BTreeSet::new();
        for step in &self.steps {
            if let Step::Open(open) = step {
                polys.extend(open.committed());
            }
        }
        for relation in &self.relations {
            polys.extend(relation.committed());
        }
        polys.into_iter().collect()
    }

    /// The sumchecks a proof of this schedule carries, in step order: each one's members.
    pub fn sumchecks(&self) -> impl Iterator<Item = &[RelId]> {
        self.steps.iter().filter_map(|step| match step {
            Step::Sumcheck { members, .. } => Some(members.as_slice()),
            _ => None,
        })
    }

    /// The node whose turn a sumcheck over `members` runs in: the node all of them belong to.
    /// `None` for a sumcheck over relations of several nodes or of none.
    pub fn sumcheck_node(&self, members: &[RelId]) -> Option<usize> {
        let node = |id: &RelId| self.relations.get(id.0).and_then(OpRelation::node);
        let shared = members.first().and_then(node)?;
        members
            .iter()
            .all(|id| node(id) == Some(shared))
            .then_some(shared)
    }

    /// The steps, in transcript order.
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// This schedule's relations under other steps. `Err` if the steps are not a valid
    /// schedule of them.
    #[cfg(any(test, feature = "test-util"))]
    pub fn with_steps(&self, steps: Vec<Step>) -> Result<Self, ProofVerifyError> {
        let schedule = Self {
            relations: self.relations.clone(),
            steps,
        };
        schedule.validate()?;
        Ok(schedule)
    }

    /// The relations the steps name, indexed by [`RelId`].
    pub fn relations(&self) -> &[OpRelation<F>] {
        &self.relations
    }

    /// The relation `id` names. `Err` if the schedule holds no such relation.
    pub fn relation(&self, id: RelId) -> Result<&OpRelation<F>, ProofVerifyError> {
        self.relations
            .get(id.0)
            .ok_or_else(|| ProofVerifyError::Malformed(format!("schedule: no relation {}", id.0)))
    }

    /// The zero-round relation a `Tie` step names. `Err` if `id` does not name an op relation.
    pub fn tie(&self, id: RelId) -> Result<&OpRel<F>, ProofVerifyError> {
        match self.relation(id)? {
            OpRelation::Op(relation) => Ok(relation),
            _ => Err(ProofVerifyError::Malformed(format!(
                "schedule: relation {} is tied but is not an op relation",
                id.0
            ))),
        }
    }

    /// Every relation is the target of at most one `Setup`, and a member of exactly one
    /// `Sumcheck` or the target of exactly one `Tie`; a `Setup` precedes the `Sumcheck` that
    /// holds its relation; no `Sumcheck` is empty; a `Point` reads the claim of a node whose
    /// point one earlier `Draw` drew, with the same variables, and every `Draw` has its
    /// `Point`; and no step appends a claim on a node's output after that node's `Reduce`
    /// ([`OpRelation::appended_outputs`]), whatever order the steps were written in.
    fn validate(&self) -> Result<(), ProofVerifyError> {
        let malformed =
            |what: String| Err(ProofVerifyError::Malformed(format!("schedule: {what}")));
        let mut set_up = vec![false; self.relations.len()];
        let mut verified = vec![false; self.relations.len()];
        // Mark `id` in `marks`; `Ok(true)` if it was marked already.
        let mark = |marks: &mut Vec<bool>, id: RelId| match marks.get_mut(id.0) {
            Some(mark) => Ok(std::mem::replace(mark, true)),
            None => Err(ProofVerifyError::Malformed(format!(
                "schedule: no relation {}",
                id.0
            ))),
        };
        let is_marked = |marks: &[bool], id: RelId| marks.get(id.0).copied().unwrap_or(false);
        // The nodes whose output claims have been reduced to one: no claim may be appended on
        // one of them afterwards.
        let mut reduced = BTreeSet::new();
        // The points drawn whose claims are not yet read, by node, with their variables.
        let mut drawn: BTreeMap<usize, usize> = BTreeMap::new();
        let appended_late = |reduced: &BTreeSet<usize>, id: RelId, at_setup: bool| {
            self.relations.get(id.0).and_then(|relation| {
                relation
                    .appended_outputs(at_setup)
                    .into_iter()
                    .find(|node| reduced.contains(node))
            })
        };
        let late = |node: usize| {
            malformed(format!(
                "a claim on node {node}'s output is appended after its reduction"
            ))
        };
        for step in &self.steps {
            match step {
                Step::Reduce { node } => {
                    reduced.insert(*node);
                }
                Step::Draw { node, vars } => {
                    if drawn.insert(*node, *vars).is_some() {
                        return malformed(format!("node {node}'s point is drawn twice"));
                    }
                }
                Step::Point { node, vars } => {
                    if drawn.remove(node) != Some(*vars) {
                        return malformed(format!(
                            "node {node}'s claim is read at a point not drawn for it"
                        ));
                    }
                }
                Step::Open(_) => {}
                Step::Setup(id) => {
                    if let Some(node) = appended_late(&reduced, *id, true) {
                        return late(node);
                    }
                    if is_marked(&verified, *id) {
                        return malformed(format!(
                            "relation {} is set up after it is verified",
                            id.0
                        ));
                    }
                    if mark(&mut set_up, *id)? {
                        return malformed(format!("relation {} is set up twice", id.0));
                    }
                }
                Step::Sumcheck { members, .. } => {
                    if members.is_empty() {
                        return malformed("a sumcheck has no member".to_string());
                    }
                    for id in members {
                        let own_setup = !is_marked(&set_up, *id);
                        if let Some(node) = (own_setup
                            .then(|| appended_late(&reduced, *id, true))
                            .flatten())
                        .or_else(|| appended_late(&reduced, *id, false))
                        {
                            return late(node);
                        }
                        if mark(&mut verified, *id)? {
                            return malformed(format!("relation {} is verified twice", id.0));
                        }
                    }
                }
                Step::Tie(id) => {
                    self.tie(*id)?;
                    if is_marked(&set_up, *id) {
                        return malformed(format!("the tie {} has a setup step", id.0));
                    }
                    if let Some(node) = appended_late(&reduced, *id, true)
                        .or_else(|| appended_late(&reduced, *id, false))
                    {
                        return late(node);
                    }
                    if mark(&mut verified, *id)? {
                        return malformed(format!("relation {} is verified twice", id.0));
                    }
                }
            }
        }
        if let Some((node, _)) = drawn.first_key_value() {
            return malformed(format!(
                "node {node}'s point is drawn and its claim never read"
            ));
        }
        match verified.iter().position(|verified| !verified) {
            Some(id) => malformed(format!("relation {id} is never verified")),
            None => Ok(()),
        }
    }
}

/// What the per-op plan functions write a schedule to.
pub struct ScheduleBuilder<F: JoltField> {
    /// The relations registered so far.
    pub(crate) relations: Vec<OpRelation<F>>,
    /// The steps emitted so far.
    pub(crate) steps: Vec<Step>,
    /// The table reads set up in their node's turn, in the order registered.
    pub(crate) lookups: Vec<RelId>,
    /// The one-hot checks, in the order registered.
    pub(crate) one_hot: Vec<RelId>,
    /// The node whose steps are being written; `usize::MAX` before any node is entered.
    node: usize,
    /// For a digit-backed node, the number of coordinates of its output's point.
    point_vars: Option<usize>,
}

impl<F: JoltField> ScheduleBuilder<F> {
    /// An empty schedule.
    pub fn new() -> Self {
        Self {
            relations: Vec::new(),
            steps: Vec::new(),
            lookups: Vec::new(),
            one_hot: Vec::new(),
            node: usize::MAX,
            point_vars: None,
        }
    }

    /// Write the steps that follow as node `node`'s.
    pub fn enter_node(&mut self, node: usize) {
        self.node = node;
        self.point_vars = None;
    }

    /// The node entered is digit-backed, its output a polynomial of `vars` variables: its
    /// plan's [`reduce`](Self::reduce) takes the output's claim at a fresh point.
    pub fn fresh_point(&mut self, vars: usize) {
        self.point_vars = Some(vars);
    }

    fn register(&mut self, relation: OpRelation<F>) -> RelId {
        self.relations.push(relation);
        RelId(self.relations.len() - 1)
    }

    /// The claim the node's relation runs on: its evaluation reduction, or for a
    /// digit-backed node its output's claim at a fresh point.
    pub fn reduce(&mut self) {
        self.steps.push(match self.point_vars {
            Some(vars) => Step::Point {
                node: self.node,
                vars,
            },
            None => Step::Reduce { node: self.node },
        });
    }

    /// Advice openings at a stated point.
    pub fn open(&mut self, open: OpOpen) {
        self.steps.push(Step::Open(open));
    }

    /// `relation`'s sumcheck: the relation is set up, then verified, at this step.
    pub fn relation(&mut self, relation: OpRelation<F>) {
        let id = self.register(relation);
        self.steps.push(Step::Sumcheck {
            members: vec![id],
            batched: false,
        });
    }

    /// Several relations as one batched sumcheck, their setups run in list order at this
    /// step.
    pub fn batch(&mut self, relations: Vec<OpRelation<F>>) {
        let members = relations
            .into_iter()
            .map(|relation| self.register(relation))
            .collect();
        self.steps.push(Step::Sumcheck {
            members,
            batched: true,
        });
    }

    /// A zero-round relation, checked at this step.
    pub fn tie(&mut self, relation: OpRel<F>) {
        let id = self.register(OpRelation::Op(relation));
        self.steps.push(Step::Tie(id));
    }

    /// A table read: set up at this step, in its node's turn, and proven with every other
    /// read in the lookups sumcheck.
    pub fn lookup(&mut self, relation: OpRelation<F>) {
        let id = self.register(relation);
        self.steps.push(Step::Setup(id));
        self.lookups.push(id);
    }

    /// A one-hot check of a read's addresses: set up and proven with every other one-hot check
    /// in the one-hot sumcheck, which runs after the lookups sumcheck has produced the
    /// read-address claim it reads.
    pub fn one_hot(&mut self, relation: OpRelation<F>) {
        let id = self.register(relation);
        self.one_hot.push(id);
    }

    /// The schedule written, closed by the model-wide steps: the value sumcheck of `buckets`
    /// (one batch over the buckets), the lookups sumcheck, then
    /// the one-hot sumcheck; a step with no member is left out. `Err` unless every relation is
    /// set up at most once, before its sumcheck, and verified exactly once (by one sumcheck or
    /// one tie), and every sumcheck has a member.
    pub fn finish(mut self, buckets: Vec<RangeBucket>) -> Result<Schedule<F>, ProofVerifyError> {
        if !buckets.is_empty() {
            self.batch(
                buckets
                    .into_iter()
                    .map(OpRelation::RemainderBucketValue)
                    .collect(),
            );
        }
        for members in [
            std::mem::take(&mut self.lookups),
            std::mem::take(&mut self.one_hot),
        ] {
            if !members.is_empty() {
                self.steps.push(Step::Sumcheck {
                    members,
                    batched: true,
                });
            }
        }
        let schedule = Schedule {
            relations: self.relations,
            steps: self.steps,
        };
        schedule.validate()?;
        Ok(schedule)
    }
}

/// An op's verification, written to a schedule.
pub trait OpPlan {
    /// Write the steps that verify `node`, from the model alone.
    fn plan<F: JoltField>(
        &self,
        node: &ComputationNode,
        model: &Model,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError>;
}

/// A relation proven by a sumcheck, as a schedule holds it: one variant per kind of verifier
/// construction.
#[derive(Clone)]
pub enum OpRelation<F: JoltField> {
    /// An op relation.
    Op(OpRel<F>),
    /// A prefix-suffix table read: the operand's opening is appended at the read's cycle
    /// point, then the read-raf relation is set up.
    TableRead(TableRead),
    /// The one-hot checks of a read's addresses: two relations (the address polynomial as the
    /// product of its chunks; the chunks' Hamming weight).
    OneHot(RaEncoding),
    /// `GatherSmall`: the Hamming-Booleanity check of its one-hot `ra`.
    GatherSmallOneHot(GatherSmallShape),
    /// `GatherSmall`: the Hamming-weight check of its one-hot `ra`.
    GatherSmallHammingWeight(GatherSmallShape),
    /// `SoftmaxLastAxis`: the read of the output, sum and shifted sum tables at the reads'
    /// point.
    SoftmaxRead(SoftmaxShape),
    /// A packed range bucket's value relation: the range claims of its members
    /// ([`range_bucket::RangeClaim`](crate::onnx_proof::range_bucket::RangeClaim)).
    RemainderBucketValue(RangeBucket),
}

impl<F: JoltField> OpRelation<F> {
    /// The nodes on whose outputs the relation appends a claim: when it is set up
    /// (`at_setup`: an op relation's setup opens, a table read's operand) or when it is
    /// verified (an op relation's produced openings).
    pub fn appended_outputs(&self, at_setup: bool) -> Vec<usize> {
        let ids: Vec<OpeningId> = match (self, at_setup) {
            (Self::Op(relation), true) => relation.setup_opens.iter().map(|open| open.id).collect(),
            (Self::Op(relation), false) => relation.produced.clone(),
            (Self::TableRead(read), true) => vec![read.operand],
            _ => Vec::new(),
        };
        ids.into_iter()
            .filter_map(|id| match id.virtual_poly() {
                Some(VirtualPoly::NodeOutput(node)) => Some(node),
                _ => None,
            })
            .collect()
    }

    /// The node the relation belongs to; `None` for a range bucket's, which spans nodes.
    pub fn node(&self) -> Option<usize> {
        match self {
            Self::Op(relation) => Some(relation.node.idx),
            Self::TableRead(read) => Some(read.node.idx),
            Self::OneHot(encoding) => Some(encoding.node),
            Self::GatherSmallOneHot(shape) | Self::GatherSmallHammingWeight(shape) => {
                Some(shape.node.idx)
            }
            Self::SoftmaxRead(shape) => Some(shape.idx()),
            Self::RemainderBucketValue(_) => None,
        }
    }

    /// The relation's name in a rendered schedule: an op relation's kind, a table read's
    /// table, a one-hot check's read-address polynomial, a bucket relation's bucket.
    pub fn name(&self) -> String {
        match self {
            Self::Op(relation) => format!("{:?}", relation.kind),
            Self::TableRead(read) => format!("TableRead({:?})", read.table),
            Self::OneHot(encoding) => {
                // The read-address polynomial checked, without its node.
                let address = match encoding.ra_source.polynomial {
                    PolynomialId::Virtual(poly) => format!("{poly:?}"),
                    PolynomialId::Committed(poly) => format!("{poly:?}"),
                };
                let address = address.split('(').next().unwrap_or_default();
                format!("OneHot({address})")
            }
            Self::GatherSmallOneHot(_) => "GatherSmallOneHot".to_string(),
            Self::GatherSmallHammingWeight(_) => "GatherSmallHammingWeight".to_string(),
            Self::SoftmaxRead(_) => "SoftmaxRead".to_string(),
            Self::RemainderBucketValue(bucket) => format!("BucketValue({})", bucket.idx),
        }
    }

    /// The committed polynomials the relation opens: an op relation's committed produced and
    /// setup openings, a one-hot check's address chunks, a small gather's one-hot `ra`, a
    /// range bucket's digit columns.
    pub fn committed(&self) -> Vec<CommittedPoly> {
        match self {
            Self::Op(relation) => relation
                .produced
                .iter()
                .chain(relation.setup_opens.iter().map(|open| &open.id))
                .filter_map(|id| match id.polynomial {
                    PolynomialId::Committed(poly) => Some(poly),
                    PolynomialId::Virtual(_) => None,
                })
                .collect(),
            Self::OneHot(encoding) => encoding.committed_polys(),
            Self::RemainderBucketValue(bucket) => bucket.committed_polys(),
            Self::GatherSmallOneHot(shape) | Self::GatherSmallHammingWeight(shape) => {
                vec![CommittedPoly::GatherRa(shape.node.idx)]
            }
            Self::TableRead(_) | Self::SoftmaxRead(_) => Vec::new(),
        }
    }

    /// The relation's verifiers (one, or the several of a stage or a one-hot check), set up
    /// against the accumulator and transcript.
    pub fn verifiers<T: VerifyingTranscript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<Vec<Box<dyn DynRelationVerifier<F, T>>>, ProofVerifyError> {
        fn boxed<F: JoltField, T: VerifyingTranscript>(
            verifier: impl DynRelationVerifier<F, T> + 'static,
        ) -> Vec<Box<dyn DynRelationVerifier<F, T>>> {
            vec![Box::new(verifier)]
        }
        match self {
            Self::Op(relation) => Ok(boxed(op_rel_verifier(
                relation.clone(),
                accumulator,
                transcript,
            )?)),
            Self::TableRead(read) => Ok(vec![table_read_verifier(read, accumulator, transcript)?]),
            Self::OneHot(encoding) => {
                Ok(shout::ra_onehot_verifiers(encoding, accumulator, transcript)?.into())
            }
            Self::GatherSmallOneHot(shape) => Ok(boxed(gather::small::build_stage2_verifier(
                shape,
                accumulator,
                transcript,
            )?)),
            Self::GatherSmallHammingWeight(shape) => Ok(boxed(
                gather::small::build_stage3_verifier(shape, accumulator, transcript)?,
            )),
            Self::SoftmaxRead(shape) => Ok(vec![softmax_last_axis::read_verifier(
                shape.clone(),
                accumulator,
                transcript,
            )?]),
            Self::RemainderBucketValue(bucket) => Ok(boxed(ValueParams::into_verifier(
                range_check::remainder_value_params(bucket, accumulator, transcript)?,
            ))),
        }
    }
}

/// Advice openings a step appends; the point is stated, the claim is read from the transcript.
#[derive(Clone)]
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
    /// The node whose advice is opened.
    pub fn node(&self) -> usize {
        match self {
            Self::RescaleRemainder(node)
            | Self::TrigPeriodQuotient(node)
            | Self::TrigDownscaled(node)
            | Self::SmoothReluRelu(node) => node.idx,
            Self::Softmax(SoftmaxOpen::Gaps(shape) | SoftmaxOpen::Addresses(shape)) => shape.idx(),
        }
    }

    /// The opening's name in a rendered schedule.
    pub fn name(&self) -> &'static str {
        match self {
            Self::RescaleRemainder(_) => "RescaleRemainder",
            Self::TrigPeriodQuotient(_) => "TrigPeriodQuotient",
            Self::TrigDownscaled(_) => "TrigDownscaled",
            Self::SmoothReluRelu(_) => "SmoothReluRelu",
            Self::Softmax(SoftmaxOpen::Gaps(_)) => "SoftmaxGaps",
            Self::Softmax(SoftmaxOpen::Addresses(_)) => "SoftmaxAddresses",
        }
    }

    /// The committed polynomials the step opens.
    pub fn committed(&self) -> Vec<CommittedPoly> {
        match self {
            Self::TrigPeriodQuotient(node) => vec![CommittedPoly::TrigPeriodQuotient(node.idx)],
            Self::Softmax(open) => open.committed(),
            Self::RescaleRemainder(_) | Self::TrigDownscaled(_) | Self::SmoothReluRelu(_) => {
                Vec::new()
            }
        }
    }

    /// Append the openings.
    pub fn verify<F: JoltField, T: VerifyingTranscript>(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<(), ProofVerifyError> {
        /// Append the node's advice `poly` at its reduced output point.
        fn at_reduced_output<F: JoltField, T: VerifyingTranscript, Poly: Into<PolynomialId>>(
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
/// `Cube`): the remainder advice, then the arithmetic sumcheck, whose input claim
/// is `out·2^S + rem`. The remainder, and the output unless it is public, are
/// range-checked in the packed buckets. A node without a rescale (`bits == 0`) proves the
/// arithmetic relation alone against its reduced output.
pub fn fused_rescale_plan<F: JoltField>(
    node: &ComputationNode,
    relation: OpRelation<F>,
    schedule: &mut ScheduleBuilder<F>,
) {
    schedule.reduce();
    if fused_rebase::rebase_bits(node).is_some_and(|bits| bits > 0) {
        schedule.open(OpOpen::RescaleRemainder(node.clone()));
    }
    schedule.relation(relation);
}

#[cfg(test)]
mod tests {
    use super::{OpRelation, RelId, ScheduleBuilder, Step};
    use crate::onnx_proof::range_bucket::RangeBucket;
    use atlas_algebra::field::fp128::Fp128;

    /// A relation that needs no node: an empty bucket's value relation.
    fn relation() -> OpRelation<Fp128> {
        OpRelation::RemainderBucketValue(RangeBucket {
            idx: 0,
            width: 8,
            log_t: 0,
            nodes: Vec::new(),
        })
    }

    fn sumcheck(members: Vec<RelId>) -> Step {
        Step::Sumcheck {
            members,
            batched: true,
        }
    }

    /// A builder holding one relation and no step.
    fn one_relation() -> (ScheduleBuilder<Fp128>, RelId) {
        let mut builder = ScheduleBuilder::new();
        builder.relations.push(relation());
        (builder, RelId(0))
    }

    #[test]
    fn schedule_accepts_a_setup_then_its_sumcheck() {
        let (mut builder, id) = one_relation();
        builder.steps = vec![Step::Setup(id), sumcheck(vec![id])];
        assert!(builder.finish(Vec::new()).is_ok());
    }

    #[test]
    fn schedule_rejects_a_double_setup() {
        let (mut builder, id) = one_relation();
        builder.steps = vec![Step::Setup(id), Step::Setup(id), sumcheck(vec![id])];
        assert!(builder.finish(Vec::new()).is_err());
    }

    #[test]
    fn schedule_rejects_a_relation_in_two_sumchecks() {
        let (mut builder, id) = one_relation();
        builder.steps = vec![sumcheck(vec![id]), sumcheck(vec![id])];
        assert!(builder.finish(Vec::new()).is_err());
    }

    #[test]
    fn schedule_rejects_a_setup_after_its_sumcheck() {
        let (mut builder, id) = one_relation();
        builder.steps = vec![sumcheck(vec![id]), Step::Setup(id)];
        assert!(builder.finish(Vec::new()).is_err());
    }

    #[test]
    fn schedule_rejects_an_empty_sumcheck_and_an_unverified_relation() {
        let (mut builder, _) = one_relation();
        builder.steps = vec![sumcheck(Vec::new())];
        assert!(builder.finish(Vec::new()).is_err());
        let (builder, _) = one_relation();
        assert!(builder.finish(Vec::new()).is_err());
    }
}
