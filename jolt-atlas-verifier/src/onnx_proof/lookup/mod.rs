//! Table reads.
//!
//! A small public table (a clamped activation's, a smooth ReLU's correction, a trig op's) is
//! read through one-hot addresses by one relation, [`small_table`], described by the table's
//! [`SmallTableDef`].

use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::{activation_clamped, smooth_relu, trig};
use crate::onnx_proof::relation::{
    low_to_high_point, not_derived, Custom, OpExpr, OpRel, OpRelKind, RelationChallenge,
    RelationDerived, RelationOpening, RelationSetup, SetupOpen,
};
use crate::utils::{opening_access::OpeningIdBuilder, usize_to_n_bits};
use atlas_algebra::{
    field::JoltField,
    poly::{
        identity::IdentityPolynomial,
        multilinear::{MultilinearPolynomial, PolynomialEvaluation},
        signed_identity::SignedIdentityPoly,
    },
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_claims::{challenge, derived, opening};
use atlas_sumcheck::{
    config::{OneHotConfig, OneHotParams},
    opening_proof::{OpeningId, SumcheckId},
    shout::RaOneHotEncoding,
};
use atlas_trace::{semantics::tables::TableKind, Op, SemVersion};
use common::{consts::MODEL_SCALE, CommittedPoly, VirtualPoly};

/// A small public table and how a node reads it.
#[derive(Clone, Copy)]
pub struct SmallTableDef {
    /// The semantics table read.
    pub kind: TableKind,
    /// Address variables of the table.
    pub vars: usize,
    /// Whether addresses are two's complement: the table is indexed by signed value and the
    /// address identity is signed.
    pub signed: bool,
    /// The advice polynomial of the read's one-hot addresses.
    pub ra: fn(usize) -> VirtualPoly,
    /// Chunk `d` of the committed one-hot addresses.
    pub ra_chunk: fn(usize, usize) -> CommittedPoly,
    /// The advice polynomial of the addresses read, opened at the node's output point.
    pub address: fn(usize) -> VirtualPoly,
    /// Whether the read's setup appends the address opening, after its draw; otherwise an
    /// earlier step of the plan has.
    pub opens_address: bool,
    /// The value the table read is claimed to equal at the node's output point.
    pub value: fn(&ComputationNode) -> OpExpr,
}

/// The node's reduced output: the value of a read whose table holds the whole output.
pub fn output_value(node: &ComputationNode) -> OpExpr {
    opening(RelationOpening::ReducedNodeOutput(node.idx))
}

impl SmallTableDef {
    /// The small table `node` reads. `Err` if its operator reads none.
    pub fn for_node(node: &ComputationNode) -> Result<Self, ProofVerifyError> {
        Ok(match &node.operator {
            Op::Erf(_) => activation_clamped::table_def(TableKind::Erf),
            Op::Sigmoid(_) => activation_clamped::table_def(TableKind::Sigmoid),
            Op::Tanh(_) => activation_clamped::table_def(TableKind::Tanh),
            Op::SmoothRelu(op) => smooth_relu::table_def(op.form),
            Op::Cos(_) => trig::cos_table(),
            Op::Sin(_) => trig::sin_table(),
            other => {
                return Err(ProofVerifyError::Malformed(format!(
                    "model: node {} is {other}, which reads no small table",
                    node.idx
                )))
            }
        })
    }

    /// The table's values over its `2^vars` addresses, from the deterministic integer tables
    /// of `atlas_trace::semantics` at the model scale: a signed table by the two's-complement
    /// value of each address, an unsigned one by the address. `Err` if the scale is outside
    /// the semantics' range or the semantics' table has fewer entries than addresses.
    pub fn materialize(&self) -> Result<Vec<i32>, ProofVerifyError> {
        let scale = atlas_trace::Scale::new(MODEL_SCALE as u8)
            .map_err(|e| ProofVerifyError::Malformed(format!("model: table scale: {e}")))?;
        let table = atlas_trace::semantics::tables::table(self.kind, scale, SemVersion::V1);
        let size = 1usize << self.vars;
        if self.signed {
            return Ok((0..size)
                .map(|address| table.lookup(usize_to_n_bits(address, self.vars)))
                .collect());
        }
        table
            .entries
            .get(..size)
            .map(<[i32]>::to_vec)
            .ok_or_else(|| {
                ProofVerifyError::Malformed(format!(
                    "the {:?} table has {} entries, fewer than its {size} addresses",
                    self.kind,
                    table.entries.len(),
                ))
            })
    }

    /// The opening of the addresses `node` reads.
    pub fn address_opening(&self, node: &ComputationNode) -> OpeningId {
        OpeningIdBuilder::new(node).advice(self.address)
    }

    /// The opening of `node`'s one-hot read addresses.
    pub fn ra_opening(&self, node: &ComputationNode) -> OpeningId {
        OpeningIdBuilder::new(node).advice(self.ra)
    }
}

/// A small-table lookup:
///
/// ```text
/// input claim   value(r_out) + γ·address(r_out)        (the read values, the read addresses)
/// output claim  ra(ρ ‖ r_out) · (table(ρ) + γ·id(ρ))  (ρ the low-to-high bound address point)
/// ```
///
/// with `id` the signed identity for two's-complement addresses. `Err` if the node's
/// operator reads no small table.
///
/// **S ⊆ R.** The values ([`SmallTableDef::value`], the op's output unless the table holds
/// only a part of it) are the table read at the addresses, so an honest one-hot `ra` reads
/// them and, through the identity, the addresses. The setup takes the reduced output point as
/// its incoming point, draws `γ`, and appends the address opening there when the table's
/// definition says so.
pub fn small_table<F: JoltField>(node: &ComputationNode) -> Result<OpRel<F>, ProofVerifyError> {
    let def = SmallTableDef::for_node(node)?;
    let ra = def.ra_opening(node);
    let address = def.address_opening(node);
    Ok(OpRel {
        rounds: def.vars,
        degree: 2,
        input: (def.value)(node)
            + challenge(RelationChallenge::Gamma) * opening(RelationOpening::Id(address)),
        output: opening(RelationOpening::Id(ra))
            * (derived(RelationDerived::Table(0))
                + challenge(RelationChallenge::Gamma) * derived(RelationDerived::Identity)),
        produced: vec![ra],
        draws: vec![RelationChallenge::Gamma],
        setup_opens: if def.opens_address {
            vec![SetupOpen {
                id: address,
                range: 0..node.pow2_padded_num_output_elements().log_2(),
            }]
        } else {
            Vec::new()
        },
        custom: Custom {
            points: Some(address_then_output_point),
            derive: Some(table_and_identity),
            ..Custom::default()
        },
        ..OpRel::base(OpRelKind::SmallTable, node)
    })
}

/// The opening point of a [`small_table`]'s one-hot addresses: the bound address point, then
/// the output point.
fn address_then_output_point<F: JoltField>(
    _relation: &OpRel<F>,
    setup: &RelationSetup<F>,
    challenges: &[F],
) -> Result<Vec<Vec<F>>, ProofVerifyError> {
    Ok(vec![[
        low_to_high_point(challenges),
        setup.incoming_point.clone(),
    ]
    .concat()])
}

/// The derived values of a [`small_table`]: the table's extension and the address identity
/// (signed for two's-complement addresses) at the bound address point.
fn table_and_identity<F: JoltField>(
    relation: &OpRel<F>,
    id: &RelationDerived,
    _setup: &RelationSetup<F>,
    challenges: &[F],
) -> Result<F, ProofVerifyError> {
    let def = SmallTableDef::for_node(&relation.node)?;
    let point = low_to_high_point(challenges);
    Ok(match id {
        RelationDerived::Table(0) => {
            MultilinearPolynomial::<F>::from(def.materialize()?).evaluate(&point)
        }
        RelationDerived::Identity if def.signed => {
            SignedIdentityPoly::new(def.vars).evaluate(&point)
        }
        RelationDerived::Identity => IdentityPolynomial::new(def.vars).evaluate(&point),
        other => return Err(not_derived(other)),
    })
}

/// One-hot encoding of a small table's read addresses.
pub struct SmallTableRaEncoding {
    node_idx: usize,
    def: SmallTableDef,
}

impl SmallTableRaEncoding {
    /// The encoding of `node`'s small-table read. `Err` if its operator reads no small table.
    pub fn for_node(node: &ComputationNode) -> Result<Self, ProofVerifyError> {
        Ok(Self {
            node_idx: node.idx,
            def: SmallTableDef::for_node(node)?,
        })
    }
}

impl RaOneHotEncoding for SmallTableRaEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPoly {
        (self.def.ra_chunk)(self.node_idx, d)
    }

    fn r_cycle_source(&self) -> OpeningId {
        OpeningId::new(
            (self.def.address)(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn ra_source(&self) -> OpeningId {
        OpeningId::new(
            (self.def.ra)(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn log_k(&self) -> usize {
        self.def.vars
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), self.def.vars)
    }
}
