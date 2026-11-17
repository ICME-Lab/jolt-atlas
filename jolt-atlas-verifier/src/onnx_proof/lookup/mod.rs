//! Table reads.
//!
//! A small public table (a clamped activation's, a smooth ReLU's correction, a trig op's) is
//! read through one-hot addresses by one relation, [`small_table`], described by the table's
//! [`SmallTableDef`]. A prefix-suffix table of `atlas-lookup-tables` is read by a
//! [`TableRead`], verified by [`table_read_verifier`]. The one-hot checks of any read's
//! addresses start from its [`RaEncoding`].

use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::{activation_clamped, smooth_relu, trig};
use crate::onnx_proof::relation::{
    low_to_high_point, not_derived, Custom, OpExpr, OpRel, OpRelKind, RelationChallenge,
    RelationDerived, RelationOpening, RelationSetup, SetupOpen,
};
use crate::utils::{
    opening_access::{OpeningIdBuilder, Target},
    usize_to_n_bits,
};
use atlas_algebra::{
    field::JoltField,
    poly::{
        identity::IdentityPolynomial,
        multilinear::{MultilinearPolynomial, PolynomialEvaluation},
        point::{OpeningPoint, BIG_ENDIAN},
        signed_identity::SignedIdentityPoly,
    },
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_claims::{challenge, derived, opening};
use atlas_lookup_tables::{
    clamp::{ActivationClampTable, ClampTable, SiluClampTable, SoftmaxClampTable},
    relu::ReluTable,
    right_shift::RightShiftTable,
    JoltLookupTable, PrefixSuffixDecompositionTrait,
};
use atlas_sumcheck::{
    config::{OneHotConfig, OneHotParams},
    opening_proof::{
        OpeningAccumulator, OpeningId, PolynomialId, SumcheckId, VerifierOpeningAccumulator,
    },
    ps_shout::{ps_read_raf_verifier, PrefixSuffixShoutProvider, RafShoutProvider, ReadRafClaims},
    relation::{resolve, DynRelationVerifier},
    shout::RaOneHotEncoding,
};
use atlas_trace::{semantics::tables::TableKind, Op, SemVersion};
use atlas_transcript::Transcript;
use common::{
    consts::{MODEL_SCALE, XLEN},
    CommittedPoly, VirtualPoly,
};

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

    /// The one-hot encoding of `node`'s read addresses: the cycle point is the address
    /// opening's.
    pub fn ra_encoding(&self, node: &ComputationNode) -> RaEncoding {
        RaEncoding::chunked(
            node.idx,
            self.ra_chunk,
            self.ra_opening(node),
            self.address_opening(node),
            self.vars,
        )
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

/// The one-hot encoding of a read's addresses: the committed chunks, and the openings the
/// one-hot checks start from.
#[derive(Clone, Debug)]
pub struct RaEncoding {
    /// The node whose read the addresses are.
    pub node: usize,
    /// Chunk `d` of the committed one-hot addresses of node `node`.
    pub committed: fn(usize, usize) -> CommittedPoly,
    /// The opening of the virtual read-address polynomial, at `address ‖ cycle`.
    pub ra_source: OpeningId,
    /// The opening whose point is the read's cycle point.
    pub r_cycle_source: OpeningId,
    /// Address variables: where the read-address opening's point splits.
    pub log_k: usize,
    /// The chunking of the addresses.
    pub params: OneHotParams,
}

impl RaEncoding {
    /// The encoding of `log_k` address variables in the default chunking.
    pub fn chunked(
        node: usize,
        committed: fn(usize, usize) -> CommittedPoly,
        ra_source: OpeningId,
        r_cycle_source: OpeningId,
        log_k: usize,
    ) -> Self {
        Self {
            node,
            committed,
            ra_source,
            r_cycle_source,
            log_k,
            params: OneHotParams::from_config_and_log_K(&OneHotConfig::default(), log_k),
        }
    }

    /// The committed chunks of the addresses, in chunk order.
    pub fn committed_polys(&self) -> Vec<CommittedPoly> {
        (0..self.params.instruction_d)
            .map(|d| (self.committed)(self.node, d))
            .collect()
    }
}

impl RaOneHotEncoding for RaEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPoly {
        (self.committed)(self.node, d)
    }

    fn r_cycle_source(&self) -> OpeningId {
        self.r_cycle_source
    }

    fn ra_source(&self) -> OpeningId {
        self.ra_source
    }

    fn log_k(&self) -> usize {
        self.log_k
    }

    fn one_hot_params(&self) -> OneHotParams {
        self.params.clone()
    }
}

/// The `atlas-lookup-tables` tables the verifier reads through a prefix-suffix read-raf.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrefixSuffixTable {
    /// `ReLU`.
    Relu,
    /// `Clamp`'s symmetric clamp.
    Clamp,
    /// A clamped activation's input clamp.
    ActivationClamp,
    /// SiLU's input clamp, twice as wide as the clamped activations'.
    SiluClamp,
    /// A trig op's downscale of its period remainder.
    RightShift,
    /// Softmax's clamp of the address `u = c − x` to the tables' addresses.
    SoftmaxClamp,
}

/// A prefix-suffix read: `value = table[operand]` at the cycle point, through one-hot
/// addresses over `log_k` address variables.
#[derive(Clone, Debug)]
pub struct TableRead {
    /// The node the read belongs to.
    pub node: ComputationNode,
    /// The table read.
    pub table: PrefixSuffixTable,
    /// The value read, an opening held at the cycle point.
    pub value: RelationOpening,
    /// The operand the table is read at: its opening is appended at the cycle point when the
    /// read is set up.
    pub operand: OpeningId,
    /// The opening whose point is the read's cycle point.
    pub cycle: RelationOpening,
    /// The one-hot encoding of the read's addresses.
    pub ra: RaEncoding,
    /// The virtual read-address polynomial [`RaEncoding::ra_source`] opens.
    pub address: VirtualPoly,
    /// Address variables of the table.
    pub log_k: usize,
}

impl TableRead {
    /// A read of `node`'s first operand at the node's reduced output point: `value` is
    /// `table[operand]` there, the addresses `ra` over the committed `committed` chunks.
    pub fn of_operand(
        node: &ComputationNode,
        table: PrefixSuffixTable,
        value: RelationOpening,
        ra: fn(usize) -> VirtualPoly,
        committed: fn(usize, usize) -> CommittedPoly,
    ) -> Self {
        let ids = OpeningIdBuilder::new(node);
        Self::new(
            node,
            table,
            value,
            ids.nodeio(Target::Input(0)),
            RelationOpening::ReducedNodeOutput(node.idx),
            ids.nodeio(Target::Current),
            ra,
            committed,
        )
    }

    /// A read of `operand` at the point of `cycle`, which `r_cycle_source` names as an
    /// accumulator opening for the one-hot checks.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        node: &ComputationNode,
        table: PrefixSuffixTable,
        value: RelationOpening,
        operand: OpeningId,
        cycle: RelationOpening,
        r_cycle_source: OpeningId,
        ra: fn(usize) -> VirtualPoly,
        committed: fn(usize, usize) -> CommittedPoly,
    ) -> Self {
        let address = ra(node.idx);
        Self {
            node: node.clone(),
            table,
            value,
            operand,
            cycle,
            address,
            ra: RaEncoding {
                node: node.idx,
                committed,
                ra_source: OpeningId::new(address, SumcheckId::NodeExecution(node.idx)),
                r_cycle_source,
                log_k: XLEN,
                params: OneHotParams::new(node.pow2_padded_num_output_elements().log_2(), XLEN),
            },
            log_k: XLEN,
        }
    }

    /// The read's cycle point. `Err` if the accumulator lacks the opening that fixes it.
    pub fn cycle_point<F: JoltField>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Result<OpeningPoint<BIG_ENDIAN, F>, ProofVerifyError> {
        Ok(match self.cycle {
            RelationOpening::ReducedNodeOutput(node) => {
                accumulator.get_node_output_opening(node)?.0
            }
            RelationOpening::Id(id) => match id.polynomial {
                PolynomialId::Virtual(_) => accumulator.get_virtual_polynomial_opening(id)?.0,
                PolynomialId::Committed(_) => accumulator.get_committed_polynomial_opening(id)?.0,
            },
        })
    }
}

impl<F: JoltField> RafShoutProvider<F> for TableRead {
    fn r_cycle(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Result<OpeningPoint<BIG_ENDIAN, F>, ProofVerifyError> {
        self.cycle_point(accumulator)
    }

    fn ra_poly(&self) -> (VirtualPoly, SumcheckId) {
        (self.address, self.ra.ra_source.sumcheck)
    }
}

impl<F, LUT, const LOG_K: usize> PrefixSuffixShoutProvider<F, LUT, LOG_K> for TableRead
where
    F: JoltField,
    LUT: JoltLookupTable + PrefixSuffixDecompositionTrait<LOG_K> + Default,
{
    fn read_raf_claims(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Result<ReadRafClaims<F>, ProofVerifyError> {
        Ok(ReadRafClaims {
            rv_claim: resolve(accumulator, &self.value)?,
            operand_claim: accumulator.get_virtual_polynomial_opening(self.operand)?.1,
        })
    }
}

/// The verifier of a prefix-suffix read: the operand's opening appended at the cycle point,
/// then the read-raf relation of the read's table set up. `Err` if the accumulator lacks a
/// claim the read reads.
pub fn table_read_verifier<F: JoltField, T: Transcript>(
    read: &TableRead,
    accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
) -> Result<Box<dyn DynRelationVerifier<F, T>>, ProofVerifyError> {
    let r_cycle = read.cycle_point(&*accumulator)?;
    accumulator.append_virtual(transcript, read.operand, r_cycle)?;
    Ok(match read.table {
        PrefixSuffixTable::Relu => Box::new(ps_read_raf_verifier::<F, T, ReluTable<XLEN>, XLEN>(
            read,
            accumulator,
            transcript,
        )?),
        PrefixSuffixTable::Clamp => Box::new(ps_read_raf_verifier::<F, T, ClampTable<XLEN>, XLEN>(
            read,
            accumulator,
            transcript,
        )?),
        PrefixSuffixTable::ActivationClamp => Box::new(ps_read_raf_verifier::<
            F,
            T,
            ActivationClampTable<XLEN>,
            XLEN,
        >(read, accumulator, transcript)?),
        PrefixSuffixTable::SiluClamp => Box::new(ps_read_raf_verifier::<
            F,
            T,
            SiluClampTable<XLEN>,
            XLEN,
        >(read, accumulator, transcript)?),
        PrefixSuffixTable::RightShift => Box::new(ps_read_raf_verifier::<
            F,
            T,
            RightShiftTable<XLEN>,
            XLEN,
        >(read, accumulator, transcript)?),
        PrefixSuffixTable::SoftmaxClamp => Box::new(ps_read_raf_verifier::<
            F,
            T,
            SoftmaxClampTable<XLEN>,
            XLEN,
        >(read, accumulator, transcript)?),
    })
}
