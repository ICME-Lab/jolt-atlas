//! A lookup into a small public table (a teleported Sin or Cos, a clamped activation): a
//! read-raf check of the looked-up addresses against the op's output, as a relation.

use super::{
    cos::{CosTable, COS_TABLE_VARS},
    sin::{SinTable, SIN_TABLE_VARS},
};
use crate::node_view::ComputationNode;
use crate::onnx_proof::relation::{
    low_to_high_point, Produces, Relation, RelationChallenge, RelationDerived, RelationExpr,
    RelationId, RelationOpening, RelationSetup, SetupRelation,
};
use crate::utils::opening_access::{AccOpeningAccessor, OpeningIdBuilder};
use atlas_algebra::{
    field::JoltField,
    poly::{
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        signed_identity_poly::SignedIdentityPoly,
    },
};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_transcript::Transcript;
use common::VirtualPoly;
use jolt_claims::{challenge, derived, opening, SymbolicSumcheck};
use jolt_field::Ring;
use std::marker::PhantomData;

/// A small public lookup table and where an op reads it.
pub trait SmallTable: Send + Sync + 'static {
    /// Address variables of the table.
    const VARS: usize;
    /// The advice polynomial of the lookup's one-hot read addresses.
    const RA: fn(usize) -> VirtualPoly;
    /// Whether addresses are two's complement, so the address identity is signed.
    const SIGNED: bool;
    /// The table's values.
    fn materialize() -> Vec<i32>;
    /// The opening of the looked-up addresses at the node's output point.
    fn address(node: &ComputationNode) -> OpeningId;
}

impl SmallTable for CosTable {
    const VARS: usize = COS_TABLE_VARS;
    const RA: fn(usize) -> VirtualPoly = VirtualPoly::CosRa;
    const SIGNED: bool = false;
    fn materialize() -> Vec<i32> {
        CosTable::materialize()
    }
    fn address(node: &ComputationNode) -> OpeningId {
        OpeningIdBuilder::new(node).advice(VirtualPoly::TrigDownscaled)
    }
}

impl SmallTable for SinTable {
    const VARS: usize = SIN_TABLE_VARS;
    const RA: fn(usize) -> VirtualPoly = VirtualPoly::SinRa;
    const SIGNED: bool = false;
    fn materialize() -> Vec<i32> {
        SinTable::materialize()
    }
    fn address(node: &ComputationNode) -> OpeningId {
        OpeningIdBuilder::new(node).advice(VirtualPoly::TrigDownscaled)
    }
}

/// A small-table lookup:
///
/// ```text
/// input claim   out(r_out) + γ·address(r_out)          (the reduced output, the read addresses)
/// output claim  ra(ρ ‖ r_out) · (table(ρ) + γ·id(ρ))  (ρ the low-to-high bound address point)
/// ```
///
/// with `id` the signed identity for two's-complement addresses.
///
/// **S ⊆ R.** The op's output is the table read at the addresses, so an honest one-hot `ra`
/// reads the output and, through the identity, the addresses. The setup draws `γ` and takes the
/// reduced output point as its incoming point.
pub struct SmallTableRelation<S> {
    node: ComputationNode,
    _table: PhantomData<fn() -> S>,
}

impl<S: SmallTable> SmallTableRelation<S> {
    /// The trig node.
    pub fn node(&self) -> &ComputationNode {
        &self.node
    }

    fn ra(&self) -> OpeningId {
        OpeningIdBuilder::new(&self.node).advice(S::RA)
    }
}

impl<S: SmallTable> SymbolicSumcheck for SmallTableRelation<S> {
    crate::atlas_relation_types!();
    type Shape = ComputationNode;

    fn new(node: ComputationNode) -> Self {
        Self {
            node,
            _table: PhantomData,
        }
    }

    fn id() -> RelationId {
        RelationId::SmallTable
    }

    fn rounds(&self) -> usize {
        S::VARS
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_expression<G: Ring>(&self) -> RelationExpr<G> {
        opening::<G, _, _, _>(RelationOpening::ReducedNodeOutput(self.node.idx))
            + challenge::<G, _, _, _>(RelationChallenge::Gamma)
                * opening(RelationOpening::Id(S::address(&self.node)))
    }

    fn output_expression<G: Ring>(&self) -> RelationExpr<G> {
        opening::<G, _, _, _>(RelationOpening::Id(self.ra()))
            * (derived::<G, _, _, _>(RelationDerived::Table(0))
                + challenge::<G, _, _, _>(RelationChallenge::Gamma)
                    * derived(RelationDerived::Identity))
    }
}

impl<S: SmallTable> Produces for SmallTableRelation<S> {
    fn produced(&self) -> Vec<OpeningId> {
        vec![self.ra()]
    }
}

impl<F: JoltField, S: SmallTable> Relation<F> for SmallTableRelation<S> {
    fn opening_points(&self, setup: &RelationSetup<F>, challenges: &[F]) -> Vec<Vec<F>> {
        vec![[low_to_high_point(challenges), setup.incoming_point.clone()].concat()]
    }

    fn derive(&self, id: &RelationDerived, _setup: &RelationSetup<F>, challenges: &[F]) -> F {
        let point = low_to_high_point(challenges);
        match id {
            RelationDerived::Table(0) => {
                MultilinearPolynomial::<F>::from(S::materialize()).evaluate(&point)
            }
            RelationDerived::Identity if S::SIGNED => {
                SignedIdentityPoly::new(S::VARS).evaluate(&point)
            }
            RelationDerived::Identity => IdentityPolynomial::new(S::VARS).evaluate(&point),
            other => panic!("{other:?} is not derived by this relation"),
        }
    }
}

impl<F: JoltField, S: SmallTable> SetupRelation<F> for SmallTableRelation<S> {
    fn setup<T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut T,
    ) -> RelationSetup<F> {
        let gamma = transcript.challenge_scalar();
        let (r_node_output, _) =
            AccOpeningAccessor::new(accumulator, &self.node).get_reduced_opening();
        RelationSetup {
            incoming_point: r_node_output.r,
            challenges: vec![(RelationChallenge::Gamma, gamma)],
        }
    }
}
