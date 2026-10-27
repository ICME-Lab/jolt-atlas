//! The execution check of a teleported trigonometric op (Sin, Cos): a read-raf lookup of the
//! downscaled input into the op's public table, as a relation.

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
    },
};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_transcript::Transcript;
use common::VirtualPoly;
use jolt_claims::{challenge, derived, opening, SymbolicSumcheck};
use jolt_field::Ring;
use std::marker::PhantomData;

/// A teleported trigonometric op's public table.
pub trait TrigTable: Send + Sync + 'static {
    /// Address variables of the table.
    const VARS: usize;
    /// The advice polynomial of the lookup's one-hot read addresses.
    const RA: fn(usize) -> VirtualPoly;
    /// The table's values.
    fn materialize() -> Vec<i32>;
}

impl TrigTable for CosTable {
    const VARS: usize = COS_TABLE_VARS;
    const RA: fn(usize) -> VirtualPoly = VirtualPoly::CosRa;
    fn materialize() -> Vec<i32> {
        CosTable::materialize()
    }
}

impl TrigTable for SinTable {
    const VARS: usize = SIN_TABLE_VARS;
    const RA: fn(usize) -> VirtualPoly = VirtualPoly::SinRa;
    fn materialize() -> Vec<i32> {
        SinTable::materialize()
    }
}

/// The trigonometric table lookup of a teleported Sin or Cos:
///
/// ```text
/// input claim   out(r_out) + γ·downscaled(r_out)      (the reduced output, the downscaled input)
/// output claim  ra(ρ) · (table(ρ) + γ·id(ρ))         (ρ the low-to-high bound address point)
/// ```
///
/// **S ⊆ R.** The op's output is the table read at the downscaled input, so an honest one-hot
/// `ra` reads the output and the address. The setup draws `γ` and takes the reduced output point
/// as its incoming point; `ra` opens at `ρ ‖ r_out`.
pub struct TrigTableRelation<S> {
    node: ComputationNode,
    _table: PhantomData<fn() -> S>,
}

impl<S: TrigTable> TrigTableRelation<S> {
    fn ra(&self) -> OpeningId {
        OpeningIdBuilder::new(&self.node).advice(S::RA)
    }

    fn downscaled(&self) -> OpeningId {
        OpeningIdBuilder::new(&self.node).advice(VirtualPoly::TrigDownscaled)
    }
}

impl<S: TrigTable> SymbolicSumcheck for TrigTableRelation<S> {
    crate::atlas_relation_types!();
    type Shape = ComputationNode;

    fn new(node: ComputationNode) -> Self {
        Self {
            node,
            _table: PhantomData,
        }
    }

    fn id() -> RelationId {
        RelationId::TrigTable
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
                * opening(RelationOpening::Id(self.downscaled()))
    }

    fn output_expression<G: Ring>(&self) -> RelationExpr<G> {
        opening::<G, _, _, _>(RelationOpening::Id(self.ra()))
            * (derived::<G, _, _, _>(RelationDerived::Table(0))
                + challenge::<G, _, _, _>(RelationChallenge::Gamma)
                    * derived(RelationDerived::Identity))
    }
}

impl<S: TrigTable> Produces for TrigTableRelation<S> {
    fn produced(&self) -> Vec<OpeningId> {
        vec![self.ra()]
    }
}

impl<F: JoltField, S: TrigTable> Relation<F> for TrigTableRelation<S> {
    fn opening_points(&self, setup: &RelationSetup<F>, challenges: &[F]) -> Vec<Vec<F>> {
        vec![[low_to_high_point(challenges), setup.incoming_point.clone()].concat()]
    }

    fn derive(&self, id: &RelationDerived, _setup: &RelationSetup<F>, challenges: &[F]) -> F {
        let point = low_to_high_point(challenges);
        match id {
            RelationDerived::Table(0) => {
                MultilinearPolynomial::<F>::from(S::materialize()).evaluate(&point)
            }
            RelationDerived::Identity => IdentityPolynomial::new(S::VARS).evaluate(&point),
            other => panic!("{other:?} is not derived by this relation"),
        }
    }
}

impl<F: JoltField, S: TrigTable> SetupRelation<F> for TrigTableRelation<S> {
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
