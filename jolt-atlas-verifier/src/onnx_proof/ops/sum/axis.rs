use crate::onnx_proof::relation::Produces;
use crate::onnx_proof::relation::{Relation, SetupRelation};
use atlas_algebra::utils::errors::ProofVerifyError;

use atlas_trace::{op::Sum, Model};

use crate::node_view::ComputationNode;
use crate::{
    atlas_relation_types,
    onnx_proof::relation::{
        reduced_output_setup, AtlasRelation, EngineRelation, RelationExpr, RelationId,
        RelationOpening, RelationSetup,
    },
    utils::{
        self,
        dims::{SumAxis, SumConfig},
        opening_access::{OpeningIdBuilder, Target},
    },
};
use atlas_algebra::{
    field::JoltField, poly::multilinear_polynomial::BindingOrder, utils::math::Math,
};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_transcript::Transcript;
use common::VirtualPoly;
use jolt_claims::{opening, SymbolicSumcheck};
use jolt_field::Ring;

/// A sum node with its normalized 2D shape and axis.
pub struct SumShape {
    node: ComputationNode,
    config: SumConfig,
}

/// The axis reduction of a saturating sum as a dense relation, binding the summed axis
/// high-to-low over the operand folded at the output point `r`:
///
/// ```text
/// input claim   acc(r)                 (the clamp lookup's raf opening, `ClampAcc`)
/// output claim  operand(ρ)             (ρ = (r′, r) along axis 0, (r, r′) along axis 1)
/// ```
///
/// **S ⊆ R.** The sum accumulates `acc = Σ_axis operand` in `i64` before the clamp, and the
/// clamp lookup's `acc` is that accumulation, so `acc(r) = Σ_x operand(x, r)` (or `(r, x)`).
pub struct SumAxisRelation {
    shape: SumShape,
}

impl SumAxisRelation {
    /// The relation of a sum `node` of `model`.
    pub fn for_node(model: &Model, node: ComputationNode) -> Self {
        let config = utils::dims::sum_config(&node, model);
        Self::new(SumShape { node, config })
    }

    /// The sum's normalized 2D shape and axis.
    pub fn config(&self) -> &SumConfig {
        &self.shape.config
    }

    fn ids(&self) -> OpeningIdBuilder<'_> {
        OpeningIdBuilder::new(&self.shape.node)
    }
}

impl SymbolicSumcheck for SumAxisRelation {
    atlas_relation_types!();
    type Shape = SumShape;

    fn new(shape: SumShape) -> Self {
        Self { shape }
    }

    fn id() -> RelationId {
        RelationId::AxisSum
    }

    fn rounds(&self) -> usize {
        let config = &self.shape.config;
        config.operand_dims()[config.axis().axis_index()].log_2()
    }

    fn degree(&self) -> usize {
        1
    }

    fn input_expression<F: Ring>(&self) -> RelationExpr<F> {
        opening(RelationOpening::Id(
            self.ids().advice(VirtualPoly::ClampAcc),
        ))
    }

    fn output_expression<F: Ring>(&self) -> RelationExpr<F> {
        opening(RelationOpening::Id(self.produced()[0]))
    }
}

impl Produces for SumAxisRelation {
    fn produced(&self) -> Vec<OpeningId> {
        vec![self.ids().nodeio(Target::Input(0))]
    }
}

impl<F: JoltField> Relation<F> for SumAxisRelation {
    fn opening_points(&self, setup: &RelationSetup<F>, challenges: &[F]) -> Vec<Vec<F>> {
        let r = &setup.incoming_point;
        vec![match self.shape.config.axis() {
            SumAxis::Axis0 => [challenges, r.as_slice()].concat(),
            SumAxis::Axis1 => [r.as_slice(), challenges].concat(),
        }]
    }
}

impl<F: JoltField> SetupRelation<F> for SumAxisRelation {
    fn setup<T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut T,
    ) -> Result<RelationSetup<F>, ProofVerifyError> {
        reduced_output_setup(accumulator, &self.shape.node)
    }
}

impl AtlasRelation for SumAxisRelation {
    type Spec = Sum;

    fn node(&self) -> &ComputationNode {
        &self.shape.node
    }
}

impl<F: JoltField> EngineRelation<F> for SumAxisRelation {
    fn binding_order(&self) -> BindingOrder {
        BindingOrder::HighToLow
    }
}
