use crate::onnx_proof::relation::Produces;
use crate::onnx_proof::relation::{Relation, SetupRelation};
use crate::onnx_proof::trace_view::TraceView;

use atlas_trace::{op::Sum, trace::LayerData, Model, Trace};

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
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial},
    },
    utils::math::Math,
};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_transcript::Transcript;
use common::{parallel::par_enabled, VirtualPoly};
use jolt_claims::{opening, SymbolicSumcheck};
use jolt_field::Ring;
use rayon::prelude::*;

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
    ) -> RelationSetup<F> {
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
    /// The operand folded at the output point along the kept axis, over the summed axis.
    fn witnesses(&self, trace: &Trace, setup: &RelationSetup<F>) -> Vec<MultilinearPolynomial<F>> {
        let LayerData { operands, .. } = TraceView::layer_data(trace, &self.shape.node);
        let operand = operands[0];
        let [m, n] = self.shape.config.operand_dims()[..] else {
            panic!("a sum normalizes to two dims")
        };
        let eq = EqPolynomial::<F>::evals(&setup.incoming_point);
        let folded: Vec<F> = match self.shape.config.axis() {
            SumAxis::Axis0 => {
                debug_assert_eq!(n.log_2(), setup.incoming_point.len());
                (0..m)
                    .into_par_iter()
                    .with_min_len(par_enabled())
                    .map(|h| {
                        (0..n)
                            .map(|j| F::from_i32(operand[h * n + j]) * eq[j])
                            .sum()
                    })
                    .collect()
            }
            SumAxis::Axis1 => {
                debug_assert_eq!(m.log_2(), setup.incoming_point.len());
                (0..n)
                    .into_par_iter()
                    .with_min_len(par_enabled())
                    .map(|j| {
                        (0..m)
                            .map(|h| F::from_i32(operand[h * n + j]) * eq[h])
                            .sum()
                    })
                    .collect()
            }
        };
        vec![MultilinearPolynomial::from(folded)]
    }

    fn binding_order(&self) -> BindingOrder {
        BindingOrder::HighToLow
    }
}
