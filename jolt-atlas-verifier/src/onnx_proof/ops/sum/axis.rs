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
use atlas_algebra::field::Ring;
use atlas_algebra::{field::JoltField, poly::multilinear::BindingOrder, utils::math::Math};
use atlas_claims::{opening, SymbolicSumcheck};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_transcript::Transcript;

/// A sum node with its normalized 2D shape and axis.
pub struct SumShape {
    node: ComputationNode,
    config: SumConfig,
    /// Length of the summed axis of the normalized operand.
    axis_len: usize,
}

/// The axis reduction of an exact sum as a dense relation, binding the summed axis
/// high-to-low over the operand folded at the output point `r`:
///
/// ```text
/// input claim   out(r)                 (the reduced output opening)
/// output claim  operand(ρ)             (ρ = (r′, r) along axis 0, (r, r′) along axis 1)
/// ```
///
/// **S ⊆ R.** The op computes `out = Σ_axis operand` exactly and its execution fails when a
/// sum leaves `i32`, so `out(r) = Σ_x operand(x, r)` (or `(r, x)`) on every execution that
/// exists.
///
/// **No wrap.** The operand's cells are `i32`, so a sum over an axis of `n` cells has
/// magnitude at most `n·2^31`, and `out` lies in `[−2^31, 2^31)` by its range claim (a model
/// output by being a public `i32` tensor). For any tensor that fits in memory (`n < 2^64`)
/// both sides are integers of magnitude below `2^95`, far below half the 128-bit field
/// modulus, so their equality in the field is their equality over the integers.
pub struct SumAxisRelation {
    shape: SumShape,
}

impl SumAxisRelation {
    /// The relation of a sum `node` of `model`. `Err` if the node is not a single-axis sum the
    /// reduction supports.
    pub fn for_node(model: &Model, node: ComputationNode) -> Result<Self, ProofVerifyError> {
        let config = utils::dims::sum_config(&node, model)?;
        let axis_len = config
            .operand_dims()
            .get(config.axis().axis_index())
            .copied()
            .ok_or_else(|| {
                ProofVerifyError::Malformed(format!(
                    "model: sum node {}: the summed axis is not an operand axis",
                    node.idx
                ))
            })?;
        Ok(Self::new(SumShape {
            node,
            config,
            axis_len,
        }))
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
        self.shape.axis_len.log_2()
    }

    fn degree(&self) -> usize {
        1
    }

    fn input_expression<F: Ring>(&self) -> RelationExpr<F> {
        opening(RelationOpening::ReducedNodeOutput(self.shape.node.idx))
    }

    fn output_expression<F: Ring>(&self) -> RelationExpr<F> {
        opening(RelationOpening::Id(self.ids().nodeio(Target::Input(0))))
    }
}

impl Produces for SumAxisRelation {
    fn produced(&self) -> Vec<OpeningId> {
        vec![self.ids().nodeio(Target::Input(0))]
    }
}

impl<F: JoltField> Relation<F> for SumAxisRelation {
    fn opening_points(
        &self,
        setup: &RelationSetup<F>,
        challenges: &[F],
    ) -> Result<Vec<Vec<F>>, ProofVerifyError> {
        let r = &setup.incoming_point;
        Ok(vec![match self.shape.config.axis() {
            SumAxis::Axis0 => [challenges, r.as_slice()].concat(),
            SumAxis::Axis1 => [r.as_slice(), challenges].concat(),
        }])
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
