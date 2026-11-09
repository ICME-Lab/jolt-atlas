use crate::node_view::{ComputationNode, ModelView};
use crate::onnx_proof::plan::{OpPlan, OpRelation, Plan, Step};
use crate::onnx_proof::relation::Produces;
use crate::onnx_proof::relation::{Relation, SetupRelation};
use crate::utils::{
    dims::{coord_to_linear, linear_to_coord},
    opening_access::{OpeningIdBuilder, Target},
};
use crate::{
    atlas_relation_types,
    onnx_proof::{
        relation::{
            reduced_output_setup, AtlasRelation, EngineRelation, RelationDerived, RelationExpr,
            RelationId, RelationOpening, RelationSetup,
        },
        ProofType,
    },
};
use atlas_algebra::{
    field::JoltField,
    poly::eq_poly::EqPolynomial,
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_trace::{op::Slice, Model, Op};
use atlas_transcript::Transcript;
use jolt_claims::{derived, opening, SymbolicSumcheck};
use jolt_field::Ring;

crate::impl_plan_op!(Slice, SliceRelation);

impl OpPlan for Slice {
    fn plan(
        &self,
        node: &ComputationNode,
        model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        Ok(Plan(vec![
            Step::Reduce,
            Step::Relation {
                proof: ProofType::Execution,
                relation: OpRelation::Slice(SliceRelation::for_node(model, node.clone())?),
            },
        ]))
    }
}

/// A slice node with its window and its operand's unpadded dims, validated against the slice
/// shape rules.
pub struct SliceShape {
    node: ComputationNode,
    input_raw_dims: Vec<usize>,
    axis: usize,
    start: usize,
}

fn unsupported(what: impl std::fmt::Display) -> ProofVerifyError {
    ProofVerifyError::Malformed(format!("model: slice: {what}"))
}

/// `Slice` as a dense relation over the padded input domain:
///
/// ```text
/// input claim   out(r₀)                (the reduced output opening)
/// output claim  selector(r′) · in(r′)
/// ```
///
/// where `selector` places `eq(r₀, ·)` at the output cell of each input element inside the
/// window and zero outside it.
///
/// **S ⊆ R.** A slice copies the window `[start, end)` of the axis unchanged, so
/// `out(r₀) = Σ_i selector(i)·in(i)` over the input domain.
pub struct SliceRelation {
    shape: SliceShape,
}

impl SliceRelation {
    /// The relation of a slice `node` of `model`. `Err` if the node is not a slice or its dims
    /// break the slice shape rules.
    pub fn for_node(model: &Model, node: ComputationNode) -> Result<Self, ProofVerifyError> {
        let Op::Slice(op) = &node.operator else {
            return Err(unsupported("expected a Slice operator"));
        };
        let (axis, start, end) = (op.axis, op.start, op.end);
        let input = node
            .inputs
            .first()
            .copied()
            .filter(|&input| input < model.num_nodes())
            .ok_or_else(|| unsupported("the operand is not a node"))?;
        let input_raw_dims = model.view(input).output_dims;
        validate_slice_shapes(&input_raw_dims, &node.output_dims, axis, start, end)?;
        Ok(Self::new(SliceShape {
            node,
            input_raw_dims,
            axis,
            start,
        }))
    }

    fn input(&self) -> OpeningId {
        OpeningIdBuilder::new(&self.shape.node).nodeio(Target::Input(0))
    }
}

impl SymbolicSumcheck for SliceRelation {
    atlas_relation_types!();
    type Shape = SliceShape;

    fn new(shape: SliceShape) -> Self {
        Self { shape }
    }

    fn id() -> RelationId {
        RelationId::Slice
    }

    fn rounds(&self) -> usize {
        self.shape
            .input_raw_dims
            .iter()
            .map(|dim| dim.next_power_of_two())
            .product::<usize>()
            .log_2()
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_expression<F: Ring>(&self) -> RelationExpr<F> {
        opening(RelationOpening::ReducedNodeOutput(self.shape.node.idx))
    }

    fn output_expression<F: Ring>(&self) -> RelationExpr<F> {
        derived(RelationDerived::Table(0)) * opening(RelationOpening::Id(self.input()))
    }
}

impl Produces for SliceRelation {
    fn produced(&self) -> Vec<OpeningId> {
        vec![self.input()]
    }
}

impl<F: JoltField> Relation<F> for SliceRelation {
    fn table(&self, index: usize, setup: &RelationSetup<F>) -> Result<Vec<F>, ProofVerifyError> {
        if index != 0 {
            return Err(unsupported(format!(
                "a slice has one selector, no table {index}"
            )));
        }
        build_slice_selector(
            &self.shape.input_raw_dims,
            &self.shape.node.output_dims,
            self.shape.axis,
            self.shape.start,
            &setup.incoming_point,
        )
    }
}

impl<F: JoltField> SetupRelation<F> for SliceRelation {
    fn setup<T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut T,
    ) -> Result<RelationSetup<F>, ProofVerifyError> {
        reduced_output_setup(accumulator, &self.shape.node)
    }
}

impl AtlasRelation for SliceRelation {
    type Spec = Slice;

    fn node(&self) -> &ComputationNode {
        &self.shape.node
    }
}

impl<F: JoltField> EngineRelation<F> for SliceRelation {}

fn build_slice_selector<F: JoltField>(
    input_raw_dims: &[usize],
    output_raw_dims: &[usize],
    axis: usize,
    start: usize,
    r_output: &[F],
) -> Result<Vec<F>, ProofVerifyError> {
    let input_raw_len: usize = input_raw_dims.iter().product();
    let input_padded_dims: Vec<usize> = input_raw_dims
        .iter()
        .map(|dim| dim.next_power_of_two())
        .collect();
    let output_padded_dims: Vec<usize> = output_raw_dims
        .iter()
        .map(|dim| dim.next_power_of_two())
        .collect();
    let input_domain_len: usize = input_padded_dims.iter().product();
    let output_domain_len: usize = output_padded_dims.iter().product();
    let output_num_vars = output_domain_len.log_2();
    if output_num_vars != r_output.len() {
        return Err(ProofVerifyError::Malformed(format!(
            "slice selector: output point has {} coordinates, its padded domain {output_num_vars}",
            r_output.len()
        )));
    }

    let mut selector = vec![F::zero(); input_domain_len];
    let eq_evals = EqPolynomial::evals(r_output);
    for output_linear_idx in 0..output_raw_dims.iter().product() {
        let output_coord = linear_to_coord(output_linear_idx, output_raw_dims);
        let mut input_coord = output_coord.clone();
        *input_coord
            .get_mut(axis)
            .ok_or_else(|| unsupported(format!("axis {axis} out of range")))? += start;

        let input_index = coord_to_linear(&input_coord, &input_padded_dims)?;
        let output_index = coord_to_linear(&output_coord, &output_padded_dims)?;
        let (Some(cell), Some(eq)) = (selector.get_mut(input_index), eq_evals.get(output_index))
        else {
            return Err(unsupported("an element outside the padded domain"));
        };
        *cell = *eq;
    }

    if input_raw_len < output_raw_dims.iter().product() {
        return Err(unsupported("output raw length exceeds input raw length"));
    }

    Ok(selector)
}

fn validate_slice_shapes(
    input_dims: &[usize],
    output_dims: &[usize],
    axis: usize,
    start: usize,
    end: usize,
) -> Result<(), ProofVerifyError> {
    if input_dims.len() != output_dims.len() {
        return Err(unsupported("rank mismatch"));
    }
    let axis_dim = *input_dims
        .get(axis)
        .ok_or_else(|| unsupported("axis out of range"))?;
    if start > end {
        return Err(unsupported("start must be <= end"));
    }
    if end > axis_dim {
        return Err(unsupported("end out of bounds"));
    }

    for (dim, (&input_dim, &output_dim)) in input_dims.iter().zip(output_dims).enumerate() {
        let expected = if dim == axis { end - start } else { input_dim };
        if output_dim != expected {
            return Err(unsupported(format!("output shape mismatch at axis {dim}")));
        }
    }
    Ok(())
}
