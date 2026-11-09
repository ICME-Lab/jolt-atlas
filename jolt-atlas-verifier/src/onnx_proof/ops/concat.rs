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
            low_to_high_point, reduced_output_setup, AtlasRelation, EngineRelation,
            RelationDerived, RelationExpr, RelationId, RelationOpening, RelationSetup,
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
use atlas_trace::{op::Concat, Model, Op};
use atlas_transcript::Transcript;
use jolt_claims::{derived, opening, Expr, SymbolicSumcheck};
use jolt_field::Ring;

crate::impl_plan_op!(Concat, ConcatRelation);

impl OpPlan for Concat {
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
                relation: OpRelation::Concat(ConcatRelation::for_node(model, node.clone())?),
            },
        ]))
    }
}

/// A concat node with its axis and its operands' unpadded dims, validated against the concat
/// shape rules.
pub struct ConcatShape {
    node: ComputationNode,
    input_raw_dims: Vec<Vec<usize>>,
    axis: usize,
}

fn unsupported(what: impl std::fmt::Display) -> ProofVerifyError {
    ProofVerifyError::Malformed(format!("model: concat: {what}"))
}

/// `Concat` as a dense relation over the largest operand's padded domain:
///
/// ```text
/// input claim   out(r₀)                (the reduced output opening)
/// output claim  Σ_i selector_i(r′) · in_i(r′[..n_i])
/// ```
///
/// Each operand is lifted to the largest domain by repeating it across the low variables, so
/// its claim is its own MLE at the first `n_i` coordinates of `r′`; `selector_i` places
/// `eq(r₀, ·)` at the output cell of each of its elements.
///
/// **S ⊆ R.** A concatenation copies each operand unchanged into its range of the axis, so
/// `out(r₀) = Σ_i Σ_x selector_i(x)·in_i(x)`.
pub struct ConcatRelation {
    shape: ConcatShape,
}

impl ConcatRelation {
    /// The relation of a concat `node` of `model`. `Err` if the node is not a concat or its
    /// dims break the concat shape rules.
    pub fn for_node(model: &Model, node: ComputationNode) -> Result<Self, ProofVerifyError> {
        let Op::Concat(op) = &node.operator else {
            return Err(unsupported("expected a Concat operator"));
        };
        let axis = op.axis;
        if node.inputs.iter().any(|&input| input >= model.num_nodes()) {
            return Err(unsupported("an operand is not a node"));
        }
        let input_raw_dims: Vec<Vec<usize>> = model
            .operand_views(&node)
            .iter()
            .map(|input| input.output_dims.clone())
            .collect();
        validate_concat_shapes(
            &input_raw_dims
                .iter()
                .map(|dims| dims.as_slice())
                .collect::<Vec<_>>(),
            &node.output_dims,
            axis,
        )?;
        Ok(Self::new(ConcatShape {
            node,
            input_raw_dims,
            axis,
        }))
    }

    fn input_num_vars(&self) -> Vec<usize> {
        self.shape
            .input_raw_dims
            .iter()
            .map(|dims| padded_domain_len(dims).log_2())
            .collect()
    }
}

impl SymbolicSumcheck for ConcatRelation {
    atlas_relation_types!();
    type Shape = ConcatShape;

    fn new(shape: ConcatShape) -> Self {
        Self { shape }
    }

    fn id() -> RelationId {
        RelationId::Concat
    }

    fn rounds(&self) -> usize {
        self.input_num_vars().into_iter().max().unwrap_or(0)
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_expression<F: Ring>(&self) -> RelationExpr<F> {
        opening(RelationOpening::ReducedNodeOutput(self.shape.node.idx))
    }

    fn output_expression<F: Ring>(&self) -> RelationExpr<F> {
        self.produced()
            .into_iter()
            .enumerate()
            .fold(Expr::zero(), |sum, (i, id)| {
                sum + derived(RelationDerived::Table(i)) * opening(RelationOpening::Id(id))
            })
    }
}

impl Produces for ConcatRelation {
    fn produced(&self) -> Vec<OpeningId> {
        let ids = OpeningIdBuilder::new(&self.shape.node);
        (0..self.shape.input_raw_dims.len())
            .map(|i| ids.nodeio(Target::Input(i)))
            .collect()
    }
}

impl<F: JoltField> Relation<F> for ConcatRelation {
    fn opening_points(
        &self,
        _setup: &RelationSetup<F>,
        challenges: &[F],
    ) -> Result<Vec<Vec<F>>, ProofVerifyError> {
        let point = low_to_high_point(challenges);
        self.input_num_vars()
            .into_iter()
            .map(|n| {
                point.get(..n).map(<[F]>::to_vec).ok_or_else(|| {
                    ProofVerifyError::Malformed(format!(
                        "concat: {} challenges for an operand of {n} variables",
                        point.len()
                    ))
                })
            })
            .collect()
    }

    fn table(&self, index: usize, setup: &RelationSetup<F>) -> Result<Vec<F>, ProofVerifyError> {
        build_concat_selector(
            &self.shape.input_raw_dims,
            &self.shape.node.output_dims,
            self.shape.axis,
            index,
            &setup.incoming_point,
            self.rounds(),
        )
    }
}

impl<F: JoltField> SetupRelation<F> for ConcatRelation {
    fn setup<T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut T,
    ) -> Result<RelationSetup<F>, ProofVerifyError> {
        reduced_output_setup(accumulator, &self.shape.node)
    }
}

impl AtlasRelation for ConcatRelation {
    type Spec = Concat;

    fn node(&self) -> &ComputationNode {
        &self.shape.node
    }
}

impl<F: JoltField> EngineRelation<F> for ConcatRelation {}

fn padded_domain_len(dims: &[usize]) -> usize {
    dims.iter().map(|dim| dim.next_power_of_two()).product()
}

fn build_concat_selector<F: JoltField>(
    raw_input_dims: &[Vec<usize>],
    output_raw_dims: &[usize],
    axis: usize,
    input_idx: usize,
    r_output: &[F],
    max_input_num_vars: usize,
) -> Result<Vec<F>, ProofVerifyError> {
    let input_raw_dims = raw_input_dims
        .get(input_idx)
        .ok_or_else(|| unsupported(format!("input_idx {input_idx} out of range")))?;
    let input_raw_len: usize = input_raw_dims.iter().product();
    let input_padded_dims: Vec<usize> = input_raw_dims
        .iter()
        .map(|dim| dim.next_power_of_two())
        .collect();
    let input_domain_len: usize = input_padded_dims.iter().product();
    let input_num_vars = input_domain_len.log_2();

    let output_padded_dims: Vec<usize> = output_raw_dims
        .iter()
        .map(|dim| dim.next_power_of_two())
        .collect();
    let output_domain_len: usize = output_padded_dims.iter().product();
    let output_num_vars = output_domain_len.log_2();
    if output_num_vars != r_output.len() {
        return Err(ProofVerifyError::Malformed(format!(
            "concat selector: output point has {} coordinates, its padded domain {output_num_vars}",
            r_output.len()
        )));
    }
    let shift = max_input_num_vars
        .checked_sub(input_num_vars)
        .ok_or_else(|| unsupported("an operand larger than the sumcheck domain"))?;

    let max_domain_len = 1usize << max_input_num_vars;
    let axis_offset = axis_offset(raw_input_dims, input_idx, axis)?;
    let mut selector = vec![F::zero(); max_domain_len];
    let eq_evals = EqPolynomial::evals(r_output);

    for linear_idx in 0..input_raw_len {
        let input_coord = linear_to_coord(linear_idx, input_raw_dims);
        let mut output_coord = input_coord.clone();
        *output_coord
            .get_mut(axis)
            .ok_or_else(|| unsupported(format!("axis {axis} out of range")))? += axis_offset;

        let input_index = coord_to_linear(&input_coord, &input_padded_dims)?;
        let output_index = coord_to_linear(&output_coord, &output_padded_dims)?;
        let full_index = input_index << shift;
        let (Some(cell), Some(eq)) = (selector.get_mut(full_index), eq_evals.get(output_index))
        else {
            return Err(unsupported("an element outside the padded domain"));
        };
        *cell = *eq;
    }

    Ok(selector)
}

/// Validates concat shape rules.
fn validate_concat_shapes(
    inputs_dims: &[&[usize]],
    out_dims: &[usize],
    axis: usize,
) -> Result<(), ProofVerifyError> {
    if inputs_dims.is_empty() {
        return Err(unsupported("expects at least one input"));
    }
    let rank = out_dims.len();
    if axis >= rank {
        return Err(unsupported(format!(
            "axis {axis} out of range for rank {rank}"
        )));
    }
    if inputs_dims.iter().any(|dims| dims.len() != rank) {
        return Err(unsupported("input rank mismatch"));
    }

    for (dim, &out_dim) in out_dims.iter().enumerate() {
        // Every input has the output's rank, so `dim` is one of its axes.
        let input_dims = inputs_dims.iter().filter_map(|dims| dims.get(dim).copied());
        if dim == axis {
            if out_dim != input_dims.sum::<usize>() {
                return Err(unsupported(
                    "output axis dimension must equal sum of input dimensions",
                ));
            }
        } else if input_dims.clone().any(|input_dim| input_dim != out_dim) {
            return Err(unsupported("non-axis dimensions must match between inputs"));
        }
    }
    Ok(())
}

/// Returns the axis offset (in output coordinates) for the given input.
fn axis_offset(
    raw_input_dims: &[Vec<usize>],
    input_idx: usize,
    axis: usize,
) -> Result<usize, ProofVerifyError> {
    raw_input_dims
        .iter()
        .take(input_idx)
        .map(|dims| {
            dims.get(axis)
                .copied()
                .ok_or_else(|| unsupported(format!("axis {axis} out of range")))
        })
        .sum()
}
