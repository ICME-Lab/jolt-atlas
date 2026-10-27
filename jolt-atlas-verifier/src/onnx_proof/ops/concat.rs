use crate::node_view::{ComputationNode, ModelView};
use crate::onnx_proof::relation::{Relation, SetupRelation};
use crate::onnx_proof::trace_view::TraceView;
use crate::utils::{
    dims::{coord_to_linear, linear_to_coord},
    opening_access::{OpeningIdBuilder, Target},
};
use crate::{
    atlas_relation_types, impl_standard_sumcheck_proof_api,
    onnx_proof::{
        relation::{
            low_to_high_point, reduced_output_setup, AtlasRelation, EngineRelation,
            RelationDerived, RelationExpr, RelationId, RelationOpening, RelationSetup,
        },
        ProofId, ProofType, Verifier,
    },
};
use atlas_algebra::{
    field::JoltField,
    poly::{eq_poly::EqPolynomial, multilinear_polynomial::MultilinearPolynomial},
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_trace::{op::Concat, trace::LayerData, Model, Op, Trace};
use atlas_transcript::Transcript;
use jolt_claims::{derived, opening, Expr, SymbolicSumcheck};
use jolt_field::Ring;

impl_standard_sumcheck_proof_api!(Concat, model_relation ConcatRelation);

/// A concat node with its operands' unpadded dims.
pub struct ConcatShape {
    node: ComputationNode,
    input_raw_dims: Vec<Vec<usize>>,
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
    /// The relation of a concat `node` of `model`.
    pub fn for_node(model: &Model, node: ComputationNode) -> Self {
        let input_raw_dims = model
            .operand_views(&node)
            .iter()
            .map(|input| input.output_dims.clone())
            .collect();
        Self::new(ConcatShape {
            node,
            input_raw_dims,
        })
    }

    fn axis(&self) -> usize {
        let Op::Concat(op) = &self.shape.node.operator else {
            panic!("Expected Concat operator")
        };
        op.axis
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
        let relation = Self { shape };
        validate_concat_shapes(
            &relation
                .shape
                .input_raw_dims
                .iter()
                .map(|dims| dims.as_slice())
                .collect::<Vec<_>>(),
            &relation.shape.node.output_dims,
            relation.axis(),
        );
        relation
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

impl Relation for ConcatRelation {
    fn produced(&self) -> Vec<OpeningId> {
        let ids = OpeningIdBuilder::new(&self.shape.node);
        (0..self.shape.input_raw_dims.len())
            .map(|i| ids.nodeio(Target::Input(i)))
            .collect()
    }

    fn opening_points<F: JoltField>(
        &self,
        _setup: &RelationSetup<F>,
        challenges: &[F],
    ) -> Vec<Vec<F>> {
        let point = low_to_high_point(challenges);
        self.input_num_vars()
            .into_iter()
            .map(|n| point[..n].to_vec())
            .collect()
    }

    fn table<F: JoltField>(&self, index: usize, setup: &RelationSetup<F>) -> Vec<F> {
        build_concat_selector(
            &self.shape.input_raw_dims,
            &self.shape.node.output_dims,
            self.axis(),
            index,
            &setup.incoming_point,
            self.rounds(),
        )
    }
}

impl SetupRelation for ConcatRelation {
    fn setup<F: JoltField, T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut T,
    ) -> RelationSetup<F> {
        reduced_output_setup(accumulator, &self.shape.node)
    }
}

impl AtlasRelation for ConcatRelation {
    type Spec = Concat;

    fn node(&self) -> &ComputationNode {
        &self.shape.node
    }
}

impl EngineRelation for ConcatRelation {
    fn witnesses<F: JoltField>(
        &self,
        trace: &Trace,
        _setup: &RelationSetup<F>,
    ) -> Vec<MultilinearPolynomial<F>> {
        let LayerData { operands, .. } = TraceView::layer_data(trace, &self.shape.node);
        let max_num_vars = self.rounds();
        operands
            .iter()
            .map(|operand| {
                let padded = operand.padded_pow2();
                let extended =
                    extend_input_to_max_domain(padded.data(), padded.len().log_2(), max_num_vars);
                MultilinearPolynomial::from(extended)
            })
            .collect()
    }
}

fn padded_domain_len(dims: &[usize]) -> usize {
    dims.iter().map(|dim| dim.next_power_of_two()).product()
}

fn extend_input_to_max_domain<T: Copy>(
    input_padded: &[T],
    input_num_vars: usize,
    max_num_vars: usize,
) -> Vec<T> {
    let pad_repeat = 1usize << (max_num_vars - input_num_vars);
    let mut extended = Vec::with_capacity(input_padded.len() * pad_repeat);
    for value in input_padded {
        for _ in 0..pad_repeat {
            extended.push(*value);
        }
    }
    extended
}

fn build_concat_selector<F: JoltField>(
    raw_input_dims: &[Vec<usize>],
    output_raw_dims: &[usize],
    axis: usize,
    input_idx: usize,
    r_output: &[F],
    max_input_num_vars: usize,
) -> Vec<F> {
    let input_raw_dims = raw_input_dims
        .get(input_idx)
        .unwrap_or_else(|| panic!("Concat input_idx {input_idx} out of range"));
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
    assert_eq!(
        output_num_vars,
        r_output.len(),
        "Concat selector expects output challenge length to match output padded domain"
    );

    let max_domain_len = 1usize << max_input_num_vars;
    let axis_offset = axis_offset(raw_input_dims, input_idx, axis);
    let mut selector = vec![F::zero(); max_domain_len];
    let eq_evals = EqPolynomial::evals(r_output);

    for linear_idx in 0..input_raw_len {
        let input_coord = linear_to_coord(linear_idx, input_raw_dims);
        let mut output_coord = input_coord.clone();
        output_coord[axis] += axis_offset;

        let input_index = coord_to_linear(&input_coord, &input_padded_dims);
        let output_index = coord_to_linear(&output_coord, &output_padded_dims);
        let full_index = input_index << (max_input_num_vars - input_num_vars);
        selector[full_index] = eq_evals[output_index];
    }

    selector
}

/// Validates concat shape rules.
fn validate_concat_shapes(inputs_dims: &[&[usize]], out_dims: &[usize], axis: usize) {
    assert!(!inputs_dims.is_empty(), "Concat expects at least one input");
    let rank = out_dims.len();
    for dims in inputs_dims {
        assert_eq!(dims.len(), rank, "Concat input rank mismatch");
    }

    for dim in 0..out_dims.len() {
        if dim == axis {
            let sum: usize = inputs_dims.iter().map(|dims| dims[dim]).sum();
            assert_eq!(
                out_dims[dim], sum,
                "Concat output axis dimension must equal sum of input dimensions"
            );
        } else {
            let out_dim = out_dims[dim];
            for dims in inputs_dims {
                assert_eq!(
                    dims[dim], out_dim,
                    "Concat non-axis dimensions must match between inputs"
                );
            }
        }
    }
}

/// Returns the axis offset (in output coordinates) for the given input.
fn axis_offset(raw_input_dims: &[Vec<usize>], input_idx: usize, axis: usize) -> usize {
    raw_input_dims
        .iter()
        .take(input_idx)
        .map(|dims| dims[axis])
        .sum()
}
