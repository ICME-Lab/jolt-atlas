use crate::node_view::{ComputationNode, ModelView};
use crate::onnx_proof::relation::Produces;
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
            reduced_output_setup, AtlasRelation, EngineRelation, RelationDerived, RelationExpr,
            RelationId, RelationOpening, RelationSetup,
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
use atlas_trace::{op::Slice, trace::LayerData, Model, Op, Trace};
use atlas_transcript::Transcript;
use jolt_claims::{derived, opening, SymbolicSumcheck};
use jolt_field::Ring;

impl_standard_sumcheck_proof_api!(Slice, model_relation SliceRelation);

/// A slice node with its operand's unpadded dims.
pub struct SliceShape {
    node: ComputationNode,
    input_raw_dims: Vec<usize>,
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
    /// The relation of a slice `node` of `model`.
    pub fn for_node(model: &Model, node: ComputationNode) -> Self {
        let input_raw_dims = model.view(node.inputs[0]).output_dims;
        Self::new(SliceShape {
            node,
            input_raw_dims,
        })
    }

    fn window(&self) -> (usize, usize, usize) {
        let Op::Slice(op) = &self.shape.node.operator else {
            panic!("Expected Slice operator")
        };
        (op.axis, op.start, op.end)
    }
}

impl SymbolicSumcheck for SliceRelation {
    atlas_relation_types!();
    type Shape = SliceShape;

    fn new(shape: SliceShape) -> Self {
        let relation = Self { shape };
        let (axis, start, end) = relation.window();
        validate_slice_shapes(
            &relation.shape.input_raw_dims,
            &relation.shape.node.output_dims,
            axis,
            start,
            end,
        );
        relation
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
        derived(RelationDerived::Table(0)) * opening(RelationOpening::Id(self.produced()[0]))
    }
}

impl Produces for SliceRelation {
    fn produced(&self) -> Vec<OpeningId> {
        vec![OpeningIdBuilder::new(&self.shape.node).nodeio(Target::Input(0))]
    }
}

impl<F: JoltField> Relation<F> for SliceRelation {
    fn table(&self, index: usize, setup: &RelationSetup<F>) -> Vec<F> {
        assert_eq!(index, 0, "a slice has one selector");
        let (axis, start, _) = self.window();
        build_slice_selector(
            &self.shape.input_raw_dims,
            &self.shape.node.output_dims,
            axis,
            start,
            &setup.incoming_point,
        )
    }
}

impl<F: JoltField> SetupRelation<F> for SliceRelation {
    fn setup<T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut T,
    ) -> RelationSetup<F> {
        reduced_output_setup(accumulator, &self.shape.node)
    }
}

impl AtlasRelation for SliceRelation {
    type Spec = Slice;

    fn node(&self) -> &ComputationNode {
        &self.shape.node
    }
}

impl<F: JoltField> EngineRelation<F> for SliceRelation {
    fn witnesses(&self, trace: &Trace, _setup: &RelationSetup<F>) -> Vec<MultilinearPolynomial<F>> {
        let LayerData { operands, .. } = TraceView::layer_data(trace, &self.shape.node);
        vec![MultilinearPolynomial::from(operands[0].padded_pow2())]
    }
}

fn build_slice_selector<F: JoltField>(
    input_raw_dims: &[usize],
    output_raw_dims: &[usize],
    axis: usize,
    start: usize,
    r_output: &[F],
) -> Vec<F> {
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
    assert_eq!(
        output_num_vars,
        r_output.len(),
        "Slice selector expects output challenge length to match output padded domain"
    );

    let mut selector = vec![F::zero(); input_domain_len];
    let eq_evals = EqPolynomial::evals(r_output);
    for output_linear_idx in 0..output_raw_dims.iter().product() {
        let output_coord = linear_to_coord(output_linear_idx, output_raw_dims);
        let mut input_coord = output_coord.clone();
        input_coord[axis] += start;

        let input_index = coord_to_linear(&input_coord, &input_padded_dims);
        let output_index = coord_to_linear(&output_coord, &output_padded_dims);
        selector[input_index] = eq_evals[output_index];
    }

    assert!(
        input_raw_len >= output_raw_dims.iter().product(),
        "Slice output raw length exceeds input raw length"
    );

    selector
}

fn validate_slice_shapes(
    input_dims: &[usize],
    output_dims: &[usize],
    axis: usize,
    start: usize,
    end: usize,
) {
    assert_eq!(input_dims.len(), output_dims.len(), "Slice rank mismatch");
    assert!(axis < input_dims.len(), "Slice axis out of range");
    assert!(start <= end, "Slice start must be <= end");
    assert!(end <= input_dims[axis], "Slice end out of bounds");

    for dim in 0..input_dims.len() {
        let expected = if dim == axis {
            end - start
        } else {
            input_dims[dim]
        };
        assert_eq!(
            output_dims[dim], expected,
            "Slice output shape mismatch at axis {dim}"
        );
    }
}
