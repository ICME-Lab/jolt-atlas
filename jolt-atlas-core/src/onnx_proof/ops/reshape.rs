//! Reshape proof using an eq-selector over the padded input domain.
//!
//! For each raw input element `t`, we compute the padded output index that the
//! same element occupies after reshape and place `eq(r_output, mapped_index(t))`
//! at the corresponding padded input position. The resulting selector MLE
//! satisfies:
//!
//! `out(r_output) = sum_i selector(i) * input(i)`.

use atlas_trace::{op::Reshape, trace::LayerData, Model, Trace};

use crate::node_view::{ComputationNode, ModelView};
use crate::onnx_proof::trace_view::TraceView;
use crate::utils::dims::{coord_to_linear, linear_to_coord};
use crate::{
    atlas_relation_types, impl_standard_sumcheck_proof_api,
    onnx_proof::{
        relation::{
            reduced_output_setup, AtlasRelation, EngineRelation, RelationDerived, RelationExpr,
            RelationId, RelationOpening, RelationSetup,
        },
        ProofId, ProofType, Prover, Verifier,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use jolt_claims::{derived, opening, SymbolicSumcheck};
use jolt_field::Ring;
use joltworks::{
    field::{ChallengeFieldOps, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{OpeningAccumulator, OpeningId},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};

impl_standard_sumcheck_proof_api!(Reshape, model_relation ReshapeRelation);

/// Build a reshape selector vector over the padded input domain.
///
/// For each raw input position `t`, this computes the padded input index
/// corresponding to `t`, the padded output index that the same raw element maps
/// to after reshape, and stores `eq(r_output, mapped_output_index)` at that
/// padded input position. Padding cells remain zero.
///
/// This gives a selector MLE such that:
/// `out(r_output) = Σ_i selector(i) * input(i)`.
pub(crate) fn build_reshape_selectors<F: JoltField + ChallengeFieldOps<F>>(
    input_raw_dims: &[usize],
    output_raw_dims: &[usize],
    r_output: &[F],
) -> Vec<F> {
    let input_raw_len: usize = input_raw_dims.iter().product();
    let output_raw_len: usize = output_raw_dims.iter().product();
    assert_eq!(
        input_raw_len, output_raw_len,
        "Reshape selector requires equal raw element counts"
    );

    let input_padded_dims: Vec<usize> = input_raw_dims
        .iter()
        .map(|d| d.next_power_of_two())
        .collect();
    let output_padded_dims: Vec<usize> = output_raw_dims
        .iter()
        .map(|d| d.next_power_of_two())
        .collect();
    let input_padded_len: usize = input_padded_dims.iter().product();
    let output_padded_len: usize = output_padded_dims.iter().product();
    assert_eq!(
        input_padded_len, output_padded_len,
        "Reshape selector requires equal padded domain sizes"
    );

    // Selector values are zero on padding cells by default.
    let mut selector = vec![F::zero(); input_padded_len];
    let eq_evals = EqPolynomial::evals(r_output);

    for t in 0..input_raw_len {
        let input_coord = linear_to_coord(t, input_raw_dims);
        let output_coord = linear_to_coord(t, output_raw_dims);
        let input_padded_index = coord_to_linear(&input_coord, &input_padded_dims);
        let output_padded_index = coord_to_linear(&output_coord, &output_padded_dims);
        selector[input_padded_index] = eq_evals[output_padded_index];
    }

    selector
}

/// A reshape node with its operand's and its own unpadded dims.
pub struct ReshapeShape {
    node: ComputationNode,
    input_raw_dims: Vec<usize>,
}

/// `Reshape` as a dense relation over the padded input domain:
///
/// ```text
/// input claim   out(r₀)                (the reduced output opening)
/// output claim  selector(r′) · in(r′)
/// ```
///
/// where `selector` ([`build_reshape_selectors`]) places `eq(r₀, ·)` at the padded output cell
/// each input element lands in.
///
/// **S ⊆ R.** A reshape moves raw elements in row-major order without changing them, so
/// `out(r₀) = Σ_i selector(i)·in(i)` over the input domain.
pub struct ReshapeRelation {
    shape: ReshapeShape,
}

impl ReshapeRelation {
    /// The relation of a reshape `node` of `model`.
    pub fn for_node(model: &Model, node: ComputationNode) -> Self {
        let input_raw_dims = model.view(node.inputs[0]).output_dims;
        Self::new(ReshapeShape {
            node,
            input_raw_dims,
        })
    }
}

impl SymbolicSumcheck for ReshapeRelation {
    atlas_relation_types!();
    type Shape = ReshapeShape;

    fn new(shape: ReshapeShape) -> Self {
        Self { shape }
    }

    fn id() -> RelationId {
        RelationId::Reshape
    }

    fn rounds(&self) -> usize {
        self.shape.node.pow2_padded_num_output_elements().log_2()
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

impl AtlasRelation for ReshapeRelation {
    type Spec = Reshape;

    fn node(&self) -> &ComputationNode {
        &self.shape.node
    }

    fn setup<F: JoltField, T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut T,
    ) -> RelationSetup<F> {
        reduced_output_setup(accumulator, &self.shape.node)
    }

    fn produced(&self) -> Vec<OpeningId> {
        vec![OpeningIdBuilder::new(&self.shape.node).nodeio(Target::Input(0))]
    }

    fn table<F: JoltField>(&self, index: usize, setup: &RelationSetup<F>) -> Vec<F> {
        assert_eq!(index, 0, "a reshape has one selector");
        build_reshape_selectors(
            &self.shape.input_raw_dims,
            &self.shape.node.output_dims,
            &setup.incoming_point,
        )
    }
}

impl EngineRelation for ReshapeRelation {
    fn witnesses<F: JoltField>(
        &self,
        trace: &Trace,
        _setup: &RelationSetup<F>,
    ) -> Vec<MultilinearPolynomial<F>> {
        let LayerData { operands, .. } = TraceView::layer_data(trace, &self.shape.node);
        vec![MultilinearPolynomial::from(operands[0].padded_pow2())]
    }
}
